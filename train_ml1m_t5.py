"""
train_ml1m_t5.py

独立训练脚本：将 ML-1M 数据集微调到一个 T5（t5-small 默认）用于推荐任务。
主要目标：保证 Prompt 格式与 `src/dual_memory.py::_build_prompt` 和
`evaluate_datasets.py` 中使用的 prompt 完全一致。

用法示例:
    python train_ml1m_t5.py --output_dir models/MY_ML1M_sequential_v1 --epochs 3 --batch_size 16

此脚本独立于 OpenP5-main 中的训练脚本，仅参考其数据和 prompt 约定。
"""
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import pandas as pd
import time
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/ML1M')
    parser.add_argument('--ml1m_raw_dir', type=str, default='data/ml-1m', help='原始 ML-1M 文件目录，包含 ratings.dat users.dat movies.dat')
    parser.add_argument('--output_dir', type=str, default='models/MY_ML1M_sequential_v1')
    parser.add_argument('--t5_model', type=str, default=None, help='本地或远程 t5 模型标识（默认使用 t5-small）')
    parser.add_argument('--add_item_tokens', action='store_true', default=True, help='是否将所有 item_{id} 注入 tokenizer')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Accumulate gradients N steps before optimizer.step() to simulate larger batch size')
    parser.add_argument('--fp16', action='store_true', default=False, help='Use mixed precision training (torch.cuda.amp) when available')
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_target_length', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--save_state_dict_pt', action='store_true', help='同时保存为 .pt state_dict 以兼容 model_wrapper.py')
    parser.add_argument('--sample_debug', action='store_true', help='使用小样本进行快速调试')
    # Retrain (gold) options: exclude forget interactions
    parser.add_argument('--exclude_forget_samples', action='store_true', help='从训练集中剔除遗忘请求中的交互 (用于训练 gold 模型)')
    parser.add_argument('--forget_file', type=str, default='results/forget_samples_subset.json', help='遗忘请求文件路径')
    return parser.parse_args()


# Prompt format MUST match dual_memory._build_prompt and evaluate_datasets.py
def build_prompt_for_user(mapped_user_id: str, history: List[str]) -> str:
    # history elements are mapped_item_id strings (already mapped)
    history_str = ' '.join(f'item_{it}' for it in history[-20:]) if history else '<empty>'
    return f"User {mapped_user_id} recent history: {history_str}. Recommend next item."


class ML1MDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: T5Tokenizer, max_input_length: int = 256, max_target_length: int = 8):
        # df should have columns: mapped_user_id, mapped_item_id, timestamp
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Build per-user histories ordered by timestamp
        self.user_histories: Dict[str, List[str]] = {}
        grouped = df.sort_values('timestamp').groupby('mapped_user_id')
        for uid, g in grouped:
            self.user_histories[str(uid)] = g['mapped_item_id'].astype(str).tolist()

        # Create training pairs: for each user, for each position t produce (history up to t, target=item_at_t)
        self.samples: List[Dict] = []
        for uid, hist in self.user_histories.items():
            if len(hist) < 2:
                continue
            for t in range(1, len(hist)):
                history = hist[:t]
                target = hist[t]
                self.samples.append({'user_id': uid, 'history': history, 'target': target})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        prompt = build_prompt_for_user(s['user_id'], s['history'])
        target_text = f"item_{s['target']}"
        return prompt, target_text


def collate_fn(batch, tokenizer: T5Tokenizer, max_input_length: int, max_target_length: int):
    prompts, targets = zip(*batch)
    inputs = tokenizer(list(prompts), padding=True, truncation=True, max_length=max_input_length, return_tensors='pt')
    labels = tokenizer(list(targets), padding=True, truncation=True, max_length=max_target_length, return_tensors='pt')
    labels_ids = labels.input_ids.clone()
    labels_ids[labels_ids == tokenizer.pad_token_id] = -100
    batch_dict = {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': labels_ids,
    }
    return batch_dict


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    # ensure logging is configured
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.data_dir)
    inter_file = data_dir / 'ml-1m.inter'
    user_index_file = data_dir / 'user_indexing.txt'

    # If standard inter file not present but raw ML-1M is available, preprocess it
    raw_dir = Path(args.ml1m_raw_dir)
    ratings_dat = raw_dir / 'ratings.dat'
    users_dat = raw_dir / 'users.dat'
    movies_dat = raw_dir / 'movies.dat'
    readme = raw_dir / 'README'

    if not inter_file.exists():
        if ratings_dat.exists():
            logging.info(f"Found raw ML-1M at {raw_dir}, generating {inter_file} and {user_index_file}...")
            data_dir.mkdir(parents=True, exist_ok=True)
            # Convert ratings.dat (UserID::MovieID::Rating::Timestamp) -> tab-separated inter file
            with open(ratings_dat, 'r', encoding='latin-1') as fin, open(inter_file, 'w', encoding='utf-8') as fout:
                for line in fin:
                    parts = line.strip().split('::')
                    if len(parts) != 4:
                        continue
                    user_id, item_id, rating, timestamp = parts
                    fout.write('\t'.join([user_id, item_id, rating, timestamp]) + '\n')
            # Create identity user_indexing.txt mapping original->mapped (simple identity mapping)
            users = set()
            with open(inter_file, 'r', encoding='utf-8') as fh:
                for l in fh:
                    u = l.split('\t', 1)[0].strip()
                    users.add(u)
            with open(user_index_file, 'w', encoding='utf-8') as fidx:
                for u in sorted(users, key=lambda x: int(x)):
                    fidx.write(f"{u} {u}\n")
            logging.info(f"Preprocessing complete: wrote {inter_file} ({sum(1 for _ in open(inter_file))} lines) and {user_index_file} ({len(users)} users)")
        else:
            raise FileNotFoundError(f"Interaction file not found: {inter_file}. Also raw ratings.dat not found at {ratings_dat}.")

    # load mappings
    user_map = {line.strip().split()[0]: line.strip().split()[1] for line in open(user_index_file, 'r', encoding='utf-8')}

    # load interactions
    # ml-1m.inter may not have a header (we generate it from ratings.dat), so specify column names
    df = pd.read_csv(inter_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], dtype={'user_id': str, 'item_id': str, 'rating': str, 'timestamp': str})
    # strip whitespace in case of irregularities
    df['user_id'] = df['user_id'].astype(str).str.strip()
    df['item_id'] = df['item_id'].astype(str).str.strip()
    df['mapped_user_id'] = df['user_id'].map(user_map)
    df.dropna(subset=['mapped_user_id'], inplace=True)
    df['mapped_user_id'] = df['mapped_user_id'].astype(str)
    df['mapped_item_id'] = df['item_id'].astype(str)

    # split train/test 80/20 by time to avoid leakage
    df = df.sort_values('timestamp')
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()

    # Optionally remove forget interactions from train_df to produce a "gold" model
    if args.exclude_forget_samples and os.path.exists(args.forget_file):
        try:
            import json
            with open(args.forget_file, 'r', encoding='utf-8') as f:
                forget_samples = json.load(f)
            # Build a fast lookup: mapped_user_id -> set(mapped_item_id to remove)
            to_remove = {}
            for s in forget_samples:
                uid = str(s.get('user_id'))
                mapped_uid = user_map.get(uid)
                items = [str(x) for x in (s.get('suppression_targets') or [])]
                if mapped_uid and items:
                    to_remove.setdefault(mapped_uid, set()).update(items)
            if to_remove:
                before = len(train_df)
                mask = train_df.apply(
                    lambda r: not (str(r['mapped_user_id']) in to_remove and str(r['mapped_item_id']) in to_remove[str(r['mapped_user_id'])]),
                    axis=1
                )
                train_df = train_df[mask].copy()
                removed = before - len(train_df)
                logging.info(f"Excluding forget interactions from train: removed {removed} rows (from {before})")
            else:
                logging.info("No forget interactions matched for exclusion; proceeding with full train_df")
        except Exception as e:
            logging.warning(f"Failed to exclude forget interactions: {e}")

    # keep a copy of the full train set for token-injection purposes
    train_df_full = train_df.copy()

    if args.sample_debug:
        # sample from train split only for quick debugging (but keep full train_df for token injection)
        train_df = train_df.sample(n=min(2000, len(train_df)), random_state=args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Prefer explicit --t5_model, then env P5_T5_MODEL_DIR, then local hf_models/t5-small
    t5_source = args.t5_model or os.environ.get('P5_T5_MODEL_DIR') or 'hf_models/t5-small'
    logging.info(f"Loading tokenizer/model from: {t5_source}")
    tokenizer = T5Tokenizer.from_pretrained(t5_source)
    model = T5ForConditionalGeneration.from_pretrained(t5_source).to(device)

    # Optionally add all item tokens to tokenizer to make item_{id} likely one token
    if args.add_item_tokens:
        # use the full train set (not the sampled one) to collect items to inject
        unique_items = sorted(train_df_full['mapped_item_id'].astype(str).unique(), key=lambda x: int(x))
        item_tokens = [f"item_{it}" for it in unique_items]

        # Detect tokens that are missing or that would be tokenized into multiple sub-tokens
        to_add = []
        for t in item_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(t)
            except Exception:
                token_id = tokenizer.unk_token_id
            toks = tokenizer.tokenize(t)
            # If token_id equals unk OR tokenization splits into multiple tokens, schedule to add
            if token_id == tokenizer.unk_token_id or len(toks) != 1:
                to_add.append(t)

        if to_add:
            # Add in one batch
            n_added = tokenizer.add_tokens(to_add)
            logging.info(f"Added {n_added} item tokens to tokenizer (requested={len(to_add)}, total_train_items={len(item_tokens)})")
            # Resize model embeddings to accommodate new tokens.
            # The default HF resizing may initialize new embeddings via a multivariate-normal
            # using the old embedding covariance which can be slow for large vocab expansions.
            # We perform a fast, safe extension: allocate a new embedding matrix, copy old
            # weights, and initialize the new rows with a normal distribution.
            try:
                old_emb = model.get_input_embeddings()
                old_num, emb_dim = old_emb.weight.size()
                new_num = len(tokenizer)
                if new_num != old_num:
                    # allocate new embedding on same device as model parameters
                    model_dev = next(model.parameters()).device
                    new_emb = nn.Embedding(new_num, emb_dim).to(model_dev)
                    # copy existing weights (ensure same device)
                    new_emb.weight.data[:old_num].copy_(old_emb.weight.data.to(model_dev))
                    # initialize new rows with normal(mean=0, std=old_std or 0.02)
                    old_std = float(old_emb.weight.data.std()) if old_emb.weight.data.numel() > 0 else 0.02
                    new_emb.weight.data[old_num:].normal_(mean=0.0, std=old_std)
                    model.set_input_embeddings(new_emb)
                    # ensure lm_head/output embeddings are tied to shared embeddings and update config
                    try:
                        # For T5: tie lm_head weights to shared embeddings to keep output vocabulary aligned
                        if hasattr(model, 'lm_head') and model.lm_head is not None:
                            model.lm_head.weight = new_emb.weight
                        # update config vocab size
                        if hasattr(model, 'config'):
                            model.config.vocab_size = new_num
                    except Exception as _e:
                        logging.warning('Could not tie lm_head to new embeddings: %s', _e)
                    logging.info("Resized embeddings by manual fast extension: %d -> %d", old_num, new_num)
            except Exception as e:
                # fallback to HF helper if anything unexpected fails
                logging.warning("Fast embedding extension failed, falling back to HF resize_token_embeddings: %s", e)
                model.resize_token_embeddings(len(tokenizer))
        else:
            logging.info("No new item tokens to add; tokenizer seems to cover train items and tokenizes them as single tokens")

    # Ensure special tokens are present
    added = 0
    for token in ['<forget>', '<retain>']:
        if tokenizer.convert_tokens_to_ids(token) is None or tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id:
            added += tokenizer.add_tokens([token])
    if added > 0:
        # Use the same fast extension approach as above
        try:
            old_emb = model.get_input_embeddings()
            old_num, emb_dim = old_emb.weight.size()
            new_num = len(tokenizer)
            if new_num != old_num:
                model_dev = next(model.parameters()).device
                new_emb = nn.Embedding(new_num, emb_dim).to(model_dev)
                new_emb.weight.data[:old_num].copy_(old_emb.weight.data.to(model_dev))
                old_std = float(old_emb.weight.data.std()) if old_emb.weight.data.numel() > 0 else 0.02
                new_emb.weight.data[old_num:].normal_(mean=0.0, std=old_std)
                model.set_input_embeddings(new_emb)
                # tie lm_head/output embeddings to shared embeddings and update config
                try:
                    if hasattr(model, 'lm_head') and model.lm_head is not None:
                        model.lm_head.weight = new_emb.weight
                    if hasattr(model, 'config'):
                        model.config.vocab_size = new_num
                except Exception as _e:
                    logging.warning('Could not tie lm_head to new embeddings for special tokens: %s', _e)
                logging.info("Resized embeddings by manual fast extension for special tokens: %d -> %d", old_num, new_num)
        except Exception as e:
            logging.warning("Fast embedding extension for special tokens failed, falling back to HF resize: %s", e)
            model.resize_token_embeddings(len(tokenizer))

    dataset = ML1MDataset(train_df, tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, args.max_input_length, args.max_target_length), num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    use_amp = args.fp16 and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # Training loop with optional tqdm progress bars and ETA
    for epoch in range(args.epochs):
        total_loss = 0.0
        steps = 0
        epoch_start = time.time()

        iterator = loader
        use_tqdm = (tqdm is not None)
        if use_tqdm:
            iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")

        # gradient accumulation support: perform optimizer.step() every N steps
        accum_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps > 0 else 1
        for i, batch in enumerate(iterator, start=1):
            for k in ['input_ids', 'attention_mask', 'labels']:
                batch[k] = batch[k].to(device)
            # forward/backward with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = outputs.loss
                scaler.scale(loss / accum_steps).backward()
            else:
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs.loss
                (loss / accum_steps).backward()

            # step optimizer every accum_steps
            if (i % accum_steps) == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            cur_loss = float(loss.item())
            total_loss += cur_loss
            steps += 1

            if use_tqdm:
                # show the instantaneous loss and running average and effective batch size
                running_avg = total_loss / max(1, steps)
                eff_bs = args.batch_size * accum_steps
                iterator.set_postfix({'loss': f'{cur_loss:.4f}', 'avg': f'{running_avg:.4f}', 'eff_bs': eff_bs})

        avg = total_loss / max(1, steps)
        epoch_elapsed = time.time() - epoch_start
        remaining_epochs = args.epochs - (epoch + 1)
        est_remaining = remaining_epochs * epoch_elapsed
        print(f"Epoch {epoch+1}/{args.epochs} | avg_loss={avg:.6f} | elapsed={epoch_elapsed:.1f}s | est_remaining={est_remaining:.1f}s")

    # Save HF format
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved HuggingFace model & tokenizer to {out_dir}")

    # Post-training diagnostic: show tokenization and a few generated examples to verify item tokens
    try:
        sample_checks = ['item_1','item_12','item_123']
        print('\nPost-training tokenization checks:')
        for t in sample_checks:
            print(t, '->', tokenizer.tokenize(t), 'id=', tokenizer.convert_tokens_to_ids(t))

        # small generation sample (use cpu device if cuda not available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        tokenizer_device = device
        sample_prompt = build_prompt_for_user('1924', ['10','20','30'])
        inputs = tokenizer(sample_prompt, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_length=80, num_beams=5, num_return_sequences=5)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print('\nPost-training generation samples:')
        for d in decoded:
            print('  ', d)
    except Exception as e:
        logging.warning('Post-training diagnostic failed: %s', e)

    # Optional save .pt state dict for compatibility
    if args.save_state_dict_pt:
        pt_path = out_dir / (out_dir.name + '.pt')
        torch.save({'model_state_dict': model.state_dict()}, str(pt_path))
        print(f"Saved state_dict .pt to {pt_path}")


if __name__ == '__main__':
    main()
