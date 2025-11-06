# File: evaluate_datasets.py (Corrected Version)

#!/usr/bin/env python3
"""
P5推荐模型数据集划分评估
按照保留集和遗忘集分别评估指定模型性能

用法:
    python evaluate_datasets.py --model_path models/ML1M_sequential.pt
    python evaluate_datasets.py --model_path models/ML1M_sequential_unlearned.pt --output_suffix _after_unlearning
    python evaluate_datasets.py --original_model models/original.pt --unlearned_model models/unlearned.pt --compare

参数说明:
    --model_path          单个模型路径
    --original_model      原始模型路径（对比模式）
    --unlearned_model     遗忘后模型路径（对比模式）
    --compare             启用对比模式，同时评估原始和遗忘后模型
    --output_suffix       输出文件后缀
    --forget_ratio        遗忘集比例 (默认: 0.05)
    --eval_sample_size    评估样本大小 (默认: 50)
    --k_values            评估的K值列表 (默认: 10,20)
    --save_predictions    保存推荐结果
    --verbose             详细输出
"""

import os
import sys
import time
import json
import random
import re
import logging
import argparse
from types import SimpleNamespace
from collections import defaultdict
from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import math

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Local imports
from src.model_wrapper import P5ModelWrapper
from src.dual_memory import load_dual_memory_runtime
from src.p5_evaluator import P5RecommendationEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# Fallback stub for WISEUnlearningEditor when real implementation isn't imported here.
# The real implementation lives in src.dual_memory or other modules and will be used
# when initialize_model loads dual artifacts. This stub preserves expected attributes
# to allow static analysis and partial runtime flows that don't exercise editor behavior.
class WISEUnlearningEditor:
    def __init__(self, model, tokenizer=None, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.side_modules = {}
        self.router_classifier = None
        self.epsilon_threshold = 0.0
        self.router_feature_dataset = None
        self.router_input_dim = 1
        self.router_feature_mode = 'delta_l2'
        self.router_prob_threshold = None
        self.rescan_before_eval = False


class DatasetEvaluator:
    def __init__(self, args: argparse.Namespace):
        # Save original args
        self.args = args
        self.compare_mode = bool(getattr(args, 'compare', False))
        self.model_path = getattr(args, 'model_path', None)
        self.original_model_path = getattr(args, 'original_model', None)
        self.unlearned_model_path = getattr(args, 'unlearned_model', None)
        self.output_suffix = getattr(args, 'output_suffix', '') or ''
        self.forget_ratio = getattr(args, 'forget_ratio', 0.01)
        self.eval_sample_size = getattr(args, 'eval_sample_size', 50)
        k_vals = getattr(args, 'k_values', '10,20')
        if isinstance(k_vals, str):
            self.k_values = [int(x) for x in k_vals.split(',') if x.strip()]
        else:
            self.k_values = list(map(int, k_vals))
        self.eval_rescan_mode = getattr(args, 'eval_rescan_mode', 'auto')
        self.save_predictions = getattr(args, 'save_predictions', False)
        self.skip_all_users = getattr(args, 'skip_all_users', False)
        self.verbose = getattr(args, 'verbose', False)
        # control verbose per-user diagnostics (off by default)
        self.show_user_diagnostics = getattr(args, 'show_user_diagnostics', False)
        # control progress bar display (on by default)
        self.show_progress = not getattr(args, 'no_progress', False)
        self.dual_artifacts_path = getattr(args, 'dual_memory_artifacts', None)
        self.dual_threshold = getattr(args, 'dual_memory_threshold', None)
        # inference-time calibration and fallback controls
        self.base_temperature = float(getattr(args, 'base_temperature', 1.0))
        self.side_temperature = float(getattr(args, 'side_temperature', 1.2))
        self.use_entropy_fallback = bool(getattr(args, 'use_entropy_fallback', False))
        self.conf_fallback_threshold = float(getattr(args, 'conf_fallback_threshold', 0.85))
        self.min_unique_ratio = float(getattr(args, 'min_unique_ratio', 0.3))

        # runtime placeholders
        self.model_wrapper = None
        self.dual_runtime = None
        self.evaluator = None
        self.results = {}

        # target metrics (used by evaluator) - include recall to ensure Recall@K is computed
        self.target_metrics = []
        for k in self.k_values:
            self.target_metrics.extend([f'hit@{k}', f'ndcg@{k}', f'recall@{k}'])


        

    def initialize_model(self, model_path=None):
        """Centralized model loading shim.

        Previously evaluate_datasets implemented a large initialize_model which handled
        many checkpoint formats and dual-memory artifacts. That logic now lives in
        src.model_wrapper.P5ModelWrapper._load_model. Here we call the wrapper and
        attach the resulting model/tokenizer to the evaluator.
        """
        if model_path is None:
            model_path = self.model_path

        # If caller requested to load a dual-memory artifact directly, defer to the
        # specialized initializer which constructs a DualMemoryRuntime.
        if self.dual_artifacts_path:
            self.initialize_dual_runtime(self.dual_artifacts_path, self.dual_threshold)
            return

        logger.info("通过 P5ModelWrapper 加载模型 (centralized)")
        device = os.environ.get('P5_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')

        # If user passed a directory, try to locate a checkpoint inside or load as HF model dir
        if model_path and os.path.isdir(model_path):
            # common checkpoint candidates
            candidates = [
                os.path.join(model_path, os.path.basename(model_path) + '.pt'),
                #os.path.join(model_path, 'pytorch_model.bin'),
                #os.path.join(model_path, 'model.safetensors'),
            ]
            found = None
            for c in candidates:
                if os.path.exists(c):
                    found = c
                    break
            if found:
                logger.info("检测到模型目录，使用内部 checkpoint: %s", found)
                # prefer to load tokenizer from the directory where checkpoint was found
                checkpoint_path = found
                t5_local_dir = model_path
                # set model_path to the checkpoint for downstream loader
                model_path = checkpoint_path
            else:
                # attempt to load as HuggingFace model directory (contains config/tokenizer files)
                try:
                    logger.info("尝试将目录作为 HuggingFace 模型加载: %s", model_path)
                    tokenizer = T5Tokenizer.from_pretrained(model_path)
                    model = T5ForConditionalGeneration.from_pretrained(model_path)
                    # normalize device to torch.device and move model
                    torch_device = torch.device(device)
                    # Ensure model embeddings cover tokenizer vocab (avoid piece id out of range)
                    try:
                        tok_size = len(tokenizer)
                        emb_size = model.get_input_embeddings().weight.size(0)
                        if tok_size > emb_size:
                            logger.info("检测到 tokenizer vocab (%d) > model embeddings (%d), 重新调整模型 embedding 大小...", tok_size, emb_size)
                            model.resize_token_embeddings(tok_size)
                    except Exception as e:
                        logger.warning("在同步 tokenizer 与模型 vocab 大小时发生异常: %s", e)

                    model = model.to(torch_device)
                    wrapper = SimpleNamespace(model=model, tokenizer=tokenizer, model_path=model_path)
                    wrapper.device = torch_device
                    self.model_wrapper = wrapper
                    self.model = model
                    self.tokenizer = tokenizer
                    self.wise_editor = getattr(self.model, 'wise_editor_ref', None)
                    self.evaluator = P5RecommendationEvaluator()
                    return wrapper
                except Exception as e:
                    logger.warning("无法将目录作为 HuggingFace 模型直接加载: %s", e)

        # fallback: pass path (file) to P5ModelWrapper which expects a checkpoint file
        # If we detected a local model dir with tokenizer earlier, pass t5_local_dir
        try:
            wrapper = P5ModelWrapper(model_path=model_path, device=device, t5_local_dir=locals().get('t5_local_dir', None))
        except TypeError:
            # older wrapper signature fallback
            wrapper = P5ModelWrapper(model_path=model_path, device=device)
        self.model_wrapper = wrapper
        self.model = wrapper.model
        self.tokenizer = wrapper.tokenizer
        self.wise_editor = getattr(self.model, 'wise_editor_ref', None)
        # ensure evaluator exists
        self.evaluator = P5RecommendationEvaluator()
        return wrapper

    def _canonical_metric_name(self, metric: str) -> str:
        """Normalize metric names so aliases like 'hit_rate@10' map to 'hit@10'."""
        if not metric:
            return metric
        m = metric.lower()
        # normalize hit_rate@K -> hit@K
        if m.startswith('hit_rate@'):
            k = m.split('@', 1)[1]
            return f'hit@{k}'
        # already canonical if hit@, ndcg@, recall@
        if m.startswith('hit@') or m.startswith('ndcg@') or m.startswith('recall@'):
            return m
        return metric

    def load_and_split_data(self) -> Dict[str, Any]:
        """Load the ML1M interaction file and return a standardized data_info dict

        The returned dict contains keys expected by the evaluation pipeline:
         - train_df, test_df: pandas DataFrame with mapped_user_id and mapped_item_id
         - mappings: {'user_to_mapped': {...}, 'mapped_to_user': {...}}
         - all_users, retain_users, forget_users: lists of mapped user ids
         - eval_retain_users, eval_forget_users: lists (may be empty)
        """
        import pandas as pd

        data_dir = os.path.join(PROJECT_ROOT, 'data', 'ML1M')
        inter_file = os.path.join(data_dir, 'ml-1m.inter')
        user_index_file = os.path.join(data_dir, 'user_indexing.txt')

        if not os.path.exists(inter_file):
            raise FileNotFoundError(f"交互文件不存在: {inter_file}")
        if not os.path.exists(user_index_file):
            raise FileNotFoundError(f"用户索引文件不存在: {user_index_file}")

        # load mappings
        user_map = {line.strip().split()[0]: line.strip().split()[1] for line in open(user_index_file, 'r', encoding='utf-8')}
        mapped_to_user = {v: k for k, v in user_map.items()}

        # load interactions
        # ml-1m.inter may not have a header; ensure consistent column names
        df = pd.read_csv(inter_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], dtype={'user_id': str, 'item_id': str, 'rating': str, 'timestamp': str})
        # normalize whitespace
        df['user_id'] = df['user_id'].astype(str).str.strip()
        df['item_id'] = df['item_id'].astype(str).str.strip()
        # map users
        df['mapped_user_id'] = df['user_id'].map(user_map)
        df.dropna(subset=['mapped_user_id'], inplace=True)
        # ensure string typed ids
        df['mapped_user_id'] = df['mapped_user_id'].astype(str)
        df['mapped_item_id'] = df['item_id'].astype(str)
        # ensure rating and timestamp are numeric for comparisons and sorting
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(int)

        # split train/test 80/20 by time
        df = df.sort_values('timestamp')
        split_point = int(len(df) * 0.8)
        train_df = df.iloc[:split_point].copy()
        test_df = df.iloc[split_point:].copy()

        all_train_users = sorted(train_df['mapped_user_id'].unique().astype(str).tolist())

        # try to load precomputed forget/retain user lists if available
        forget_users = []
        retain_users = []
        forget_path = os.path.join(PROJECT_ROOT, 'results', 'forget_samples_subset.json')
        retain_path = os.path.join(PROJECT_ROOT, 'results', 'retain_samples_subset.json')
        if os.path.exists(forget_path):
            with open(forget_path, 'r', encoding='utf-8') as f:
                forget_samples = json.load(f)
            # map original ids to mapped ids using user_map
            forget_users = []
            for s in forget_samples:
                orig = str(s.get('user_id'))
                mapped = user_map.get(orig)
                if mapped:
                    forget_users.append(mapped)
        if os.path.exists(retain_path):
            with open(retain_path, 'r', encoding='utf-8') as f:
                retain_samples = json.load(f)
            retain_users = []
            for s in retain_samples:
                orig = str(s.get('user_id'))
                mapped = user_map.get(orig)
                if mapped:
                    retain_users.append(mapped)

        # fallback default sets
        if not retain_users:
            retain_users = [u for u in all_train_users if u not in set(forget_users)]

        # prepare evaluation subsets (empty by default)
        eval_retain_users = list(retain_users)[:self.eval_sample_size]
        eval_forget_users = list(forget_users)[:self.eval_sample_size]

        mappings = {'user_to_mapped': user_map, 'mapped_to_user': mapped_to_user}

        data_info = {
            'train_df': train_df,
            'test_df': test_df,
            'mappings': mappings,
            'all_users': all_train_users,
            'retain_users': retain_users,
            'forget_users': forget_users,
            'eval_retain_users': eval_retain_users,
            'eval_forget_users': eval_forget_users,
        }
        logger.info(f"数据加载完成: train={len(train_df)} rows, test={len(test_df)} rows, users={len(all_train_users)}")
        return data_info
        

    def initialize_dual_runtime(self, artifact_path: str, threshold_override: Optional[float] = None) -> None:
        """加载DualMemory推理运行时，用于主记忆+侧记忆+路由的完整模型。"""
        if not artifact_path:
            raise ValueError("Dual-memory artifact path is required")
        logger.info("🔧 初始化DualMemory运行时: %s", artifact_path)
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Dual-memory artifacts 文件不存在: {artifact_path}")

        device = os.environ.get('P5_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
        runtime = load_dual_memory_runtime(artifact_path, device=device, threshold=threshold_override)
        self.dual_runtime = runtime
        device_str = str(runtime.device)
        self.model_wrapper = SimpleNamespace(
            tokenizer=runtime.tokenizer,
            model=runtime.model,
            device=device_str,
        )
        self.evaluator = P5RecommendationEvaluator()

        self.dual_artifact_metadata = {}
        try:
            payload = torch.load(artifact_path, map_location='cpu')
            router_metrics = payload.get('router_metrics') or {}
            router_threshold = float(payload.get('router_threshold', runtime.threshold))
            epsilon_val = payload.get('router_feature_stats', {}).get('epsilon', runtime.epsilon)
            precision_val = router_metrics.get('precision')
            recall_val = router_metrics.get('recall')
            precision_str = f"{precision_val:.3f}" if isinstance(precision_val, (int, float)) else str(precision_val)
            recall_str = f"{recall_val:.3f}" if isinstance(recall_val, (int, float)) else str(recall_val)
            epsilon_float = float(epsilon_val) if epsilon_val is not None else float('nan')
            self.dual_artifact_metadata = {
                'router_metrics': router_metrics,
                'router_threshold': router_threshold,
                'epsilon': epsilon_val,
            }
            logger.info(
                "✅ Dual-memory路由指标: threshold=%.4f, epsilon=%.4f, precision=%s, recall=%s",
                router_threshold,
                epsilon_float,
                precision_str,
                recall_str,
            )
        except Exception as exc:
            logger.warning("⚠️ 无法解析Dual-memory路由统计信息: %s", exc)

        logger.info("🚀 Dual-memory runtime 准备完成 (device=%s, threshold=%.4f)", device_str, float(runtime.threshold))
            
    def evaluate_model_performance(self, data_info: Dict, user_set_info: Dict, set_name: str = "all", save_recs: bool = False) -> Dict[str, float]:
            """
            [最终修复版]
            评估模型性能。
            1. "保留效用" 使用测试集进行评估 (预测未来)。
            2. "遗忘效能" 直接使用已知的遗忘列表进行评估 (检验过去)。
            """
            if isinstance(user_set_info, dict):
                user_set = {str(uid) for uid in user_set_info.get('users', [])}
                # unlearning_requests is now a dict: {user_id: [item1, item2]}
                unlearning_requests = user_set_info.get('unlearning_requests', {})
            else:
                # Fallback for old format
                user_set = {str(uid) for uid in user_set_info}
                unlearning_requests = {}

            logger.info(f"📊 [最终修复版路由] 评估模型在 {set_name} 集上的性能 (用户数: {len(user_set)})...")
            
            train_df = data_info['train_df']
            test_df = data_info['test_df']

            # 1. [逻辑不变] "保留效用"的正确答案来自测试集 (衡量对未来的预测能力)
            gt_retained = defaultdict(set)
            for user_id in user_set:
                user_test_data = test_df[test_df['mapped_user_id'] == user_id]
                if not user_test_data.empty:
                    items = set(map(str, user_test_data[user_test_data['rating'] >= 4]['mapped_item_id'].tolist()))
                    gt_retained[user_id] = items

            # 2. [核心逻辑修改] "遗忘效能"的正确答案直接来自 unlearning_requests (衡量对过去交互的遗忘能力)
            gt_forgotten = defaultdict(set)
            if unlearning_requests:
                for user_id, items_to_forget in unlearning_requests.items():
                    # Only consider users who are part of the current evaluation set
                    if user_id in user_set:
                        gt_forgotten[user_id] = set(map(str, items_to_forget))

            predictions = []
            ground_truth_retain_list, ground_truth_forget_list = [], []
            processed_users_count = 0
            routed_to_side_count = 0
            
            dual_runtime = getattr(self, 'dual_runtime', None)
            
            user_list_to_eval = sorted(list(user_set))

            # choose an iterator: tqdm if available and requested, else plain list
            if self.show_progress and tqdm is not None:
                iterator = tqdm(user_list_to_eval, desc=f"Evaluating {set_name}", unit="user")
            else:
                iterator = user_list_to_eval

            for mapped_user_id in iterator:
                # Wrap the entire per-user processing in a try/except so a single failing user
                # doesn't abort the whole evaluation run. Any exceptions will be logged with
                # prompt and traceback for debugging.
                try:
                    user_history = train_df[train_df['mapped_user_id'] == mapped_user_id]['mapped_item_id'].astype(str).tolist()
                    if len(user_history) < 2:
                        # not enough history to evaluate
                        continue

                    # Use a consistent history prompt for both original and unlearned models
                    # The prompt should represent the state *before* the items were forgotten
                    items_to_forget_for_user = gt_forgotten.get(mapped_user_id, set())
                    full_history_for_prompt = user_history + list(items_to_forget_for_user)
                    history_str = " ".join(f"item_{item}" for item in full_history_for_prompt[-20:]) if full_history_for_prompt else "<empty>"
                    prompt = f"User {mapped_user_id} recent history: {history_str}. Recommend next item."

                    is_forget_user = mapped_user_id in unlearning_requests
                    use_side_memory = is_forget_user and (dual_runtime is not None)

                    if use_side_memory:
                        dual_runtime.adapter.set_mode("side")
                        routed_to_side_count += 1
                    elif dual_runtime:
                        dual_runtime.adapter.set_mode("main")

                    # tokenize and move tensors to model device
                    inputs = self.model_wrapper.tokenizer(prompt, return_tensors="pt")
                    torch_device = getattr(self.model_wrapper, 'device', ('cuda' if torch.cuda.is_available() else 'cpu'))
                    inputs = {k: v.to(torch_device) for k, v in inputs.items()}

                    num_return_sequences = getattr(self.args, 'num_return_sequences', 10)
                    num_beams = getattr(self.args, 'num_beams', 10)
                    max_gen_len = getattr(self.args, 'max_gen_len', 150)

                    # Ensure we request at least max_k sequences so de-duplication can still
                    # produce up to K unique recommendations without relying on fallback.
                    max_k = max(self.k_values) if hasattr(self, 'k_values') else 20
                    if num_return_sequences < max_k:
                        if self.verbose:
                            logger.debug("调整 generate.num_return_sequences: %d -> %d (以满足 max_k=%d)", num_return_sequences, max_k, max_k)
                        num_return_sequences = max_k

                    # transformers requires num_return_sequences <= num_beams when using beam search.
                    # If user set num_beams smaller, bump it up to avoid ValueError.
                    if num_beams < num_return_sequences:
                        if self.verbose:
                            logger.debug("调整 generate.num_beams: %d -> %d (确保 num_beams >= num_return_sequences)", num_beams, num_return_sequences)
                        num_beams = num_return_sequences

                    # helper: temperature-scaled generation with optional score outputs
                    try:
                        from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
                    except Exception:
                        LogitsProcessor = object  # type: ignore
                        class LogitsProcessorList(list):  # type: ignore
                            pass

                    class TemperatureLogitsProcessor(LogitsProcessor):  # type: ignore
                        def __init__(self, temperature: float) -> None:
                            self.t = max(1e-6, float(temperature))
                        def __call__(self, input_ids, scores):
                            return scores / self.t

                    class TokenPenaltyProcessor(LogitsProcessor):  # type: ignore
                        def __init__(self, banned_token_ids: set, penalty: float = 20.0) -> None:
                            self.banned = set(int(x) for x in banned_token_ids)
                            self.penalty = float(max(0.0, penalty))
                        def __call__(self, input_ids, scores):
                            if not self.banned or self.penalty <= 0:
                                return scores
                            try:
                                vocab = scores.size(-1)
                                import torch as _torch
                                idx = _torch.tensor(list(self.banned), device=scores.device, dtype=_torch.long)
                                valid = (idx >= 0) & (idx < vocab)
                                idx = idx[valid]
                                if idx.numel() > 0:
                                    scores.index_fill_(dim=-1, index=idx, value=(scores.min().item() - self.penalty))
                            except Exception:
                                pass
                            return scores

                    def _build_processors(temp: float, penalty_tokens: Optional[set] = None, penalty_val: float = 0.0):
                        procs = LogitsProcessorList()
                        if temp and abs(float(temp) - 1.0) > 1e-6:
                            procs.append(TemperatureLogitsProcessor(float(temp)))
                        if penalty_tokens and penalty_val > 0.0:
                            procs.append(TokenPenaltyProcessor(set(penalty_tokens), float(penalty_val)))
                        return procs

                    def _encode_bad_words(_tokenizer, forbidden_items):
                        seqs = []
                        first_tokens = set()
                        for it in forbidden_items:
                            text = f"item_{it}"
                            try:
                                ids = _tokenizer.encode(text, add_special_tokens=False)
                            except Exception:
                                ids = []
                            if ids:
                                seqs.append(ids)
                                first_tokens.add(ids[0])
                        return seqs, first_tokens

                    def _generate_with_temp(_model, _tokenizer, _inputs: dict, temp: float,
                                             bad_words_ids: Optional[list] = None,
                                             penalty_tokens: Optional[set] = None,
                                             penalty_val: float = 0.0):
                        processors = _build_processors(temp, penalty_tokens=penalty_tokens, penalty_val=penalty_val)
                        with torch.no_grad():
                            out = _model.generate(
                                **_inputs,
                                max_length=max_gen_len,
                                num_return_sequences=num_return_sequences,
                                num_beams=num_beams,
                                do_sample=False,
                                early_stopping=True,
                                logits_processor=processors,
                                bad_words_ids=bad_words_ids,
                                return_dict_in_generate=True,
                                output_scores=True,
                            )
                        decoded_local = self.model_wrapper.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
                        # compute normalized entropy from first step
                        norm_ent = None
                        try:
                            if out.scores and len(out.scores) > 0:
                                first_scores = out.scores[0]
                                p = torch.softmax(first_scores, dim=-1)
                                H = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)
                                V = p.size(-1)
                                norm_ent = float((H / math.log(V)).mean().item())
                        except Exception:
                            norm_ent = None
                        return decoded_local, norm_ent

                    # choose temperature by route
                    temp_to_use = self.side_temperature if use_side_memory else self.base_temperature
                    # prepare inputs once
                    gen_inputs = {k: v.to(torch_device) for k, v in inputs.items()}
                    # certified-forgetful decoding: build bad_words_ids & penalties only for forget users on side
                    bad_words_ids = None
                    penalty_tokens = None
                    penalty_val = 0.0
                    if use_side_memory and bool(getattr(self.args, 'certified_forgetful_decoding', False)):
                        if items_to_forget_for_user:
                            bad_words_ids, penalty_tokens = _encode_bad_words(self.model_wrapper.tokenizer, list(items_to_forget_for_user))
                            penalty_val = float(getattr(self.args, 'forbidden_penalty', 20.0))

                    generated_texts, norm_entropy = _generate_with_temp(
                        self.model_wrapper.model,
                        self.model_wrapper.tokenizer,
                        gen_inputs,
                        temp_to_use,
                        bad_words_ids=bad_words_ids,
                        penalty_tokens=penalty_tokens,
                        penalty_val=penalty_val,
                    )

                    recommended_items = []
                    seen_items = set()
                    item_pattern = re.compile(r'item_(\d+)', re.IGNORECASE)
                    for text in generated_texts:
                        items = item_pattern.findall(text)
                        for item in items:
                            if item not in seen_items:
                                recommended_items.append(item)
                                seen_items.add(item)

                    # ========== 详细诊断（可帮助定位 tokenization / vocab mismatch / 越界问题） ==========
                    try:
                        # tokenization of the input prompt
                        tokenized_prompt = self.model_wrapper.tokenizer.tokenize(prompt)
                        tokenized_ids = self.model_wrapper.tokenizer(prompt, return_tensors='pt')['input_ids'][0].tolist()

                        # model embedding table size (rows)
                        emb_rows = None
                        try:
                            emb = self.model_wrapper.model.get_input_embeddings()
                            emb_rows = emb.weight.size(0)
                        except Exception:
                            emb_rows = None

                        # count how many generated sequences contained at least one item_{id}
                        total_generated = len(generated_texts)
                        generated_with_match = sum(1 for t in generated_texts if item_pattern.search(t))

                        diag = {
                            'user_id': mapped_user_id,
                            'prompt_tokens': tokenized_prompt,
                            'prompt_token_ids': tokenized_ids,
                            'tokenizer_vocab_size': len(self.model_wrapper.tokenizer) if hasattr(self.model_wrapper, 'tokenizer') else None,
                            'model_embedding_rows': emb_rows,
                            'generated_sequences': total_generated,
                            'generated_with_item_match': generated_with_match,
                            'unique_recommended_items': len(recommended_items),
                            'sample_recommended_items': recommended_items[:10]
                        }

                        # detect any out-of-range ids in prompt tokens
                        out_of_range_ids = []
                        if emb_rows is not None:
                            for tid in tokenized_ids:
                                if isinstance(tid, int) and tid >= emb_rows:
                                    out_of_range_ids.append(int(tid))

                        # also check generated token ids (best-effort): decode back to ids
                        try:
                            # we can attempt to decode generated_texts into ids via tokenizer
                            gen_ids = [self.model_wrapper.tokenizer(text, return_tensors='pt')['input_ids'][0].tolist() for text in generated_texts]
                            gen_out_of_range = []
                            if emb_rows is not None:
                                for gid_list in gen_ids:
                                    for gid in gid_list:
                                        if gid >= emb_rows:
                                            gen_out_of_range.append(int(gid))
                            if gen_out_of_range:
                                out_of_range_ids.extend(gen_out_of_range)
                        except Exception:
                            # best-effort only; ignore if tokenizer can't re-tokenize snippets
                            pass

                        if out_of_range_ids:
                            diag['out_of_range_ids'] = sorted(set(out_of_range_ids))

                        # attach diag to results for later inspection and log at debug level
                        self.results.setdefault('per_user_diagnostics', []).append(diag)
                        if self.show_user_diagnostics:
                            logger.debug("[EVAL DIAG FULL] user=%s diag=%s", mapped_user_id, {k: diag.get(k) for k in ('tokenizer_vocab_size','model_embedding_rows','generated_sequences','generated_with_item_match','unique_recommended_items','out_of_range_ids')})
                    except Exception as _diag_exc:
                        # Do not fail evaluation because diagnostics failed
                        if self.show_user_diagnostics:
                            logger.debug("[EVAL DIAG ERROR] 无法生成诊断信息 user=%s error=%s", mapped_user_id, _diag_exc)

                    # ========== 诊断输出：记录每个用户的推荐数量与样例 ==========
                    max_k = max(self.k_values) if hasattr(self, 'k_values') else 20
                    if len(recommended_items) < max_k:
                        # minimal debug output: only log counts when user diagnostics enabled
                        if self.show_user_diagnostics:
                            logger.debug("[EVAL DIAG] user=%s num_unique_recs=%d (less than max_k=%d). Generated_texts_count=%d.", mapped_user_id, len(recommended_items), max_k, len(generated_texts))

                    # entropy/uniqueness-based fallback: for forget users routed to side
                    trigger_fallback = False
                    if use_side_memory and self.use_entropy_fallback:
                        if norm_entropy is not None and self.conf_fallback_threshold is not None:
                            if float(norm_entropy) > float(self.conf_fallback_threshold):
                                trigger_fallback = True
                        unique_ratio = (len(recommended_items) / float(max_k)) if max_k > 0 else 0.0
                        if unique_ratio < float(self.min_unique_ratio):
                            trigger_fallback = True

                    if trigger_fallback and dual_runtime:
                        # regenerate with main (base) at base temperature
                        dual_runtime.adapter.set_mode("main")
                        gen_inputs_fb = {k: v.to(torch_device) for k, v in inputs.items()}
                        generated_texts, _ = _generate_with_temp(self.model_wrapper.model, self.model_wrapper.tokenizer, gen_inputs_fb, self.base_temperature)
                        # rebuild items
                        recommended_items = []
                        seen_items = set()
                        for text in generated_texts:
                            items = item_pattern.findall(text)
                            for item in items:
                                if item not in seen_items:
                                    recommended_items.append(item)
                                    seen_items.add(item)
                                if len(recommended_items) >= max_k:
                                    break
                            if len(recommended_items) >= max_k:
                                break

                    # ========== 回退策略：若去重后候选过少，则用训练集流行物品补齐，避免 top-K 全为 0 的误导性结果 ==========
                    # Fallback strategy: optionally pad with popular items from training set
                    used_fallback = False
                    if len(recommended_items) < max_k and not getattr(self.args, 'disable_fallback', False):
                        try:
                            popular = list(train_df['mapped_item_id'].value_counts().index.astype(str))
                        except Exception:
                            popular = []
                        for p in popular:
                            if p not in seen_items:
                                recommended_items.append(p)
                                seen_items.add(p)
                                used_fallback = True
                            if len(recommended_items) >= max_k:
                                break
                    if used_fallback:
                        # track fallback usage in results dict
                        self.results.setdefault('fallback_count', 0)
                        self.results['fallback_count'] += 1

                    if self.show_user_diagnostics:
                        logger.debug("[EVAL DIAG] user=%s final_rec_count=%d sample_recs=%s", mapped_user_id, len(recommended_items), recommended_items[:10])

                    # Certified check & certificate emission
                    if use_side_memory and bool(getattr(self.args, 'emit_certificates', False)):
                        try:
                            forb_set = set(map(str, items_to_forget_for_user)) if items_to_forget_for_user else set()
                            inter = [x for x in recommended_items if x in forb_set]
                            import time as _time, os as _os, json as _json
                            _os.makedirs('results/certificates', exist_ok=True)
                            cert = {
                                'time': int(_time.time()),
                                'user_id': mapped_user_id,
                                'route': 'side' if use_side_memory else 'main',
                                'forbidden_items': list(forb_set),
                                'topk_items': recommended_items[:max_k],
                                'violation_count': len(inter),
                                'violations': inter,
                            }
                            with open(_os.path.join('results/certificates', f'cert_{mapped_user_id}_{int(_time.time())}.json'), 'w', encoding='utf-8') as f:
                                _json.dump(cert, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

                    predictions.append({'user_id': mapped_user_id, 'recommended_items': recommended_items})

                    # For each user, populate the ground truth lists for the two separate evaluations
                    for item_id in gt_retained.get(mapped_user_id, set()):
                        ground_truth_retain_list.append({'user_id': mapped_user_id, 'item_id': item_id})
                    for item_id in gt_forgotten.get(mapped_user_id, set()):
                        ground_truth_forget_list.append({'user_id': mapped_user_id, 'item_id': item_id})

                    processed_users_count += 1
                except Exception as e:
                    # Log detailed debug info for the failing user/prompt
                    import traceback
                    tb = traceback.format_exc()
                    logger.error("❌ 用户评估时发生异常 user=%s prompt=%s error=%s", mapped_user_id, locals().get('prompt', '<no-prompt>'), e)
                    logger.debug("Traceback:\n%s", tb)
                    # add minimal placeholder to keep evaluations moving
                    predictions.append({'user_id': mapped_user_id, 'recommended_items': []})
                    # continue with next user
                    continue

            if dual_runtime:
                dual_runtime.adapter.set_mode("main")

            if processed_users_count == 0:
                logger.warning(f"{set_name}集没有可评估的用户。")
                return {}
            
            if dual_runtime:
                logger.info("=" * 60)
                logger.info(f"确定性路由决策分析 ({set_name}):")
                logger.info(f"  总计 {routed_to_side_count} / {processed_users_count} ({routed_to_side_count/processed_users_count:.2%}) 个用户被路由到侧记忆。")
                logger.info("=" * 60)

            final_metrics = {}
            if ground_truth_retain_list:
                retain_metrics = self.evaluator.evaluate_recommendations(predictions, ground_truth_retain_list, self.target_metrics)
                # compute precision@k and user-averaged recall@k
                precision_metrics = {}
                user_avg_recall_metrics = {}
                for k in self.k_values:
                    # precision@k = total_hits_on_topk / (k * num_users_with_preds)
                    # We'll compute precision as hits / (k * N_users) where N_users is users with gt
                    prec = self._precision_at_k(predictions, ground_truth_retain_list, k)
                    user_avg_rec = self._user_averaged_recall(predictions, ground_truth_retain_list, k)
                    precision_metrics[f'precision@{k}'] = prec
                    user_avg_recall_metrics[f'user_avg_recall@{k}'] = user_avg_rec

                logger.info(f"📈 {set_name} - 保留效用 (Retain Utility):")
                for metric, value in {**retain_metrics, **precision_metrics, **user_avg_recall_metrics}.items():
                    logger.info(f"  {metric}: {value:.4f}")
                    final_metrics[f"{metric}_retain"] = value
            else:
                logger.info(f"ℹ️ {set_name} - 无需评估的保留项。")
            
            if ground_truth_forget_list:
                forget_target_metrics = []
                for k in self.k_values:
                    forget_target_metrics.extend([f'hit@{k}', f'ndcg@{k}', f'recall@{k}'])
                
                forget_metrics = self.evaluator.evaluate_recommendations(predictions, ground_truth_forget_list, forget_target_metrics)
                # also compute precision & user-avg recall for forget set
                precision_metrics_f = {}
                user_avg_recall_metrics_f = {}
                for k in self.k_values:
                    precision_metrics_f[f'precision@{k}'] = self._precision_at_k(predictions, ground_truth_forget_list, k)
                    user_avg_recall_metrics_f[f'user_avg_recall@{k}'] = self._user_averaged_recall(predictions, ground_truth_forget_list, k)

                logger.info(f"📉 {set_name} - 遗忘效能 (Forget Efficacy):")
                for metric, value in {**forget_metrics, **precision_metrics_f, **user_avg_recall_metrics_f}.items(): 
                    logger.info(f"  {metric}_forgotten: {value:.4f} (此值越低，遗忘效果越好)")
                    final_metrics[f"{metric}_forgotten"] = value
            else:
                logger.info(f"ℹ️ {set_name} - 无需评估的遗忘项。")
            
            return final_metrics

    def _precision_at_k(self, predictions: list, ground_truth: list, k: int) -> float:
        """Compute global precision@k = total_hits_on_topk / (k * num_users_with_gt)"""
        # build maps
        user_preds = self.evaluator._build_user_predictions(predictions)
        user_gt = self.evaluator._build_user_ground_truth(ground_truth)
        total_hits = 0
        total_possible = 0
        for uid, gt_items in user_gt.items():
            if not gt_items:
                continue
            preds = user_preds.get(uid, [])[:k]
            hits = sum(1 for it in preds if it in set(gt_items))
            total_hits += hits
            total_possible += k
        if total_possible == 0:
            return 0.0
        return total_hits / total_possible

    def _user_averaged_recall(self, predictions: list, ground_truth: list, k: int) -> float:
        """Compute user-averaged recall: mean_u |Pred_u@k ∩ GT_u| / |GT_u|"""
        user_preds = self.evaluator._build_user_predictions(predictions)
        user_gt = self.evaluator._build_user_ground_truth(ground_truth)
        recalls = []
        for uid, gt_items in user_gt.items():
            gt_set = set(gt_items)
            if not gt_set:
                continue
            preds = user_preds.get(uid, [])[:k]
            hits = sum(1 for it in preds if it in gt_set)
            recalls.append(hits / len(gt_set))
        if not recalls:
            return 0.0
        return float(sum(recalls) / len(recalls))

    def run_evaluation(self):
        """运行完整的数据集评估"""
        if self.compare_mode:
            return self.run_comparison_evaluation()
        else:
            return self.run_single_evaluation()
    
    def run_single_evaluation(self):
        """运行单个模型评估"""
        logger.info(f"🚀 开始P5推荐模型评估: {self.model_path}")
        
        try:
            # 1. 初始化模型
            self.initialize_model()
            
            # 2. 加载和划分数据
            data_info = self.load_and_split_data()
            
            # 3. 评估模型性能
            logger.info("\n" + "=" * 80)
            logger.info("📊 模型在不同数据集上的性能评估")
            logger.info("=" * 80)
            
            # 评估全体用户性能（可跳过以加速）
            if not getattr(self.args, 'skip_all_users', False):
                logger.info("\n🔍 评估全体用户性能:")
                self.results['all_users'] = self.evaluate_model_performance(
                    data_info, data_info['all_users'], "全体用户", save_recs=self.save_predictions
                )
            else:
                logger.info("⏭️ 已跳过全体用户评估 (--skip_all_users)")
            
            # 评估保留集性能
            logger.info("\n🔍 评估保留集性能:")
            self.results['retain_set'] = self.evaluate_model_performance(
                data_info, data_info['eval_retain_users'], "保留集", save_recs=self.save_predictions
            )
            
            # 评估遗忘集性能
            logger.info("\n🔍 评估遗忘集性能:")
            self.results['forget_set'] = self.evaluate_model_performance(
                data_info, data_info['eval_forget_users'], "遗忘集", save_recs=self.save_predictions
            )
            
            # 4. 性能分析
            self.analyze_performance()
            
            # 5. 保存结果
            self.save_results(data_info)
            
            logger.info("\n🎉 模型评估完成!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 评估失败: {e}")
            return False

    def run_comparison_evaluation(self):
            """
            [最终修复版] 运行对比评估。
            直接从 forget/retain 文件中加载用户，以匹配新的、更真实的遗忘任务。
            """
            logger.info("🚀 开始P5推荐模型对比评估")
            logger.info(f"📋 原始模型: {self.original_model_path}")
            logger.info(f"📋 遗忘后模型 (工件路径): {self.unlearned_model_path}")
            
            try:
                # 1. 加载所有必要的数据
                data_info = self.load_and_split_data()
                
                forget_samples_path = os.path.join(PROJECT_ROOT, 'results', 'forget_samples_subset.json')
                if not os.path.exists(forget_samples_path):
                    raise FileNotFoundError(f"遗忘样本文件不存在: {forget_samples_path}。")
                
                with open(forget_samples_path, 'r', encoding='utf-8') as f:
                    forget_samples = json.load(f)

                # 2. [核心逻辑修复] 直接从已生成的文件中加载评估用户，不再进行无效搜索
                logger.info("🎯 直接从 forget_samples.json 加载评估用户...")
                
                user_map = data_info['mappings']['user_to_mapped']
                
                # a. 加载所有遗忘请求
                unlearning_requests = defaultdict(list)
                all_forget_user_ids = []
                for sample in forget_samples:
                    user_id = str(sample.get("user_id"))
                    mapped_user_id = user_map.get(user_id)
                    items_to_forget = sample.get("suppression_targets", [])
                    
                    if mapped_user_id and items_to_forget:
                        unlearning_requests[mapped_user_id].extend(map(str, items_to_forget))
                        if mapped_user_id not in all_forget_user_ids:
                            all_forget_user_ids.append(mapped_user_id)

                # b. 从加载的用户中抽样
                if len(all_forget_user_ids) < self.eval_sample_size:
                    logger.warning(
                        f"警告：请求评估的遗忘用户数({self.eval_sample_size}) > 文件中实际用户数({len(all_forget_user_ids)})。将使用全部用户。"
                    )
                    eval_forget_users = all_forget_user_ids
                else:
                    eval_forget_users = random.sample(all_forget_user_ids, self.eval_sample_size)

                if not eval_forget_users:
                    raise RuntimeError("致命错误：从 forget_samples.json 中未能加载到任何有效的遗忘用户。")

                # c. 从保留用户列表中抽样相同数量的用户
                eval_retain_users = random.sample(
                    [u for u in data_info['retain_users'] if u not in eval_forget_users],
                    min(len(eval_forget_users), len(data_info['retain_users']))
                )

                logger.info("=" * 60)
                logger.info("📊 公平评估集构造完成:")
                logger.info(f"  遗忘集评估用户数: {len(eval_forget_users)}")
                logger.info(f"  保留集评估用户数: {len(eval_retain_users)}")
                logger.info("=" * 60)
                
                retain_set_info = {'users': eval_retain_users, 'unlearning_requests': {}}
                forget_set_info = {'users': eval_forget_users, 'unlearning_requests': unlearning_requests}
                    
                # --- 3. 后续评估流程完全不变 ---
                logger.info("\n" + "=" * 80)
                logger.info("📊 评估原始模型性能")
                self.initialize_model(self.original_model_path)
                original_results = {}
                original_results['retain_set'] = self.evaluate_model_performance(data_info, retain_set_info, "保留集(原始)")
                original_results['forget_set'] = self.evaluate_model_performance(data_info, forget_set_info, "遗忘集(原始)")
                    
                logger.info("\n" + "=" * 80)
                logger.info("📊 评估遗忘后模型性能")
                self.dual_artifacts_path = self.unlearned_model_path
                self.initialize_dual_runtime(self.dual_artifacts_path, self.dual_threshold)
                unlearned_results = {}
                unlearned_results['retain_set'] = self.evaluate_model_performance(data_info, retain_set_info, "保留集(遗忘后)")
                unlearned_results['forget_set'] = self.evaluate_model_performance(data_info, forget_set_info, "遗忘集(遗忘后)")
                    
                self.analyze_comparison(original_results, unlearned_results)
                self.save_comparison_results(data_info, original_results, unlearned_results)
                    
                logger.info("\n🎉 对比评估完成!")
                return True
                    
            except Exception as e:
                logger.error(f"❌ 对比评估失败: {e}", exc_info=True)
                return False

    def analyze_comparison(self, original_results: Dict, unlearned_results: Dict):
            """[最终版] 分析原始模型与遗忘后模型的性能对比，并提供结构化的遗忘效果解读。"""
            logger.info("\n" + "=" * 80)
            logger.info("📊 原始模型 vs 遗忘后模型性能对比分析")
            logger.info("=" * 80)
            
            # --- 1. 保留集分析 (完全不受影响的用户) ---
            set_key, set_name = 'retain_set', '保留集 (评估模型通用性)'
            if set_key in original_results and set_key in unlearned_results and original_results[set_key]:
                logger.info(f"\n📋 {set_name}性能对比:")
                logger.info("-" * 60)
                orig = original_results[set_key]
                unlearned = unlearned_results[set_key]
                all_metrics_keys = sorted(list(set(orig.keys()) | set(unlearned.keys())))
                for metric_key in all_metrics_keys:
                    orig_val = orig.get(metric_key, 0.0)
                    unlearned_val = unlearned.get(metric_key, 0.0)
                    change = unlearned_val - orig_val
                    change_pct = (change / orig_val * 100) if orig_val != 0 else float('inf') if change > 0 else 0.0
                    indicator = "✅ (性能保持良好)" if abs(change_pct) < 10 else "⚠️ (性能波动较大)"
                    logger.info(f"  {metric_key:25s}: {orig_val:.4f} → {unlearned_val:.4f} "
                                f"(Δ={change:+.4f}, {change_pct:+.1f}%) {indicator}")
            
            # --- 2. 遗忘集分析 (受影响的用户) ---
            set_key, set_name = 'forget_set', '遗忘集 (评估遗忘效果和连带损伤)'
            if set_key in original_results and set_key in unlearned_results and original_results[set_key]:
                logger.info(f"\n📋 {set_name}性能对比:")
                
                orig = original_results[set_key]
                unlearned = unlearned_results[set_key]
                all_metrics_keys = sorted(list(set(orig.keys()) | set(unlearned.keys())))
                
                # [核心修改] 将指标分为两组打印
                forgotten_metrics = [k for k in all_metrics_keys if "_forgotten" in k]
                retain_metrics = [k for k in all_metrics_keys if "_retain" in k]

                logger.info("\n  --- 1. 遗忘效能 (Forget Efficacy) ---")
                logger.info("  (此部分所有指标越接近0，下降幅度越大越好)")
                logger.info("  " + "-" * 50)
                if forgotten_metrics:
                    for metric_key in forgotten_metrics:
                        orig_val = orig.get(metric_key, 0.0)
                        unlearned_val = unlearned.get(metric_key, 0.0)
                        change = unlearned_val - orig_val
                        change_pct = (change / orig_val * 100) if orig_val != 0 else -100.0 if unlearned_val == 0 else 0.0
                        indicator = "✅ (遗忘成功)" if change < -0.05 or unlearned_val < 0.01 else "❌ (遗忘不彻底)"
                        logger.info(f"    {metric_key:23s}: {orig_val:.4f} → {unlearned_val:.4f} "
                                    f"(Δ={change:+.4f}, {change_pct:+.1f}%) {indicator}")
                else:
                    logger.info("    未能计算遗忘效能指标(数据采样问题)。")

                logger.info("\n  --- 2. 连带损伤 (Collateral Damage) ---")
                logger.info("  (此部分指标越稳定，波动越小越好)")
                logger.info("  " + "-" * 50)
                if retain_metrics:
                    for metric_key in retain_metrics:
                        orig_val = orig.get(metric_key, 0.0)
                        unlearned_val = unlearned.get(metric_key, 0.0)
                        change = unlearned_val - orig_val
                        change_pct = (change / orig_val * 100) if orig_val != 0 else float('inf') if change > 0 else 0.0
                        indicator = "👍 (可接受)" if abs(change_pct) < 40 else "👎 (损伤较大)"
                        logger.info(f"    {metric_key:23s}: {orig_val:.4f} → {unlearned_val:.4f} "
                                    f"(Δ={change:+.4f}, {change_pct:+.1f}%) {indicator}")

    def evaluate_unlearning_effectiveness(self, original_results: Dict, unlearned_results: Dict):
        """评估整体遗忘效果"""
        logger.info("\n🎯 整体遗忘效果评估:")
        logger.info("-" * 60)
        
        forget_drops = []
        retain_drops = []
        
        # --- 核心修复：使用正确的带后缀的 metric key ---
        # 使用配置中的 k_values 来计算保留/遗忘指标的变化，避免硬编码并保持一致性
        for k in getattr(self, 'k_values', [10, 20]):
            retain_metric_key = f"hit@{k}_retain"
            # 遗忘效能通常关注 Recall@K（带 _forgotten 后缀）
            forget_metric_key = f"recall@{k}_forgotten"

            if (retain_metric_key in original_results.get('retain_set', {}) and
                retain_metric_key in unlearned_results.get('retain_set', {})):
                orig_retain_val = original_results['retain_set'][retain_metric_key]
                unlearned_retain_val = unlearned_results['retain_set'][retain_metric_key]
                retain_drop = orig_retain_val - unlearned_retain_val
                retain_drops.append(retain_drop)

            if (forget_metric_key in original_results.get('forget_set', {}) and
                forget_metric_key in unlearned_results.get('forget_set', {})):
                orig_forget_val = original_results['forget_set'][forget_metric_key]
                unlearned_forget_val = unlearned_results['forget_set'][forget_metric_key]
                forget_drop = orig_forget_val - unlearned_forget_val
                forget_drops.append(forget_drop)
        
        if forget_drops and retain_drops:
            avg_forget_drop = sum(forget_drops) / len(forget_drops)
            avg_retain_drop = sum(retain_drops) / len(retain_drops)
            selectivity = avg_forget_drop - max(0, avg_retain_drop)
            
            logger.info(f"平均遗忘集性能下降(越大越好): {avg_forget_drop:.4f}")
            logger.info(f"平均保留集性能下降(越小越好): {avg_retain_drop:.4f}")
            logger.info(f"选择性指标 (Selectivity): {selectivity:.4f}")
            
            if selectivity > 0.1 and avg_retain_drop < 0.05:
                overall = "✅ 遗忘效果优秀"
            elif selectivity > 0.05:
                overall = "⚠️ 遗忘效果良好"
            else:
                overall = "❌ 遗忘效果一般或较差"
                
            logger.info(f"整体评估: {overall}")
        
    def save_comparison_results(self, data_info: Dict, original_results: Dict, unlearned_results: Dict):
        """保存对比评估结果"""
        timestamp = int(time.time())
        results_file = f"results/comparison_evaluation_{timestamp}{self.output_suffix}.json"
        
        os.makedirs("results", exist_ok=True)
        
        # 计算变化量
        changes = {}
        for set_key in ['all_users', 'retain_set', 'forget_set']:
            changes[set_key] = {}
            # 在 --skip_all_users 情况下，部分集合可能未评估；此处需判空
            if set_key not in original_results or set_key not in unlearned_results:
                continue
            for metric in self.target_metrics:
                if metric in original_results[set_key] and metric in unlearned_results[set_key]:
                    orig_val = original_results[set_key][metric]
                    new_val = unlearned_results[set_key][metric]
                    changes[set_key][metric] = {
                        'absolute_change': new_val - orig_val,
                        'relative_change': ((new_val - orig_val) / orig_val * 100) if orig_val > 0 else 0
                    }
        
        # 保存完整对比结果
        comparison_results = {
            'evaluation_time': timestamp,
            'models': {
                'original_model': self.original_model_path,
                'unlearned_model': self.unlearned_model_path
            },
            'config': {
                'forget_ratio': self.forget_ratio,
                'eval_sample_size': self.eval_sample_size,
                'k_values': self.k_values,
                'target_metrics': self.target_metrics
            },
            'dataset_info': {
                'total_users': len(data_info['all_users']),
                'retain_users': len(data_info['retain_users']),
                'forget_users': len(data_info['forget_users']),
                'eval_retain_users': len(data_info['eval_retain_users']),
                'eval_forget_users': len(data_info['eval_forget_users'])
            },
            'performance_results': {
                'original': original_results,
                'unlearned': unlearned_results,
                'changes': changes
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 对比评估结果已保存至: {results_file}")

    def analyze_performance(self):
        """分析不同数据集的性能差异"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 不同数据集性能对比分析")
        logger.info("=" * 80)
        all_users = self.results.get('all_users', {}) or {}
        retain_set = self.results.get('retain_set', {}) or {}
        forget_set = self.results.get('forget_set', {}) or {}

        logger.info("\n📋 各数据集性能总结:")
        logger.info("-" * 60)

        # Prepare table header
        ks = sorted(self.k_values)
        header_cols = ['metric'] + [f'all@{k}' for k in ks] + [f'retain@{k}' for k in ks] + [f'forget@{k}' for k in ks]
        logger.info("| %s |", " | ".join(header_cols))
        logger.info("|%s|", "|".join(['-' * (len(c)+2) for c in header_cols]))

        # For each metric type, print a row per metric (hit, ndcg, recall)
        metric_types = ['hit', 'ndcg', 'recall']
        for mtype in metric_types:
            for k in ks:
                key = f"{mtype}@{k}"
                a = all_users.get(key, None)
                r = retain_set.get(key, None)
                f = forget_set.get(key, None)
                a_s = f"{a:.4f}" if a is not None else "-"
                r_s = f"{r:.4f}" if r is not None else "-"
                f_s = f"{f:.4f}" if f is not None else "-"
                logger.info("| %s@%d | %s | %s | %s |", mtype, k, a_s, r_s, f_s)

    def save_results(self, data_info):
        """保存评估结果"""
        timestamp = int(time.time())
        results_file = f"results/dataset_evaluation_{timestamp}{self.output_suffix}.json"
        
        os.makedirs("results", exist_ok=True)
        
        # 保存完整结果
        full_results = {
            'evaluation_time': timestamp,
            'model_path': self.model_path,
            'config': {
                'forget_ratio': self.forget_ratio,
                'eval_sample_size': self.eval_sample_size,
                'k_values': self.k_values,
                'target_metrics': self.target_metrics
            },
            'dataset_info': {
                'total_users': len(data_info['all_users']),
                'retain_users': len(data_info['retain_users']),
                'forget_users': len(data_info['forget_users']),
                'eval_retain_users': len(data_info['eval_retain_users']),
                'eval_forget_users': len(data_info['eval_forget_users'])
            },
            'performance_results': self.results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 评估结果已保存至: {results_file}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="P5推荐模型数据集划分评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    使用示例:
    # 评估单个模型
    python evaluate_datasets.py --model_path models/ML1M_sequential.pt
    
    # 评估遗忘后模型并添加后缀
    python evaluate_datasets.py --model_path models/ML1M_sequential_unlearned.pt --output_suffix _after_unlearning
    
    # 对比原始模型和遗忘后模型
    python evaluate_datasets.py --original_model models/original.pt --unlearned_model models/unlearned.pt --compare
    
    # 自定义评估参数
    python evaluate_datasets.py --model_path models/model.pt --k_values 5,10,20,50 --eval_sample_size 100 --save_predictions
            """
    )
    
    # 模型选择参数（互斥组）
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model_path', 
        type=str,
        help='单个模型文件路径'
    )
    model_group.add_argument(
        '--compare',
        action='store_true',
        help='启用对比模式（需要同时指定 --original_model 和 --unlearned_model）'
    )
    
    # 对比模式参数
    parser.add_argument(
        '--original_model',
        type=str,
        help='原始模型路径（对比模式必需）'
    )
    parser.add_argument(
        '--unlearned_model', 
        type=str,
        help='遗忘后模型路径（对比模式必需）'
    )
    
    # 输出配置
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='',
        help='输出文件名后缀'
    )
    
    # 评估参数
    parser.add_argument(
        '--forget_ratio',
        type=float,
        default=0.01,
        help='遗忘集比例 (默认: 0.01)'
    )
    parser.add_argument(
        '--eval_sample_size',
        type=int,
        default=50,
        help='每个集合的评估样本大小 (默认: 50)'
    )
    parser.add_argument(
        '--k_values',
        type=str,
        default='10,20',
        help='评估的K值列表，逗号分隔 (默认: 10,20)'
    )
    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=10,
        help='generate 时每个输入返回的序列数（默认: 10）'
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=10,
        help='generate 时使用的 beam 大小（默认: 10）'
    )
    parser.add_argument(
        '--max_gen_len',
        type=int,
        default=150,
        help='generate 时的 max_length（默认: 150）'
    )
    parser.add_argument(
        '--eval_rescan_mode',
        type=str,
        choices=['auto', 'skip', 'force'],
        default='auto',
        help='评估前是否重新扫描 Δact 阈值: auto=遵循模型配置, skip=禁用, force=强制启用'
    )
    # 推理阶段温度标定与置信回退
    parser.add_argument(
        '--base_temperature',
        type=float,
        default=1.0,
        help='主记忆/基础模型生成温度 (对beam search的logits缩放)'
    )
    parser.add_argument(
        '--side_temperature',
        type=float,
        default=1.2,
        help='侧记忆生成温度 (>1 以软化过度自信)'
    )
    parser.add_argument(
        '--use_entropy_fallback',
        action='store_true',
        help='启用基于归一化熵的置信回退 (当不确定性高时回退主记忆)'
    )
    parser.add_argument(
        '--conf_fallback_threshold',
        type=float,
        default=0.85,
        help='触发回退的归一化熵阈值 (0-1, 越大越不确定)'
    )
    parser.add_argument(
        '--min_unique_ratio',
        type=float,
        default=0.3,
        help='若去重后的推荐数/TopK 小于此比率，则触发回退到主记忆'
    )
    # Certified Forgetful Decoding (CFD)
    parser.add_argument(
        '--certified_forgetful_decoding',
        action='store_true',
        help='启用解码护栏：在侧记忆对遗忘用户生成时，禁止输出被遗忘条目（bad_words_ids + penalty）'
    )
    parser.add_argument(
        '--forbidden_penalty',
        type=float,
        default=20.0,
        help='对被禁止token施加的对数几率惩罚，数值越大越难被选中（默认: 20.0）'
    )
    parser.add_argument(
        '--emit_certificates',
        action='store_true',
        help='启用后会在 results/certificates/ 目录下为每个遗忘用户生成一次性证书，记录是否出现违规推荐'
    )
    
    # 其他选项
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='保存详细的推荐结果'
    )
    parser.add_argument(
        '--save_examples_num',
        type=int,
        default=10,
        help='保存用于人工查看的示例用户数量（默认: 10）'
    )
    parser.add_argument(
        '--skip_all_users',
        action='store_true',
        help='跳过全体用户评估，加速运行（仅评估保留集与遗忘集）'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细日志'
    )

    parser.add_argument(
        '--disable_fallback',
        action='store_true',
        help='禁用在生成结果不足时用流行物品补齐的回退策略'
    )

    parser.add_argument(
        '--dual_memory_artifacts',
        type=str,
        default=None,
        help='Dual-memory 组合模型的artifact路径 (包含侧记忆与路由器)'
    )
    parser.add_argument(
        '--dual_memory_threshold',
        type=float,
        default=None,
        help='在加载dual-memory artifact时覆盖默认路由阈值'
    )
    
    args = parser.parse_args()
    
    # 验证对比模式参数
    if args.compare:
        if not args.original_model or not args.unlearned_model:
            parser.error("对比模式需要同时指定 --original_model 和 --unlearned_model")
        if not os.path.exists(args.original_model):
            parser.error(f"原始模型文件不存在: {args.original_model}")
        if not os.path.exists(args.unlearned_model):
            parser.error(f"遗忘后模型文件不存在: {args.unlearned_model}")
    elif args.model_path:
        if not os.path.exists(args.model_path):
            parser.error(f"模型文件不存在: {args.model_path}")
    
    return args

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 打印配置信息
    logger.info("P5 推荐模型评估工具启动")
    if args.compare:
        logger.info("评估模式：对比评估")
        logger.info(f"原始模型: {args.original_model}")
        logger.info(f"遗忘后模型: {args.unlearned_model}")
    else:
        logger.info("评估模式：单模型评估")
        logger.info(f"模型路径: {args.model_path}")

    logger.info(f"评估参数: forget_ratio={args.forget_ratio}, eval_sample_size={args.eval_sample_size}, k_values={args.k_values}")
    logger.info(f"eval_rescan_mode: {args.eval_rescan_mode}")
    
    # 创建评估器并运行
    evaluator = DatasetEvaluator(args)
    success = evaluator.run_evaluation()
    
    if success:
        logger.info("✅ 评估成功完成")
    else:
        logger.error("❌ 评估失败")
        sys.exit(1)

if __name__ == "__main__":
    main()