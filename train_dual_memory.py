#!/usr/bin/env python3
"""
[最终一体化版]
Entry point for dual-memory fine-tuning.
集成了可评估数据集的自动生成功能。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import argparse
import logging
from pathlib import Path
import json
from collections import defaultdict
import random
import pandas as pd
import torch
from typing import Optional, Dict, Any
from src.dual_memory import DualMemoryConfig, train_dual_memory

# --- 数据文件路径配置 ---
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ML1M"
INTER_FILE = DEFAULT_DATA_DIR / "ml-1m.inter"
USER_INDEX_FILE = DEFAULT_DATA_DIR / "user_indexing.txt"
OUTPUT_FORGET_FILE = PROJECT_ROOT / 'results' / 'forget_samples_subset.json'
OUTPUT_RETAIN_FILE = PROJECT_ROOT / 'results' / 'retain_samples_subset.json'
# --- 配置结束 ---

def prepare_guaranteed_datasets(
    num_forget_samples: int = 500,
    num_retain_samples: int = 565, # [核心修改1] 为保留集也设置数量
    forget_percentage: float = 0.05, # [核心修改1] 遗忘交互的百分比
    force_regenerate: bool = False
):
    """
    [最终公平评估版]
    自动生成或验证遗忘/保留数据集。
    1. 筛选出一个"黄金候选池"，确保这些用户的行为在测试集中是可见的。
    2. 从同一个池子中为遗忘集和保留集抽样，保证评估基线的公平性。
    3. 为遗忘用户选取其最近5%的交互作为遗忘目标。
    """
    logging.info("--- [步骤 1/3] 准备可评估的数据集 (公平评估版) ---")
    
    # 提前加载并分割数据，因为无论如何都需要 test_df
    logging.info("⏳ 正在加载和分割交互数据...")
    df = pd.read_csv(INTER_FILE, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], dtype={'user_id': str, 'item_id': str})
    user_map = {line.strip().split()[0]: line.strip().split()[1] for line in open(USER_INDEX_FILE)}
    df['mapped_user_id'] = df['user_id'].map(user_map)
    df.dropna(subset=['mapped_user_id'], inplace=True)
    df = df.sort_values('timestamp')
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    logging.info("✅ 交互数据加载分割完成。")

    if OUTPUT_FORGET_FILE.exists() and OUTPUT_RETAIN_FILE.exists() and not force_regenerate:
        logging.info(f"✅ 数据集 '{OUTPUT_FORGET_FILE.name}' 和 '{OUTPUT_RETAIN_FILE.name}' 已存在，跳过生成。")
        return test_df # 直接返回加载好的 test_df

    logging.info("⏳ 开始构造保证评估公平的遗忘/保留数据集...")

    # 1. 加载ID映射
    rev_user_map = {v: k for k, v in user_map.items()}
    
    # 2. [核心修改2] 构造“黄金候选池”
    logging.info("🔍 正在筛选行为可预测的 '黄金候选池' 用户...")
    train_user_history_counts = train_df['mapped_user_id'].value_counts()
    # 筛选出在训练集中有足够历史（例如超过10次）的用户
    candidate_users_from_train = set(train_user_history_counts[train_user_history_counts >= 10].index)
    # 筛选出在测试集中有至少一次高分交互的用户
    test_set_active_users = set(test_df[test_df['rating'] >= 4]['mapped_user_id'].unique())
    
    golden_pool = sorted(list(candidate_users_from_train.intersection(test_set_active_users)))
    random.shuffle(golden_pool)
    logging.info(f"✅ '黄金候选池' 构造完成，共 {len(golden_pool)} 名候选用户。")

    if len(golden_pool) < num_forget_samples + num_retain_samples:
        raise RuntimeError(
            f"黄金候选池用户不足 ({len(golden_pool)})，无法满足遗忘集({num_forget_samples})和保留集({num_retain_samples})的数量需求。"
        )

    # 3. 从同一个池子中抽样，保证公平性
    forget_user_ids = set(golden_pool[:num_forget_samples])
    retain_user_ids = set(golden_pool[num_forget_samples : num_forget_samples + num_retain_samples])
    
    logging.info(f"👥 已从池中抽样: {len(forget_user_ids)} 名遗忘用户, {len(retain_user_ids)} 名保留用户。")

    # 4. [核心修改3] 生成遗忘样本，遗忘最近5%的交互
    forget_samples = []
    for user_id in forget_user_ids:
        user_history_df = train_df[train_df['mapped_user_id'] == user_id].sort_values('timestamp')
        history_items = user_history_df['item_id'].tolist()
        
        if len(history_items) < 5: continue # 历史太短的用户跳过

        num_to_forget = max(1, int(len(history_items) * forget_percentage))
        
        items_to_forget = history_items[-num_to_forget:]
        history_for_prompt = history_items[:-num_to_forget]

        if not history_for_prompt: continue

        forget_sample = {
            "user_id": rev_user_map.get(user_id, ""),
            "seq_items": history_for_prompt,
            "suppression_targets": items_to_forget
        }
        forget_samples.append(forget_sample)

    if not forget_samples:
        raise RuntimeError("未能生成任何有效的遗忘样本，请检查数据和逻辑。")

    logging.info(f"💾 成功生成 {len(forget_samples)} 条遗忘样本 (每条遗忘 {forget_percentage*100:.0f}% 交互)，正在保存...")
    with open(OUTPUT_FORGET_FILE, 'w') as f:
        json.dump(forget_samples, f, indent=2)

    # 5. 生成保留集样本 (他们也来自黄金池)
    retain_samples = []
    for user_id in retain_user_ids:
        user_history = train_df[train_df['mapped_user_id'] == user_id]['item_id'].tolist()
        if len(user_history) > 1:
            retain_samples.append({
                "user_id": rev_user_map.get(user_id, ""),
                "seq_items": user_history
            })
    
    logging.info(f"💾 成功生成 {len(retain_samples)} 条保留样本，正在保存...")
    with open(OUTPUT_RETAIN_FILE, 'w') as f:
        json.dump(retain_samples, f, indent=2)
    logging.info(f"✅ 数据集准备完成。")
    
    return test_df

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dual-memory adapter and router")
    parser.add_argument("--model", required=True, help="Path to base P5 checkpoint (.pt)")
    parser.add_argument(
        "--forget",
        default=str(OUTPUT_FORGET_FILE),
        help="Path to forget samples JSON",
    )
    parser.add_argument(
        "--retain",
        default=str(OUTPUT_RETAIN_FILE),
        help="Path to retain samples JSON",
    )
    parser.add_argument(
        "--output",
        default="results/dual_memory",
        help="Directory to store artifacts",
    )
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=8, help="Dual-memory batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Dual-memory epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Dual-memory learning rate")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader worker processes (default: 8)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) training")
    parser.add_argument(
        "--beta-weight", type=float, default=3.0, help="Weight for the forgetting loss term."
    )
    parser.add_argument(
        "--alpha-weight", type=float, default=1.0, help="Weight for the retention loss term."
    )
    parser.add_argument(
        "--kl-reg-weight", type=float, default=10.0, help="Weight for KL-divergence regularization."
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Max norm for gradient clipping")

    parser.add_argument("--forget-ratio", type=float, default=1.0, help="Subset ratio for forget set")
    parser.add_argument("--retain-ratio", type=float, default=1.0, help="Subset ratio for retain set")
    parser.add_argument("--max-input-length", type=int, default=256, help="Prompt token limit")
    parser.add_argument("--max-target-length", type=int, default=32, help="Target token limit")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--router-hidden", type=int, default=64, help="Router hidden dim")
    parser.add_argument("--router-dropout", type=float, default=0.1, help="Router dropout")
    parser.add_argument("--router-lr", type=float, default=1e-3, help="Router learning rate")
    parser.add_argument("--router-weight-decay", type=float, default=1e-4, help="Router weight decay")
    parser.add_argument("--router-epochs", type=int, default=40, help="Router training epochs")
    parser.add_argument("--router-batch-size", type=int, default=64, help="Router batch size")
    parser.add_argument(
        "--router-target-precision",
        type=float,
        default=0.8,
        help="Desired precision threshold when deriving router cutoff",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.1,
        help="Weight for KL alignment loss on the batch (default: 0.1)",
    )
    # === 新增：更强忘记与稳定性控制参数 ===
    parser.add_argument(
        "--unlikelihood-weight",
        type=float,
        default=0.0,
        help="Weight for unlikelihood loss on forgotten targets (default: 0.0 to disable)",
    )
    parser.add_argument(
        "--pairwise-weight",
        type=float,
        default=0.0,
        help="Weight for pairwise margin ranking loss (default: 0.0 to disable)",
    )
    parser.add_argument(
        "--pairwise-margin",
        type=float,
        default=1.0,
        help="Margin m in pairwise loss max(0, m + s_forgot - s_neg) (default: 1.0)",
    )
    parser.add_argument(
        "--hard-neg-k",
        type=int,
        default=50,
        help="Top-K hard negatives from main model logits for pairwise loss (default: 50)",
    )
    parser.add_argument(
        "--kl-mask-forgotten",
        action="store_true",
        help="If set, do not apply KL regularization on forgotten samples (mask them out).",
    )
    parser.add_argument(
        "--freeze-lm-head",
        action="store_true",
        help="Freeze lm_head (no LoRA/grad) to reduce collateral drift.",
    )
    parser.add_argument(
        "--topk-penalty-weight",
        type=float,
        default=0.0,
        help="Weight for TopK containment penalty to suppress forgotten target within side top-K (default: 0.0)",
    )
    parser.add_argument(
        "--topk-k",
        type=int,
        default=20,
        help="K for TopK containment penalty (default: 20)",
    )
    parser.add_argument(
        "--topk-margin",
        type=float,
        default=0.0,
        help="Margin for TopK containment penalty (default: 0.0)",
    )
    # 绝对阈值压制（将被遗忘目标的logit压到固定负阈值以下）
    parser.add_argument(
        "--abs-suppression-weight",
        type=float,
        default=0.0,
        help="Weight for absolute suppression penalty on forgotten target logits (default: 0.0 to disable)",
    )
    parser.add_argument(
        "--abs-suppression-margin",
        type=float,
        default=3.0,
        help="Margin m for absolute suppression: penalize max(0, m + logit)^2 so that logit <= -m (default: 3.0)",
    )
    parser.add_argument(
        "--edit-layer",
        action="append",
        default=None,
        help="Layer to instrument with LoRA (repeatable)",
    )
    parser.add_argument(
        "--router-layer",
        action="append",
        default=None,
        help="Layer to capture for router features (repeatable)",
    )
    parser.add_argument(
        "--init-artifacts",
        default=None,
        help="Existing dual-memory artifacts for warm start",
    )
    parser.add_argument(
        "--activation-margin",
        type=float,
        default=5.0,
        help="Margin for activation difference between side/main on forget samples",
    )
    parser.add_argument(
        "--force-regenerate-data",
        action="store_true",
        help="如果指定，将强制重新生成遗忘/保留数据集，覆盖现有文件。"
    )
    # 预设配置，便于一键复现实验方案
    parser.add_argument(
        "--preset",
        type=str,
        choices=["e1_stable", "e2_strong"],
        default=None,
        help="可选超参预设: e1_stable(稳态优先) | e2_strong(更强遗忘)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # [集成功能] 在训练前，自动调用数据准备函数
    test_df = prepare_guaranteed_datasets(force_regenerate=args.force_regenerate_data)

    # 根据预设覆盖关键超参（命令行显式传入的参数仍可覆盖这些值）
    if args.preset == "e1_stable":
        logging.info("⚙️ 使用预设: e1_stable (稳态优先)")
        # 稳定保留效用，抑制 NDCG 漏网高位
        if args.pairwise_weight == 0.0: args.pairwise_weight = 0.6
        if args.pairwise_margin == 1.0: args.pairwise_margin = 1.5
        if args.kl_reg_weight == 10.0: args.kl_reg_weight = 15.0
        if args.alpha_weight == 1.0: args.alpha_weight = 40.0
        if args.beta_weight == 3.0: args.beta_weight = 120.0
        if args.unlikelihood_weight == 0.0: args.unlikelihood_weight = 1.0
        if args.hard_neg_k == 50: args.hard_neg_k = 50
        # 强制打开两项稳态开关
        args.kl_mask_forgotten = True
        args.freeze_lm_head = True
    elif args.preset == "e2_strong":
        logging.info("⚙️ 使用预设: e2_strong (更强遗忘)")
        # 更强压制漏网项，允许更大幅度的遗忘
        if args.pairwise_weight == 0.0: args.pairwise_weight = 1.2
        if args.pairwise_margin == 1.0: args.pairwise_margin = 2.0
        if args.kl_reg_weight == 10.0: args.kl_reg_weight = 12.0
        if args.alpha_weight == 1.0: args.alpha_weight = 35.0
        if args.beta_weight == 3.0: args.beta_weight = 150.0
        if args.unlikelihood_weight == 0.0: args.unlikelihood_weight = 1.5
        if args.hard_neg_k == 50: args.hard_neg_k = 50
        # 开启绝对阈值压制（可覆盖）
        if args.abs_suppression_weight == 0.0: args.abs_suppression_weight = 1.5
        if args.abs_suppression_margin == 3.0: args.abs_suppression_margin = 3.0
        args.kl_mask_forgotten = True
        args.freeze_lm_head = True

    logging.info("--- [步骤 2/3] 开始双记忆模型训练 ---")
    config = DualMemoryConfig(
        model_path=args.model,
        forget_path=args.forget,
        retain_path=str(OUTPUT_RETAIN_FILE),
        output_dir=args.output,
        device=args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu",
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gradient_clip=args.gradient_clip,
        forget_ratio=args.forget_ratio,
        retain_ratio=args.retain_ratio,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        # 关键修复：将 CLI 的 alpha/beta 正确传递给训练器使用的字段
        alpha_weight=args.alpha_weight,
        beta_weight=args.beta_weight,
    dataloader_num_workers=args.workers,
    use_amp=bool(args.amp),
        router_hidden_dim=args.router_hidden,
        router_dropout=args.router_dropout,
        router_lr=args.router_lr,
        router_weight_decay=args.router_weight_decay,
        router_epochs=args.router_epochs,
        router_batch_size=args.router_batch_size,
        router_target_precision=args.router_target_precision,
        retain_kl_weight=args.alpha_weight, # 仅用于兼容性保留（训练器不读取该字段）
        kl_reg_weight=args.kl_reg_weight,
        activation_separation_weight=0.0,
        forget_suppression_weight=args.beta_weight, # 仅用于兼容性保留（训练器不读取该字段）
        # 新增字段传递
        unlikelihood_weight=args.unlikelihood_weight,
        pairwise_weight=args.pairwise_weight,
        pairwise_margin=args.pairwise_margin,
        hard_neg_k=args.hard_neg_k,
        kl_mask_forgotten=args.kl_mask_forgotten,
        freeze_lm_head=args.freeze_lm_head,
        topk_penalty_weight=args.topk_penalty_weight,
        topk_k=args.topk_k,
        topk_margin=args.topk_margin,
        abs_suppression_weight=args.abs_suppression_weight,
        abs_suppression_margin=args.abs_suppression_margin,
    )

    if args.edit_layer: config.edit_target_layers = args.edit_layer
    if args.router_layer: config.router_target_layers = args.router_layer

    logging.info("训练配置: %s", json.dumps(config.to_dict(), indent=2))

    init_state: Optional[Dict[str, Any]] = None
    if args.init_artifacts:
        init_path = Path(args.init_artifacts)
        if not init_path.exists():
            raise FileNotFoundError(f"Init artifacts not found: {init_path}")
        logging.info("Loading initial artifacts from %s", init_path)
        init_state = torch.load(init_path, map_location="cpu")

    artifacts = train_dual_memory(config, initial_state=init_state, test_df_for_sampling=test_df)
    logging.info("--- [步骤 3/3] 双记忆模型训练完成 ---")
    logging.info("✅ 训练产物已保存。路由器指标: %s", artifacts.router_metrics)

if __name__ == "__main__":
    main()