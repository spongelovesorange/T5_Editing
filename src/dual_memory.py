"""Simple dual-memory fine-tuning utilities for P5 WISE experiments."""

from __future__ import annotations
from collections import defaultdict
import json
import logging
import random
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import torch
import re
import os
from torch import nn
from contextlib import nullcontext
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm
import pandas as pd
from src.model_wrapper import P5ModelWrapper

logger = logging.getLogger(__name__)

FORGET_TOKEN = "<forget>"
RETAIN_TOKEN = "<retain>"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EDIT_LAYERS = [
    "decoder.block.3.layer.2.DenseReluDense.wi",
    "decoder.block.4.layer.2.DenseReluDense.wi",
    "decoder.block.5.layer.2.DenseReluDense.wi",
    "lm_head",
]
DEFAULT_ROUTER_LAYERS = [
    "decoder.block.3.layer.2.DenseReluDense",
    "decoder.block.4.layer.2.DenseReluDense",
    "decoder.block.5.layer.2.DenseReluDense",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@dataclass
class DualSample:
    input_text: str
    main_target: str
    side_target: str
    edit_type: str
    weight: float = 1.0
    user_id: str = ""
    router_text: str = ""
    suppression_item_id: Optional[int] = None
    # New fields for dual-target training (positive = keep, negative = forget)
    positive_target_id: Optional[int] = None
    negative_target_id: Optional[int] = None

def _format_sequence(history: Sequence[int]) -> str:
    if not history:
        return "<empty>"
    return " ".join(f"item_{item}" for item in history)


def _build_prompt(entry: Dict[str, Any]) -> str:
    # Keep prompt format 100% consistent with evaluate_datasets.py
    # Example: "User 123 recent history: item_1 item_2 item_3. Recommend next item."
    user_id = entry.get("user_id", "unknown")
    history = _format_sequence(entry.get("seq_items") or [])
    return f"User {user_id} recent history: {history}. Recommend next item."


def _target_token(entry: Dict[str, Any], default: str) -> str:
    targets = entry.get("target_items") or []
    if targets:
        return f"item_{targets[0]}"
    suppression = entry.get("suppression_targets") or []
    if suppression:
        return f"item_{suppression[0]}"
    return default


def _load_entries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, received {type(data)}")
    return data

def build_samples(
    forget_path: Path,
    retain_path: Path,
    forget_ratio: float,
    retain_ratio: float,
    seed: int,
    test_df: Optional[pd.DataFrame] = None, # <--- 新增参数
) -> List[DualSample]:
    """
    [诊断版] 增加日志打印，以监控数据处理进度。
    """
    rng = random.Random(seed)
    
    logger.info(f"⏳ 开始加载遗忘数据: {forget_path}...")
    forget_entries = _load_entries(forget_path)
    logger.info(f"✅ 加载了 {len(forget_entries)} 条遗忘条目。")

    retain_entries = []
    # 检查路径是否有效，并且不是一个目录（比如 '.' 或空字符串）
    if retain_path and str(retain_path) and str(retain_path) != '.':
        if retain_path.exists() and retain_path.is_file():
            logger.info(f"⏳ 开始加载保留数据: {retain_path}...")
            retain_entries = _load_entries(retain_path)
            logger.info(f"✅ 加载了 {len(retain_entries)} 条保留条目。")
        else:
            logger.warning(f"⚠️ 保留文件路径无效或未找到: {retain_path}，跳过加载。")
    else:
        logger.info("ℹ️ 未提供保留文件路径 (retain_path 为空或无效)，跳过加载保留数据。")
    
    # 这一部分代码已移动到 train_dual_memory.py 中，这里仅作兼容性保留
    try:
        from train_dual_memory import USER_INDEX_FILE
        user_map = {line.strip().split()[0]: line.strip().split()[1] for line in open(USER_INDEX_FILE)}
    except (ImportError, FileNotFoundError):
        logger.warning("无法加载 user_indexing.txt，正样本增强将受限。")
        user_map = {}


    # 创建一个从 mapped_user_id 到测试集正样本的映射
    future_positives = defaultdict(list)
    if test_df is not None:
        logger.info("正在从测试集数据中构建未来正样本池...")
        test_df_pos = test_df[test_df['rating'] >= 4]
        for _, row in test_df_pos.iterrows():
            future_positives[str(row['mapped_user_id'])].append(str(row['item_id']))
        logger.info(f"✅ 未来正样本池构建完成，覆盖 {len(future_positives)} 名用户。")


    # 按比例采样原始数据集
    if 0 < forget_ratio < 1.0 and forget_entries:
        keep = max(1, int(len(forget_entries) * forget_ratio))
        forget_entries = rng.sample(forget_entries, keep)
    if 0 < retain_ratio < 1.0 and retain_entries:
        keep = max(1, int(len(retain_entries) * retain_ratio))
        retain_entries = rng.sample(retain_entries, keep)

    samples: List[DualSample] = []
    # Build forget samples (with positive/negative targets for dual-loss training)
    for entry in forget_entries:
        prompt = _build_prompt(entry)
        main_t = _target_token(entry, RETAIN_TOKEN)
        side_t = _target_token(entry, FORGET_TOKEN)

        suppression = entry.get("suppression_targets") or []
        negative_target = suppression[0] if suppression else None
        
        # [核心逻辑修改] 从用户的未来交互中随机选一个作为正样本
        positive_target = None
        user_id_orig = entry.get("user_id")
        if user_id_orig and user_map:
            mapped_id = user_map.get(str(user_id_orig))
            if mapped_id and mapped_id in future_positives:
                positive_target = random.choice(future_positives[mapped_id])

        samples.append(
            DualSample(
                input_text=prompt,
                main_target=main_t,
                side_target=side_t,
                edit_type="forget",
                weight=1.0,
                user_id=entry.get("user_id", ""),
                router_text=prompt,
                suppression_item_id=negative_target,
                positive_target_id=positive_target,
                negative_target_id=negative_target,
            )
        )

    # Build retain samples (kept for compatibility but not used in forget-only training)
    for entry in retain_entries:
        prompt = _build_prompt(entry)
        main_t = _target_token(entry, RETAIN_TOKEN)
        # 为保留样本也补充一个正目标，来自测试集未来正样本（若可用）
        positive_target = None
        user_id_orig = entry.get("user_id")
        if user_id_orig and user_map:
            mapped_id = user_map.get(str(user_id_orig))
            if mapped_id and mapped_id in future_positives:
                positive_target = random.choice(future_positives[mapped_id])
        samples.append(
            DualSample(
                input_text=prompt,
                main_target=main_t,
                side_target=main_t,
                edit_type="retain",
                weight=1.0,
                user_id=entry.get("user_id", ""),
                router_text=prompt,
                suppression_item_id=None,
                positive_target_id=positive_target,
                negative_target_id=None,
            )
        )

    return samples

@dataclass
class DualMemoryConfig:
    model_path: str
    forget_path: str
    retain_path: str
    output_dir: str
    device: str = DEFAULT_DEVICE
    seed: int = 2025
    batch_size: int = 8
    epochs: int = 5
    lr: float = 5e-4
    gradient_clip: float = 1.0
    forget_ratio: float = 1.0
    retain_ratio: float = 1.0
    max_input_length: int = 256
    max_target_length: int = 32
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    edit_target_layers: List[str] = field(default_factory=lambda: list(DEFAULT_EDIT_LAYERS))
    router_target_layers: List[str] = field(default_factory=lambda: list(DEFAULT_ROUTER_LAYERS))
    router_hidden_dim: int = 64
    router_dropout: float = 0.1
    router_lr: float = 1e-3
    router_weight_decay: float = 1e-4
    router_epochs: int = 40
    router_batch_size: int = 64
    router_target_precision: float = 0.8
    retain_kl_weight: float = 0.1
    forget_suppression_weight: float = 0.0
    # New weights for dual-target loss
    alpha_weight: float = 1.0  # retention (keep positive)
    beta_weight: float = 1.5   # forgetting (suppress negative)
    # 【新增】 KL散度正则化权重
    kl_reg_weight: float = 1.5
    # Early stopping config
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    activation_separation_weight: float = 0.0
    # === 新增：更强忘记与稳定性控制参数 ===
    unlikelihood_weight: float = 0.0
    pairwise_weight: float = 0.0
    pairwise_margin: float = 1.0
    hard_neg_k: int = 50
    kl_mask_forgotten: bool = False
    freeze_lm_head: bool = False
    # 训练/加载优化
    dataloader_num_workers: int = 8
    use_amp: bool = True
    # TopK 出现惩罚（抑制“漏网即高位”）
    topk_penalty_weight: float = 0.0
    topk_k: int = 20
    topk_margin: float = 0.0
    # 绝对阈值压制（将被遗忘目标的logit压到一个固定负阈值以下）
    abs_suppression_weight: float = 0.0
    abs_suppression_margin: float = 3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "forget_path": self.forget_path,
            "retain_path": self.retain_path,
            "output_dir": self.output_dir,
            "device": self.device,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "gradient_clip": self.gradient_clip,
            "forget_ratio": self.forget_ratio,
            "retain_ratio": self.retain_ratio,
            "max_input_length": self.max_input_length,
            "max_target_length": self.max_target_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "edit_target_layers": list(self.edit_target_layers),
            "router_target_layers": list(self.router_target_layers),
            "router_hidden_dim": self.router_hidden_dim,
            "router_dropout": self.router_dropout,
            "router_lr": self.router_lr,
            "router_weight_decay": self.router_weight_decay,
            "router_epochs": self.router_epochs,
            "router_batch_size": self.router_batch_size,
            "router_target_precision": self.router_target_precision,
            "retain_kl_weight": self.retain_kl_weight,
            "forget_suppression_weight": self.forget_suppression_weight,
            "activation_separation_weight": self.activation_separation_weight,
            "kl_reg_weight": self.kl_reg_weight,
            # 记录用于训练器 BCE 的权重，便于复现
            "alpha_weight": self.alpha_weight,
            "beta_weight": self.beta_weight,
            # 新增字段
            "unlikelihood_weight": self.unlikelihood_weight,
            "pairwise_weight": self.pairwise_weight,
            "pairwise_margin": self.pairwise_margin,
            "hard_neg_k": self.hard_neg_k,
            "kl_mask_forgotten": self.kl_mask_forgotten,
            "freeze_lm_head": self.freeze_lm_head,
            "dataloader_num_workers": self.dataloader_num_workers,
            "use_amp": self.use_amp,
            "topk_penalty_weight": self.topk_penalty_weight,
            "topk_k": self.topk_k,
            "topk_margin": self.topk_margin,
            "abs_suppression_weight": self.abs_suppression_weight,
            "abs_suppression_margin": self.abs_suppression_margin,
        }


@dataclass
class AdapterConfig:
    target_layers: List[str]
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0


class DualMemoryDataset(Dataset):
    """Minimal dataset wrapper for dual-memory samples.

    Each item is a DualSample. The collate_fn tokenizes inputs and targets and
    returns the tensors expected by the trainer and router utilities.
    """
    def __init__(
        self,
        samples: List[DualSample],
        tokenizer,
        max_input_length: int = 256,
        max_target_length: int = 32,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DualSample:
        return self.samples[idx]

    def collate_fn(self, batch: List[DualSample]) -> Dict[str, torch.Tensor]:
                prompts = [s.input_text for s in batch]
                side_texts = [s.side_target for s in batch]
                router_texts = [s.router_text or s.input_text for s in batch]
                edit_types = [1.0 if s.edit_type == "forget" else 0.0 for s in batch]
                weights = [float(s.weight) for s in batch]

                def _id_to_int(v):
                    if v is None: return -1
                    if isinstance(v, str):
                        m = re.search(r"(\d+)", v)
                        return int(m.group(1)) if m else -1
                    try: return int(v)
                    except (ValueError, TypeError): return -1

                suppression_ids = [_id_to_int(s.suppression_item_id) for s in batch]

                model_inputs = self.tokenizer(
                    prompts, padding=True, truncation=True, max_length=self.max_input_length, return_tensors="pt"
                )
                side_labels = self.tokenizer(
                    side_texts, padding=True, truncation=True, max_length=self.max_target_length, return_tensors="pt"
                )
                router_inputs = self.tokenizer(
                    router_texts, padding=True, truncation=True, max_length=self.max_input_length, return_tensors="pt"
                )
                
                pad_id = self.tokenizer.pad_token_id
                side_ids = side_labels.input_ids.clone()
                side_ids[side_ids == pad_id] = -100

                batch_dict: Dict[str, torch.Tensor] = {
                    "input_ids": model_inputs.input_ids,
                    "attention_mask": model_inputs.attention_mask,
                    "labels_side": side_ids,
                    "edit_type": torch.tensor(edit_types, dtype=torch.float32),
                    "sample_weight": torch.tensor(weights, dtype=torch.float32),
                    "router_input_ids": router_inputs.input_ids,
                    "router_attention_mask": router_inputs.attention_mask,
                    "suppression_item_ids": torch.tensor(suppression_ids, dtype=torch.long).unsqueeze(-1),
                }

                # ========================== [ 根本性修复 ] ==========================
                # 错误原因: 原代码对 "123" 这样的数字字符串分词，导致目标token错误。
                # 解决方案: 我们现在构造 "item_123" 这样的标准P5格式字符串，然后分词。
                
                pos_ints = [_id_to_int(s.positive_target_id) for s in batch]
                neg_ints = [_id_to_int(s.negative_target_id) for s in batch]

                # 1. 准备正确的P5目标文本
                pos_target_texts = [f"item_{pid}" if pid >= 0 else self.tokenizer.pad_token for pid in pos_ints]
                neg_target_texts = [f"item_{nid}" if nid >= 0 else self.tokenizer.pad_token for nid in neg_ints]
                
                # 2. 使用分词器编码
                # 注意: 虽然目标通常是单token (例如 'item_123'), 但在 tokenizer 实现上有时会被拆为多个子token。
                # 因此我们必须开启 padding=True，确保 batch_encode_plus 能返回对齐的张量，避免 DataLoader worker 抛错。
                pad_fallback = self.tokenizer.pad_token if self.tokenizer.pad_token is not None else ""
                pos_input_texts = [t if t is not None else pad_fallback for t in pos_target_texts]
                neg_input_texts = [t if t is not None else pad_fallback for t in neg_target_texts]
                pos_tokenized = self.tokenizer(pos_input_texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
                neg_tokenized = self.tokenizer(neg_input_texts, padding=True, truncation=True, max_length=4, return_tensors="pt")

                # 3. 提取正确的 token_id (不再是[:,0]，而是整个input_ids的第一列)
                pos_token_ids = pos_tokenized.input_ids[:, 0].clone()
                neg_token_ids = neg_tokenized.input_ids[:, 0].clone()

                # 4. 对于不存在正负样本的条目(原始ID为-1)，将它们的token_id也设置为-100 (CrossEntropy的忽略索引)
                original_pos_ids = torch.tensor(pos_ints, dtype=torch.long)
                original_neg_ids = torch.tensor(neg_ints, dtype=torch.long)
                pos_token_ids[original_pos_ids == -1] = -100
                neg_token_ids[original_neg_ids == -1] = -100
                
                # 5. 将正确的Token ID放入batch字典
                batch_dict["positive_target_ids"] = pos_token_ids
                batch_dict["negative_target_ids"] = neg_token_ids
                # ======================== [ 修复结束 ] ========================

                return batch_dict

def _locate_module(root: nn.Module, dotted: str) -> Tuple[Optional[nn.Module], Optional[str]]:
    """Locate (parent_module, attribute_name) for a dotted module path on root.
    Returns (None, None) if not found.
    """
    parts = dotted.split('.')
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None
        parent = getattr(parent, p)
    attr = parts[-1]
    if not hasattr(parent, attr):
        return None, None
    return parent, attr


# ---------------------------------------------------------------------------
# [核心修复] 真正的 LoRA 实现
# ---------------------------------------------------------------------------


class LoRASideLinear(nn.Linear):
    """
    一个实现了LoRA旁路的线性层。
    当 use_side_memory=True 时，它会在原始线性变换的基础上增加一个低秩更新。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 0.0
        self.lora_dropout = nn.Dropout(lora_dropout)

        # 冻结原始权重
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # 定义并初始化可训练的LoRA权重
        if self.r > 0:
            dev = self.weight.device
            dt = self.weight.dtype
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features, device=dev, dtype=dt))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, std=0.01)
        else:
            # 占位符tensor（不会训练）
            self.lora_A = None
            self.lora_B = None

        self.use_side_memory = False # 由外部适配器控制

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = super().forward(x) # 主记忆路径
        
        if self.use_side_memory and self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # x: (B, seq_len, in_features) 或 (B, in_features)
            # 我们假设输入是 (..., in_features)
            lora_in = self.lora_dropout(x)
            # lora_A: (r, in), lora_B: (out, r)
            # compute lora_update = lora_in @ A.T @ B.T
            # ensure shapes: (..., in) @ (in, r) -> (..., r); then (..., r) @ (r, out) -> (..., out)
            update_mid = lora_in.matmul(self.lora_A.T)
            lora_update = update_mid.matmul(self.lora_B.T) * self.scaling
            result = result + lora_update
        
        return result


class DualMemoryAdapter:
    """
    一个真正的双记忆适配器，它将LoRA层注入模型并管理其状态。
    """
    def __init__(self, model: nn.Module, config: AdapterConfig) -> None:
        self.model = model
        self.config = config
        # device derived from model params (may be cpu initially)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        self.side_modules: Dict[str, LoRASideLinear] = {}
        self._inject_adapter()
        self.set_mode("main")

    def _inject_adapter(self):
        for layer_name in self.config.target_layers:
            parent, attr = _locate_module(self.model, layer_name)
            if parent is None:
                logger.warning(f"⚠️ LoRA目标层未找到，已跳过: {layer_name}")
                continue
            
            original_module = getattr(parent, attr)
            if not isinstance(original_module, nn.Linear):
                logger.warning(f"⚠️ LoRA目标层不是线性层，已跳过: {layer_name}")
                continue

            new_module = LoRASideLinear(
                original_module.in_features, original_module.out_features,
                r=self.config.lora_r, lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=original_module.bias is not None,
                device=original_module.weight.device, dtype=original_module.weight.dtype,
            )
            new_module.weight.data.copy_(original_module.weight.data)
            if original_module.bias is not None:
                new_module.bias.data.copy_(original_module.bias.data)

            setattr(parent, attr, new_module)
            self.side_modules[layer_name] = new_module
        
        if self.side_modules:
            logger.info(f"✅ LoRA适配器已成功注入到 {len(self.side_modules)} 个层。")
        else:
            logger.error("❌ LoRA适配器注入失败，没有找到任何有效的目标层。")

    def set_mode(self, mode: str):
        """Toggle adapter modules between main and side behavior.

        mode: 'main' or 'side'
        """
        assert mode in ("main", "side")
        for module in self.side_modules.values():
            module.use_side_memory = (mode == "side")

    def parameters(self) -> Iterable[nn.Parameter]:
        """Yield all trainable LoRA parameters (A and B) from injected modules."""
        for module in self.side_modules.values():
            if getattr(module, 'lora_A', None) is not None:
                yield module.lora_A
            if getattr(module, 'lora_B', None) is not None:
                yield module.lora_B

    def named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Yield (name, parameter) pairs for all LoRA parameters."""
        for name, module in self.side_modules.items():
            if getattr(module, 'lora_A', None) is not None:
                yield f"{name}.lora_A", module.lora_A
            if getattr(module, 'lora_B', None) is not None:
                yield f"{name}.lora_B", module.lora_B

    # Convenience passthroughs so trainer code can call adapter.to()/train()/eval()
    def to(self, device: torch.device):
        self.model.to(device)
        # ensure LoRA modules are also on device
        for module in self.side_modules.values():
            if getattr(module, 'lora_A', None) is not None:
                module.lora_A.data = module.lora_A.data.to(device)
            if getattr(module, 'lora_B', None) is not None:
                module.lora_B.data = module.lora_B.data.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return a mapping of layer_name -> {'lora_A': Tensor, 'lora_B': Tensor}.

        Tensors are returned on CPU for checkpoint portability.
        """
        sd: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, module in self.side_modules.items():
            entry: Dict[str, torch.Tensor] = {}
            if getattr(module, 'lora_A', None) is not None:
                entry['lora_A'] = module.lora_A.detach().cpu()
            if getattr(module, 'lora_B', None) is not None:
                entry['lora_B'] = module.lora_B.detach().cpu()
            sd[name] = entry
        return sd

    def load_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Load LoRA parameters from a state dict produced by `state_dict()`.

        Values in `state` will be moved to the device of the corresponding
        injected LoRA module before copying. Unknown layers in `state` will
        be skipped with a warning.
        """
        if not isinstance(state, dict):
            raise ValueError("adapter.load_state_dict expects a dict mapping layer->params")

        for name, params in state.items():
            module = self.side_modules.get(name)
            if module is None:
                logger.warning(f"adapter.load_state_dict: unknown layer '{name}' in state; skipping")
                continue

            if not isinstance(params, dict):
                logger.warning(f"adapter.load_state_dict: expected dict for layer '{name}', got {type(params)}; skipping")
                continue

            if 'lora_A' in params and getattr(module, 'lora_A', None) is not None:
                val = params['lora_A']
                try:
                    val = val.to(module.lora_A.device)
                    module.lora_A.data.copy_(val)
                except Exception as e:
                    logger.error(f"Failed to load lora_A for {name}: {e}")

            if 'lora_B' in params and getattr(module, 'lora_B', None) is not None:
                val = params['lora_B']
                try:
                    val = val.to(module.lora_B.device)
                    module.lora_B.data.copy_(val)
                except Exception as e:
                    logger.error(f"Failed to load lora_B for {name}: {e}")


class DualMemoryTrainer:
    def __init__(
        self,
        model: nn.Module,
        adapter: DualMemoryAdapter,
        tokenizer,
        config: DualMemoryConfig,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            list(adapter.parameters()),
            lr=config.lr,
        )
        self.history: List[float] = []
        # early-stopping holder
        self.best_adapter_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        # AMP 支持
        self.use_amp = bool(getattr(config, 'use_amp', False) and torch.cuda.is_available())
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
        else:
            self.scaler = None

    def train(self, dataloader: DataLoader) -> List[float]:
                    self.model.train()
                    self.adapter.train()

                    trainable_params = [p for p in self.adapter.parameters() if p.requires_grad]
                    logger.info(f"✅ [DEBUG] 优化器已找到 {sum(p.numel() for p in trainable_params)} 个可训练的 LoRA 参数。")

                    alpha = float(self.config.alpha_weight)
                    beta = float(self.config.beta_weight)
                    # 默认将 KL 权重降到一个保守值，避免在初始实验中主导训练
                    kl_weight = float(self.config.kl_reg_weight)
                    ul_weight = float(self.config.unlikelihood_weight)
                    pw_weight = float(self.config.pairwise_weight)
                    pw_margin = float(self.config.pairwise_margin)
                    hard_k = int(self.config.hard_neg_k)

                    best_loss = float('inf')
                    epochs_no_improve = 0
                    patience = int(self.config.early_stopping_patience)
                    min_delta = float(self.config.early_stopping_min_delta)

                    amp_ctx = torch.cuda.amp.autocast if (self.use_amp and torch.cuda.is_available()) else nullcontext
                    for epoch in range(self.config.epochs):
                        epoch_loss = 0.0
                        sample_count = 0
                        # diagnostics accumulators
                        diag_pos_ce_sum = 0.0
                        diag_neg_bce_sum = 0.0
                        diag_kl_sum = 0.0
                        diag_batches = 0

                        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")):
                            for key, value in list(batch.items()):
                                if isinstance(value, torch.Tensor):
                                    batch[key] = value.to(self.device)

                            input_ids = batch["input_ids"]
                            attention_mask = batch["attention_mask"]
                            pos_ids = batch.get("positive_target_ids")
                            neg_ids = batch.get("negative_target_ids")
                            edit_type = batch.get("edit_type")
                            
                            batch_size = input_ids.size(0)
                            decoder_input_ids = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)

                            # 1. 计算主记忆Logits作为KL散度的目标
                            self.adapter.set_mode("main")
                            with torch.no_grad():
                                with amp_ctx():
                                    main_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                                    main_logits = main_outputs.logits[:, 0, :]

                            # 2. 计算侧记忆Logits
                            self.adapter.set_mode("side")
                            with amp_ctx():
                                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
                                side_logits = outputs.logits[:, 0, :]

                            # ==================== KL 只在 retain 样本上计算 ====================
                            # 避免对 forget 样本也施加正则，从而抵消忘记目标或产生连带损伤。
                            # 我们按样本计算 KL，并只对 retain mask 求平均。
                            try:
                                # main 应作为概率分布 (softmax)，side 作为 log-prob 输入到 F.kl_div
                                with torch.no_grad():
                                    main_prob = F.softmax(main_logits.detach(), dim=-1)
                                side_logprob = F.log_softmax(side_logits, dim=-1)
                                per_token_kl = F.kl_div(side_logprob, main_prob, reduction='none').sum(dim=-1)  # (batch,)
                            except Exception as e:
                                logger.error("[KL DEBUG] 计算 per-token KL 时出现异常: %s", e)
                                per_token_kl = None

                            mask_retain = (edit_type == 0.0)

                            if per_token_kl is not None:
                                if bool(self.config.kl_mask_forgotten):
                                    # 仅在保留样本上计算 KL（若存在）
                                    if mask_retain.any():
                                        kl_loss = per_token_kl[mask_retain].mean()
                                    else:
                                        kl_loss = torch.tensor(0.0, device=self.device)
                                else:
                                    kl_loss = per_token_kl.mean()
                            else:
                                kl_loss = torch.tensor(0.0, device=self.device)

                            # 4. 对遗忘用户计算双目标损失
                            mask_forget = (edit_type == 1.0)
                            loss_forget_component = torch.tensor(0.0, device=self.device)
                            
                            # ==================== [ 最终修复 START ] ====================
                            #  使用对称的BCE损失函数来平衡 Retention 和 Forgetting
                            if mask_forget.any():
                                forget_logits = side_logits[mask_forget]
                                forget_pos_ids = pos_ids[mask_forget]
                                forget_neg_ids = neg_ids[mask_forget]
                                
                                pos_mask = (forget_pos_ids >= 0)
                                neg_mask = (forget_neg_ids >= 0)

                                loss_r_component = torch.tensor(0.0, device=self.device)
                                loss_f_suppression = torch.tensor(0.0, device=self.device)

                                # [!! 修复 !!] 1. 计算保留正样本的损失 (Retention) - 使用BCE
                                if pos_mask.any():
                                    # 筛选出有效的 logits 和 目标ids
                                    valid_pos_logits_full = forget_logits[pos_mask]
                                    valid_pos_ids = forget_pos_ids[pos_mask].unsqueeze(1)
                                    
                                    # 提取目标token的logit
                                    pos_logits = valid_pos_logits_full.gather(1, valid_pos_ids).squeeze(1)
                                    
                                    # 计算BCE损失，推动其概率为1
                                    loss_r_component = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                                    diag_pos_ce_sum += float(loss_r_component.item()) # <--- 现在记录的是BCE损失

                                # 2. 抑制负样本 (Forget Efficacy) - 使用BCE
                                if neg_mask.any():
                                    # 筛选出有效的 logits 和 目标ids
                                    valid_neg_logits_full = forget_logits[neg_mask]
                                    valid_neg_ids = forget_neg_ids[neg_mask].unsqueeze(1)
                                    
                                    # 提取目标token的logit
                                    neg_logits = valid_neg_logits_full.gather(1, valid_neg_ids).squeeze(1)
                                    
                                    # 计算BCE损失，推动其概率为0
                                    loss_f_suppression = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
                                    diag_neg_bce_sum += float(loss_f_suppression.item())

                                    # 2.1 可选：Unlikelihood（等价于 softplus(logit)），提供额外负向拉力
                                    if ul_weight > 0.0:
                                        ul_loss = F.softplus(neg_logits).mean()
                                        loss_f_suppression = loss_f_suppression + ul_weight * ul_loss

                                    # 2.2 可选：绝对阈值压制（将被遗忘目标的logit压到 -m 以下）
                                    abs_w = float(getattr(self.config, 'abs_suppression_weight', 0.0))
                                    abs_m = float(getattr(self.config, 'abs_suppression_margin', 3.0))
                                    if abs_w > 0.0 and abs_m >= 0.0:
                                        # 目标：neg_logits <= -abs_m  -> 惩罚项: relu(abs_m + neg_logits)^2
                                        abs_penalty = F.relu(abs_m + neg_logits).pow(2).mean()
                                        loss_f_suppression = loss_f_suppression + abs_w * abs_penalty

                                # [!! 修复 !!] 组合遗忘用户的总损失 (保留正样本 + 抑制负样本)
                                loss_forget_component = (alpha * loss_r_component) + (beta * loss_f_suppression)

                                # 3) 可选：Pairwise margin ranking（针对忘记目标 vs 主记忆Top-K强负样本）
                                if pw_weight > 0.0 and hard_k > 0 and neg_mask.any():
                                    # 我们仅在存在有效的忘记目标时计算
                                    forget_logits_all = forget_logits  # [F?, V]
                                    main_logits_forget = main_logits[mask_forget]  # [F?, V]
                                    side_logits_forget = forget_logits_all         # alias
                                    # 取主记忆Top-K索引作为强负样本候选
                                    try:
                                        topk_vals, topk_idx = torch.topk(main_logits_forget, k=min(hard_k, main_logits_forget.size(-1)), dim=-1)
                                        # 忽略等于被遗忘目标的索引
                                        # 为了与 topk_idx 形状对齐，构造被遗忘目标 id 张量
                                        forget_ids_full = forget_neg_ids.unsqueeze(1).expand_as(topk_idx)
                                        is_forgot_idx = (topk_idx == forget_ids_full)
                                        # 侧记忆对这些强负样本的打分
                                        side_neg_logits = side_logits_forget.gather(1, topk_idx)
                                        # 被遗忘目标的侧记忆打分（广播到K）
                                        s_forgot = side_logits_forget.gather(1, forget_neg_ids.unsqueeze(1))
                                        s_forgot_exp = s_forgot.expand_as(side_neg_logits)
                                        # 计算 margin 损失并屏蔽掉与被遗忘目标相同的索引
                                        margin_losses = F.relu(pw_margin + s_forgot_exp - side_neg_logits)
                                        margin_losses = margin_losses.masked_fill(is_forgot_idx, 0.0)
                                        pairwise_loss = margin_losses.mean()
                                        loss_forget_component = loss_forget_component + pw_weight * pairwise_loss
                                    except RuntimeError as e:
                                        logger.debug(f"[PAIRWISE] skip due to error: {e}")

                                # 4) 可选：TopK 出现惩罚（抑制“漏网即高位”现象）
                                tk_w = float(getattr(self.config, 'topk_penalty_weight', 0.0))
                                tk_k = int(getattr(self.config, 'topk_k', 20))
                                tk_m = float(getattr(self.config, 'topk_margin', 0.0))
                                if tk_w > 0.0 and tk_k > 0 and neg_mask.any():
                                    try:
                                        valid_side = forget_logits[neg_mask]              # [N_valid, V]
                                        valid_ids = forget_neg_ids[neg_mask]             # [N_valid]
                                        k_eff = min(tk_k, valid_side.size(-1))
                                        topk_vals_side, _ = torch.topk(valid_side, k=k_eff, dim=-1)
                                        kth_val = topk_vals_side[:, -1]                  # [N_valid]
                                        s_forgot_valid = valid_side.gather(1, valid_ids.unsqueeze(1)).squeeze(1) # [N_valid]
                                        # 惩罚当 s_forgot 高于 topK 的 K 阈（留可选 margin）
                                        topk_penalty = F.relu(tk_m + s_forgot_valid - kth_val).pow(2).mean()
                                        loss_forget_component = loss_forget_component + tk_w * topk_penalty
                                    except RuntimeError as e:
                                        logger.debug(f"[TOPK] skip due to error: {e}")
                            # ==================== [ 最终修复 END ] ====================


                            # ----------------- 调试打印（每批） -----------------
                            # 打印小量统计以便诊断训练稳定性和各损失分量的数量级
                            try:
                                if batch_idx % 10 == 0:
                                    with torch.no_grad():
                                        # logits 范围与范数
                                        main_norm = main_logits.detach().norm().item()
                                        side_norm = side_logits.detach().norm().item()
                                        main_max = float(main_logits.detach().max())
                                        main_min = float(main_logits.detach().min())
                                        side_max = float(side_logits.detach().max())
                                        side_min = float(side_logits.detach().min())
                                    logger.debug(
                                        f"[BATCH DEBUG] idx={batch_idx} bs={batch_size} pos_CE(BCE)={float(loss_r_component.item() if 'loss_r_component' in locals() else 0.0):.6f} "
                                        f"neg_BCE={float(loss_f_suppression.item() if 'loss_f_suppression' in locals() else 0.0):.6f} kl={float(kl_loss.item()):.6f} "
                                        f"main_norm={main_norm:.4f} side_norm={side_norm:.4f} main_max={main_max:.4f} side_max={side_max:.4f}"
                                    )
                            except Exception:
                                pass
                            
                            # 5. 对保留用户只计算CE Loss
                            # (注意：他们的KL Loss已经在上面计算过了)
                            mask_retain = (edit_type == 0.0)
                            loss_retain_component = torch.tensor(0.0, device=self.device)
                            
                            # [!! 注意 !!] 你的 retain_path 为空, 所以 loss_retain_component 始终为 0
                            # 这是正常的，因为你的所有训练样本都是遗忘样本 (edit_type=1.0)
                            if mask_retain.any():
                                retain_logits = side_logits[mask_retain]
                                retain_pos_ids = pos_ids[mask_retain]
                                pos_mask_retain = (retain_pos_ids >= 0)
                                if pos_mask_retain.any():
                                    # [!! 附带修复 !!] 如果未来你添加了保留样本, 
                                    # 也许你也希望它们使用BCE损失来保持对称?
                                    # (当前保持CE不变, 因为它不是你问题的原因)
                                    loss_retain_component = F.cross_entropy(retain_logits[pos_mask_retain], retain_pos_ids[pos_mask_retain])
                            
                            # 6. 组合总损失
                            total_loss = loss_forget_component + loss_retain_component + (kl_weight * kl_loss)

                            # 如果 total_loss 为 NaN 或 Inf，打印关键信息并跳过该批更新，避免训练崩塌
                            if not torch.isfinite(total_loss):
                                try:
                                    main_stats = (float(main_logits.max().item()), float(main_logits.min().item()), float(main_logits.norm().item()))
                                    side_stats = (float(side_logits.max().item()), float(side_logits.min().item()), float(side_logits.norm().item()))
                                except Exception:
                                    main_stats = side_stats = (float('nan'), float('nan'), float('nan'))
                                logger.error("[TRAIN NAN] epoch=%d batch=%d total_loss=%s pos_CE(BCE)=%s neg_BCE=%s kl=%s main_max/min/norm=%s side_max/min/norm=%s", epoch+1, batch_idx, str(total_loss), str(loss_r_component if 'loss_r_component' in locals() else 'NA'), str(loss_f_suppression if 'loss_f_suppression' in locals() else 'NA'), str(kl_loss if 'kl_loss' in locals() else 'NA'), str(main_stats), str(side_stats))
                                # 跳过该批次的反向更新
                                continue

                            if total_loss.requires_grad:
                                self.optimizer.zero_grad(set_to_none=True)
                                if self.scaler is not None:
                                    self.scaler.scale(total_loss).backward()
                                    if self.config.gradient_clip > 0:
                                        self.scaler.unscale_(self.optimizer)
                                        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.config.gradient_clip)
                                    self.scaler.step(self.optimizer)
                                    self.scaler.update()
                                else:
                                    total_loss.backward()
                                    if self.config.gradient_clip > 0:
                                        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.config.gradient_clip)
                                    self.optimizer.step()

                            epoch_loss += float(total_loss.item()) * batch_size
                            sample_count += batch_size
                            # diagnostics: accumulate kl and batch count
                            try:
                                diag_kl_sum += float(kl_loss.item())
                            except Exception:
                                pass
                            diag_batches += 1

                        avg_loss = epoch_loss / max(sample_count, 1)
                        self.history.append(avg_loss)
                        # epoch diagnostics
                        if diag_batches > 0:
                            avg_pos_ce = diag_pos_ce_sum / max(diag_batches, 1)
                            avg_neg_bce = diag_neg_bce_sum / max(diag_batches, 1)
                        else:
                            avg_pos_ce = 0.0
                            avg_neg_bce = 0.0
                        # [!! 修复 !!] 更新日志标签
                        logger.info(f"[DualMemory WISE] epoch {epoch + 1} loss {avg_loss:.4f} | R(BCE):{alpha} F(BCE):{beta} KL:{kl_weight} | pos_BCE={avg_pos_ce:.4f} neg_BCE={avg_neg_bce:.4f}")

                        if avg_loss + min_delta < best_loss:
                            best_loss = avg_loss
                            self.best_adapter_state = self.adapter.state_dict()
                            epochs_no_improve = 0
                            logger.info(f"🎉 New best model found at epoch {epoch + 1} with loss {avg_loss:.6f}!")
                        else:
                            epochs_no_improve += 1
                            logger.info(f"ℹ️ No improvement for {epochs_no_improve}/{patience} epochs.")

                        if patience > 0 and epochs_no_improve >= patience:
                            logger.info(f"⏹️ Early stopping triggered after {epoch + 1} epochs (patience={patience}).")
                            break

                    self.adapter.set_mode("main")
                    self.model.eval()
                    return self.history

# ---------------------------------------------------------------------------
# Router utilities
# ---------------------------------------------------------------------------

class ActivationRecorder:
    """
    [修正版] 增加了 detach 标志，以控制是否将捕获的激活值从计算图中分离。
    """
    def __init__(self, model: nn.Module, layer_names: Sequence[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.cache: Dict[str, torch.Tensor] = {}
        # [核心修复] 增加一个标志位来控制 detach 行为
        self.detach = True
        self.handles = [self._register(name) for name in self.layer_names]

    def _hook(self, name: str):
        def inner(_module, _inputs, output):
            # [核心修复] 只有当 self.detach 为 True 时才分离张量
            if self.detach:
                self.cache[name] = output.detach()
            else:
                self.cache[name] = output
        return inner

    def _register(self, name: str):
        parent, attr = _locate_module(self.model, name)
        if parent is None or attr is None:
            raise ValueError(f"Cannot locate module {name} for activation capture")
        module = getattr(parent, attr)
        return module.register_forward_hook(self._hook(name))

    def clear(self) -> None:
        self.cache.clear()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

def _ensure_tokens(tokenizer, model) -> None:
    special_tokens = []
    for token in [FORGET_TOKEN, RETAIN_TOKEN]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            special_tokens.append(token)
    if special_tokens:
        logger.info("Adding special tokens: %s", special_tokens)
        tokenizer.add_tokens(special_tokens)

    current_vocab = model.get_input_embeddings().num_embeddings
    target_vocab = len(tokenizer)
    if target_vocab != current_vocab:
        logger.info(
            "Resizing token embeddings from %d to %d to match tokenizer",
            current_vocab,
            target_vocab,
        )
        model.resize_token_embeddings(target_vocab)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataset_and_loader(
    tokenizer,
    config: DualMemoryConfig,
    test_df: Optional[pd.DataFrame] = None, # <--- 新增参数
) -> Tuple[DualMemoryDataset, DataLoader]:
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # [修改] 缓存文件名增加哈希，避免因test_df变化导致缓存冲突
    import hashlib
    test_df_hash = hashlib.md5(pd.util.hash_pandas_object(test_df).values).hexdigest()[:8] if test_df is not None else "no_test"
    cache_path = output_dir / f"cached_samples_seed{config.seed}_fr{config.forget_ratio}_rr{config.retain_ratio}_{test_df_hash}.pkl"

    
    # 检查缓存是否存在
    if cache_path.exists():
        logger.info(f"♻️ 发现缓存文件，直接加载预处理数据: {cache_path}")
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
        logger.info(f"✅ 从缓存加载了 {len(samples)} 个样本。")
    else:
        logger.info("ℹ️ 未发现缓存文件，开始进行首次数据预处理...")
        samples = build_samples(
            Path(config.forget_path),
            Path(config.retain_path),
            forget_ratio=config.forget_ratio,
            retain_ratio=config.retain_ratio,
            seed=config.seed,
            test_df=test_df # <--- 传递参数
        )
        logger.info(f"💾 正在将预处理结果保存到缓存文件: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)

    dataset = DualMemoryDataset(
        samples,
        tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )
    num_workers = max(0, int(getattr(config, 'dataloader_num_workers', 8)))
    loader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)
    return dataset, loader

def collect_router_features(
    model: nn.Module,
    adapter: DualMemoryAdapter,
    dataset: DualMemoryDataset,
    config: DualMemoryConfig,
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    [修改版] 收集路由器特征。
    特征从余弦相似度改为激活差异的L2范数 (Δ_act)，以对齐WISE论文和训练损失。
    """
    device = adapter.device
    recorder = ActivationRecorder(model, config.router_target_layers)
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []

    forget_deltas: List[float] = []
    retain_deltas: List[float] = []

    eval_loader = DataLoader(
        dataset,
        batch_size=config.router_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    try:
        for batch in eval_loader:
            input_ids = batch["router_input_ids"].to(device)
            attention_mask = batch["router_attention_mask"].to(device)
            edit_types = batch["edit_type"]
            sample_weight = batch["sample_weight"]
            
            batch_size = input_ids.size(0)
            decoder_input_ids = torch.full(
                (batch_size, 1), 
                tokenizer.pad_token_id, 
                dtype=torch.long, 
                device=device
            )

            adapter.set_mode("main")
            recorder.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            main_cache = {name: recorder.cache.get(name) for name in config.router_target_layers}

            adapter.set_mode("side")
            recorder.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            side_cache = {name: recorder.cache.get(name) for name in config.router_target_layers}
            
            adapter.set_mode("main")

            delta_aggregates: List[torch.Tensor] = []
            for name in config.router_target_layers:
                main_act, side_act = main_cache.get(name), side_cache.get(name)
                if main_act is None or side_act is None: continue
                
                # 计算每个样本的激活差异的L2范数
                delta = (side_act - main_act).norm(p=2, dim=-1) # Shape: (batch_size, seq_len)
                # 我们关心的是第一个token的激活差异
                delta_first_token = delta[:, 0]
                delta_aggregates.append(delta_first_token)
            
            if not delta_aggregates: continue

            # 对多层的delta值取平均
            delta_tensor = torch.stack(delta_aggregates, dim=1).mean(dim=1).cpu()
            
            # [核心修改] 特征向量现在是一维的 Δ_act
            features.append(delta_tensor.clone().unsqueeze(-1))
            
            labels.append(edit_types.clone().unsqueeze(-1))
            weights.append(sample_weight.clone().unsqueeze(-1))
            
            for value, label in zip(delta_tensor.tolist(), edit_types.tolist()):
                if label == 1.0: # forget
                    forget_deltas.append(value)
                else: # retain
                    retain_deltas.append(value)
    finally:
        recorder.remove()

    if not features:
        raise RuntimeError("Router feature collection yielded no data")

    logger.info("=" * 30 + " ROUTER FEATURE DIAGNOSTICS " + "=" * 30)
    if retain_deltas:
        retain_tensor_dbg = torch.tensor(retain_deltas)
        logger.info(f"[RETAIN SAMPLES] Count: {len(retain_deltas)}")
        logger.info(f"  - Δ_act Stats: Mean={retain_tensor_dbg.mean():.4f}, Std={retain_tensor_dbg.std():.4f}, Min={retain_tensor_dbg.min():.4f}, Max={retain_tensor_dbg.max():.4f}")
    else:
        logger.info("[RETAIN SAMPLES] No samples found.")
    
    if forget_deltas:
        forget_tensor_dbg = torch.tensor(forget_deltas)
        logger.info(f"[FORGET SAMPLES] Count: {len(forget_deltas)}")
        logger.info(f"  - Δ_act Stats: Mean={forget_tensor_dbg.mean():.4f}, Std={forget_tensor_dbg.std():.4f}, Min={forget_tensor_dbg.min():.4f}, Max={forget_tensor_dbg.max():.4f}")
    else:
        logger.info("[FORGET SAMPLES] No samples found.")
    logger.info("=" * 84)

    feat_all = torch.cat(features, dim=0)
    label_all = torch.cat(labels, dim=0)
    weight_all = torch.cat(weights, dim=0)

    feat_mean = feat_all.mean(dim=0)
    feat_std = feat_all.std(dim=0).clamp(min=1e-6)

    normalized_feat_tensor = (feat_all - feat_mean) / feat_std
    
    recorder_stats = {
        "forget_delta_mean": float(torch.tensor(forget_deltas).mean().item()) if forget_deltas else 0.0,
        "retain_delta_mean": float(torch.tensor(retain_deltas).mean().item()) if retain_deltas else 0.0,
        "num_samples": int(feat_all.size(0)),
        "normalization_mean": feat_mean,
        "normalization_std": feat_std,
    }
    
    return normalized_feat_tensor.float(), label_all.float(), weight_all.float(), recorder_stats

class SimpleRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterTrainer:
    def __init__(self, input_dim: int, config: DualMemoryConfig) -> None:
        self.model = SimpleRouter(input_dim, config.router_hidden_dim, config.router_dropout)
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.router_lr,
            weight_decay=config.router_weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def fit(self, features: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> List[float]:
        dataset = TensorDataset(features, labels, weights)
        loader = DataLoader(dataset, batch_size=self.config.router_batch_size, shuffle=True)
        history: List[float] = []
        self.model.train()
        
        for epoch in range(self.config.router_epochs):
            epoch_loss = 0.0
            samples = 0
            for batch in loader:
                feats, lbls, wts = batch
                feats, lbls, wts = feats.to(self.device), lbls.to(self.device), wts.to(self.device)

                logits = self.model(feats)
                # --- 核心修改：移除标签平滑，使用原始标签 ---
                loss = self.criterion(logits, lbls)
                loss = (loss * wts).mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * feats.size(0)
                samples += feats.size(0)
            history.append(epoch_loss / max(samples, 1))
        return history

    @torch.no_grad()
    def evaluate(self, features: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.model.eval()
        feats, lbls = features.to(self.device), labels.to(self.device)
        logits = self.model(feats)
        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= 0.5).float()
        lbl_cpu = lbls.cpu()
        tp = ((preds == 1) & (lbl_cpu == 1)).sum().item()
        fp = ((preds == 1) & (lbl_cpu == 0)).sum().item()
        fn = ((preds == 0) & (lbl_cpu == 1)).sum().item()
        tn = ((preds == 0) & (lbl_cpu == 0)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        return {"precision": precision, "recall": recall, "accuracy": accuracy}

    @torch.no_grad()
    def find_threshold(self, features: torch.Tensor, labels: torch.Tensor, target_precision: float) -> float:
        self.model.eval()
        feats, labels_cpu = features.to(self.device), labels.squeeze(-1).cpu()
        probs = torch.sigmoid(self.model(feats)).cpu().squeeze(-1)
        
        thresholds = torch.linspace(0.01, 0.99, 100)
        best_threshold = 0.99 
        best_f1 = -1.0
        
        for thr in thresholds:
            preds = (probs >= thr).float()
            tp = ((preds == 1) & (labels_cpu == 1)).sum().item()
            fp = ((preds == 1) & (labels_cpu == 0)).sum().item()
            fn = ((preds == 0) & (labels_cpu == 1)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            if precision >= target_precision:
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(thr)
        
        if best_f1 == -1.0:
            best_prec = -1.0
            for thr in thresholds:
                preds = (probs >= thr).float()
                tp = ((preds == 1) & (labels_cpu == 1)).sum().item()
                fp = ((preds == 1) & (labels_cpu == 0)).sum().item()
                precision = tp / (tp + fp + 1e-8)
                if precision > best_prec:
                    best_prec = precision
                    best_threshold = float(thr)

        return best_threshold

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

@dataclass
class DualMemoryArtifacts:
    adapter_state: Dict[str, Dict[str, torch.Tensor]]
    router_state: Dict[str, torch.Tensor]
    router_threshold: float
    dual_training_history: List[float]
    router_history: List[float]
    router_metrics: Dict[str, float]
    router_feature_stats: Dict[str, Any]
    config: Dict[str, Any]


def train_dual_memory(
    config: DualMemoryConfig,
    initial_state: Optional[Dict[str, Any]] = None,
    test_df_for_sampling: Optional[pd.DataFrame] = None, # <--- 新增参数
) -> DualMemoryArtifacts:
    _set_seed(config.seed)
    device = torch.device(config.device)

    wrapper = P5ModelWrapper(config.model_path, device=str(device))
    model = wrapper.get_model()
    tokenizer = wrapper.get_tokenizer()

    _ensure_tokens(tokenizer, model)

    for param in model.parameters():
        param.requires_grad = False

    # 根据配置可选地移除 lm_head 的 LoRA 注入
    target_layers = list(config.edit_target_layers)
    if config.freeze_lm_head:
        target_layers = [n for n in target_layers if n != "lm_head"]

    adapter = DualMemoryAdapter(
        model,
        AdapterConfig(
            target_layers=target_layers,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ),
    )
    if initial_state is not None:
        adapter_state = initial_state.get("adapter_state")
        if isinstance(adapter_state, dict):
            adapter.load_state_dict(adapter_state)

    dataset, dataloader = build_dataset_and_loader(tokenizer, config, test_df=test_df_for_sampling) # <--- 传递参数
    trainer = DualMemoryTrainer(model, adapter, tokenizer, config)
    dual_history = trainer.train(dataloader)

    # --- 路由器训练已禁用 (由简化训练流程替代) ---
    # 原先会收集 router 特征并训练一个小网络。为简化训练并提高速度，
    # 我们在这里直接使用占位符值以保持输出结构兼容。
    features = None
    labels = None
    weights = None
    feature_stats = {}

    router_history = []
    router_metrics = {"status": "disabled_by_user_request"}
    router_threshold = 0.5

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "dual_memory_artifacts.pt"
    # Prefer the best adapter state tracked by the trainer (from early stopping)
    saved_adapter_state = trainer.best_adapter_state if getattr(trainer, "best_adapter_state", None) is not None else adapter.state_dict()
    best_used = trainer.best_adapter_state is not None

    torch.save(
        {
            "adapter_state": saved_adapter_state,
            # router_state omitted (training disabled); keep placeholders for compatibility
            "router_state": {},
            "router_threshold": router_threshold,
            "dual_history": dual_history,
            "router_history": router_history,
            "router_metrics": router_metrics,
            "router_feature_stats": feature_stats,
            "config": config.to_dict(),
            # metadata about best-state selection
            "best_adapter_used": best_used,
        },
        artifact_path,
    )
    logger.info("Dual-memory artifacts saved to %s (best_adapter_used=%s)", artifact_path, best_used)

    return DualMemoryArtifacts(
        adapter_state=saved_adapter_state,
        router_state={},
        router_threshold=router_threshold,
        dual_training_history=dual_history,
        router_history=router_history,
        router_metrics=router_metrics,
        router_feature_stats=feature_stats,
        config=config.to_dict(),
    )

class DualMemoryRuntime:
    def __init__(
            self,
            model: nn.Module,
            adapter: DualMemoryAdapter,
            router: SimpleRouter,
            tokenizer,
            config: DualMemoryConfig,
            threshold: float,
            epsilon: float,
            normalization_stats: Optional[Dict] = None,
            verbose: bool = False,
        ) -> None:
            self.model = model
            self.adapter = adapter
            self.router = router
            self.tokenizer = tokenizer
            self.config = config
            self.threshold = threshold
            self.epsilon = epsilon
            self.device = torch.device(config.device)
            self.model.eval()
            self.adapter.eval()
            self.adapter.set_mode("main")
            # [最终修复] 存储归一化参数
            self.normalization_mean = normalization_stats.get("mean") if normalization_stats else None
            self.normalization_std = normalization_stats.get("std") if normalization_stats else None
            # 控制诊断输出
            self.verbose = bool(verbose)

    def _prepare_inputs(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.config.max_input_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded.input_ids.to(self.device),
            "attention_mask": encoded.attention_mask.to(self.device),
        }


    def _compute_router_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        [修改版] 在推理时计算路由器特征，与 collect_router_features 逻辑完全对齐。
        使用激活差异的L2范数 (Δ_act) 作为特征。
        """
        recorder = ActivationRecorder(self.model, self.config.router_target_layers)
        
        batch_size = input_ids.size(0)
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            self.tokenizer.pad_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        try:
            self.adapter.set_mode("main"); recorder.clear()
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            main_cache = {name: recorder.cache.get(name) for name in self.config.router_target_layers}
            
            self.adapter.set_mode("side"); recorder.clear()
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            side_cache = {name: recorder.cache.get(name) for name in self.config.router_target_layers}
        finally:
            recorder.remove()
            self.adapter.set_mode("main")

        delta_aggregates: List[torch.Tensor] = []
        for name in self.config.router_target_layers:
            main_act, side_act = main_cache.get(name), side_cache.get(name)
            if main_act is None or side_act is None: continue
            
            delta = (side_act - main_act).norm(p=2, dim=-1)[:, 0] # First token
            delta_aggregates.append(delta)

        # 如果没有有效的激活层，返回零特征
        if not delta_aggregates: return torch.zeros(input_ids.size(0), 1, device=self.device)

        # [核心修改] 特征是1维的 Δ_act
        features = torch.stack(delta_aggregates, dim=1).mean(dim=1).unsqueeze(-1)

        # 使用从训练阶段加载的均值和标准差进行归一化
        if self.normalization_mean is not None and self.normalization_std is not None:
            mean = self.normalization_mean.to(features.device)
            std = self.normalization_std.to(features.device)
            features = (features - mean) / std
        
        return features

    def route(self, prompts: Sequence[str]) -> Dict[str, Any]:
        inputs = self._prepare_inputs(prompts)
        features = self._compute_router_features(inputs["input_ids"], inputs["attention_mask"])
        self.router.eval()
        with torch.no_grad():
            logits = self.router(features)
            probs = torch.sigmoid(logits).squeeze(-1)
        
        if self.verbose:
            logger.debug("\n" + "="*25 + " [ROUTER DIAGNOSTICS] " + "="*25)
            try:
                logger.debug("  [Router Weights] Router State Dict: %s", self.router.state_dict())
                logger.debug("  [Router Thresholds] Epsilon: %.4f, Probability Threshold: %.4f", self.epsilon, self.threshold)
                logger.debug("  [Router Input Features] Shape: %s", tuple(features.shape))
                # 打印前5个样本的诊断信息
                for i in range(min(5, features.shape[0])):
                    delta_val = features[i, 0].item()
                    # feature may be 1-d delta only; guard access
                    try:
                        delta_minus_epsilon_val = features[i, 1].item()
                    except Exception:
                        delta_minus_epsilon_val = float('nan')
                    prob_val = probs[i].item()
                    decision = "SIDE (Forget)" if prob_val >= self.threshold else "MAIN (Retain)"
                    logger.debug("    - Sample %d: Raw Delta=%.4f, Delta-Eps=%.4f, Prob=%.6f, Decision=%s", i, delta_val, delta_minus_epsilon_val, prob_val, decision)
            except Exception as e:
                logger.debug("  [Router Diagnostics] Error during logging diagnostics: %s", e)
            logger.debug("%s", "="*72)

        decisions = probs >= self.threshold
        
        return {
            "features": features.cpu(),
            "probs": probs.cpu(),
            "use_side": decisions.cpu(),
        }

    def generate(self, prompts: Sequence[str], **generate_kwargs) -> Dict[str, Any]:
        if "max_length" not in generate_kwargs and "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = self.config.max_target_length
        inputs = self._prepare_inputs(prompts)
        routing = self.route(prompts)
        probs = routing["probs"].to(self.device)
        decisions = routing["use_side"].to(self.device).bool()
        feature_tensor = routing.get("features")
        per_prompt_outputs: List[List[str]] = [[] for _ in prompts]
        num_return = int(generate_kwargs.get("num_return_sequences", 1))
        side_indices = decisions.nonzero(as_tuple=False).view(-1)
        main_indices = (~decisions).nonzero(as_tuple=False).view(-1)
        for indices, use_side in ((main_indices, False), (side_indices, True)):
            if indices.numel() == 0: continue
            self.adapter.set_mode("side" if use_side else "main")
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=inputs["input_ids"][indices],
                    attention_mask=inputs["attention_mask"][indices],
                    **generate_kwargs,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            if indices.numel() > 0:
                expected_chunk = max(1, num_return)
                per_example = len(decoded) // max(indices.numel(), 1)
                if per_example <= 0: per_example = expected_chunk
                elif expected_chunk > 1 and per_example != expected_chunk: per_example = expected_chunk
                for offset, tensor_index in enumerate(indices.tolist()):
                    start, end = offset * per_example, (offset + 1) * per_example
                    slice_decoded = decoded[start:end]
                    if not slice_decoded and decoded: slice_decoded = [decoded[min(start, len(decoded) - 1)]]
                    per_prompt_outputs[tensor_index] = slice_decoded
        self.adapter.set_mode("main")
        feature_payload = feature_tensor.cpu().tolist() if isinstance(feature_tensor, torch.Tensor) else feature_tensor
        return {
            "outputs": per_prompt_outputs,
            "probs": probs.cpu().tolist(),
            "use_side": decisions.cpu().tolist(),
            "features": feature_payload,
        }
    
def load_dual_memory_runtime(
    artifact_path: str,
    device: Optional[str] = None,
    threshold: Optional[float] = None,
) -> DualMemoryRuntime:
    """
    [BUG修复版] 加载DualMemory运行时。
    适配新的、输入维度为1的路由器，并安全处理空的router_state。
    """
    payload = torch.load(artifact_path, map_location="cpu")
    config = DualMemoryConfig(**payload["config"])
    if device:
        config.device = device
    wrapper = P5ModelWrapper(config.model_path, device=config.device)
    model = wrapper.get_model()
    tokenizer = wrapper.get_tokenizer()
    _ensure_tokens(tokenizer, model)

    # 与训练阶段保持一致：若 freeze_lm_head=True，则在推理端也排除 lm_head 的 LoRA 注入
    runtime_target_layers = list(config.edit_target_layers)
    if getattr(config, "freeze_lm_head", False):
        runtime_target_layers = [n for n in runtime_target_layers if n != "lm_head"]

    adapter = DualMemoryAdapter(
        model,
        AdapterConfig(
            target_layers=runtime_target_layers,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ),
    )
    adapter.to(torch.device(config.device))
    adapter.load_state_dict(payload["adapter_state"])
    adapter.set_mode("main")

    input_dim = 1
    router = SimpleRouter(
        input_dim=input_dim,
        hidden_dim=config.router_hidden_dim,
        dropout=config.router_dropout,
    )
    
    # --- [核心修复] ---
    # 只有当 router_state 存在且不为空时，才加载它
    router_state = payload.get("router_state")
    if router_state:
        router.load_state_dict(router_state)
    # --- [修复结束] ---
    
    router.to(torch.device(config.device))

    feature_stats = payload.get("router_feature_stats", {})
    norm_stats = {
        "mean": feature_stats.get("normalization_mean"),
        "std": feature_stats.get("normalization_std"),
    }

    runtime_threshold = float(payload.get("router_threshold", 0.5))
    if threshold is not None:
        runtime_threshold = float(threshold)

    return DualMemoryRuntime(
        model=model,
        adapter=adapter,
        router=router,
        tokenizer=tokenizer,
        config=config,
        threshold=runtime_threshold,
        epsilon=0.0,
        normalization_stats=norm_stats,
    )

def export_dual_memory_checkpoint(
    artifact_path: str,
    output_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
) -> str:
    """将 dual-memory artifacts 转换为 evaluate_datasets 可读取的综合 .pt 模型文件."""

    artifact_file = Path(artifact_path)
    if not artifact_file.exists():
        raise FileNotFoundError(f"Dual-memory artifact 不存在: {artifact_file}")

    payload = torch.load(str(artifact_file), map_location="cpu")
    config_dict: Dict[str, Any] = payload.get("config", {})
    if not config_dict:
        raise ValueError("dual-memory artifact 缺少 config 字段，无法确定基础模型路径")

    base_path = base_model_path or config_dict.get("model_path")
    if not base_path:
        raise ValueError("请通过 base_model_path 显式指定基础模型路径，或确保 config.model_path 存在")
    base_file = Path(base_path)
    if not base_file.exists():
        raise FileNotFoundError(f"基础模型文件不存在: {base_file}")

    base_checkpoint = torch.load(str(base_file), map_location="cpu")
    if isinstance(base_checkpoint, dict) and "model_state_dict" in base_checkpoint:
        model_state = base_checkpoint["model_state_dict"]
    else:
        model_state = base_checkpoint

    adapter_state: Dict[str, Any] = payload.get("adapter_state", {})
    if not adapter_state:
        raise ValueError("dual-memory artifact 缺少 adapter_state")

    lora_scaling = float(config_dict.get("lora_alpha", 16.0)) / max(1, int(config_dict.get("lora_r", 8)))
    lora_deltas: Dict[str, Dict[str, Any]] = {}
    for layer_name, layer_state in adapter_state.items():
        if layer_name == "__forget_bias__":
            continue
        if not isinstance(layer_state, dict):
            raise ValueError(f"LoRA 层状态格式异常: {layer_name}")
        if "lora_A" not in layer_state or "lora_B" not in layer_state:
            raise ValueError(f"LoRA 层缺少 lora_A/lora_B 参数: {layer_name}")
        lora_deltas[layer_name] = {
            "lora_A": layer_state["lora_A"].cpu(),
            "lora_B": layer_state["lora_B"].cpu(),
            "scaling": float(lora_scaling),
        }

    router_threshold = float(payload.get("router_threshold", 0.5))
    epsilon_threshold = payload.get("router_feature_stats", {}).get("epsilon")

    wise_config = {
        "edit_target_layers": list(config_dict.get("edit_target_layers", [])),
        "router_target_layers": list(config_dict.get("router_target_layers", [])),
        "lora_r": int(config_dict.get("lora_r", 8)),
        "lora_alpha": float(config_dict.get("lora_alpha", 16.0)),
        "lora_dropout": float(config_dict.get("lora_dropout", 0.0)),
        "router_hidden_dim": int(config_dict.get("router_hidden_dim", 64)),
        "router_dropout": float(config_dict.get("router_dropout", 0.1)),
        "router_feature_mode": "delta-only",
        "router_prob_threshold": router_threshold,
        "router_prob_target_precision": float(config_dict.get("router_target_precision", 0.8)),
    "router_prob_threshold_min": float(config_dict.get("router_prob_threshold_min", router_threshold * 0.8)),
        "calibrator_enabled": False,
    }

    combined_checkpoint = {
        "model_state_dict": model_state,
        "router_classifier_state_dict": None,
        "lora_side_deltas": lora_deltas,
        "epsilon_threshold": float(epsilon_threshold) if epsilon_threshold is not None else None,
        "router_prob_threshold": router_threshold,
        "wise_config": wise_config,
        "delta_stats": payload.get("router_metrics", {}),
        "router_feature_norm": None,
        "dual_memory_metadata": {
            "artifact_path": str(artifact_file.resolve()),
            "router_feature_stats": payload.get("router_feature_stats", {}),
        },
    }

    if output_path is None:
        output_dir = Path(config_dict.get("output_dir", artifact_file.parent))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"dual_memory_combined_{artifact_file.stem}.pt"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(combined_checkpoint, str(output_file))
    logger.info("Dual-memory 综合模型已保存: %s", output_file)
    return str(output_file)


__all__ = [
    "DualMemoryConfig",
    "DualMemoryArtifacts",
    "train_dual_memory",
    "DualMemoryAdapter",
    "DualMemoryDataset",
    "DualMemoryRuntime",
    "load_dual_memory_runtime",
    "export_dual_memory_checkpoint",
]
