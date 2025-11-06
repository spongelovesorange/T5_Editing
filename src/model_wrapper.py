# File: model_wrapper.py (Corrected Version)

#!/usr/bin/env python3
"""
P5推荐模型包装器 - 最终修复版
"""

import os
import torch
import torch.nn as nn
import logging
import re
from typing import List, Dict, Any, Optional

# 延迟导入transformers，便于在缺失时给出清晰提示
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
except ImportError as e:
    T5ForConditionalGeneration = T5Tokenizer = T5Config = None
    _transformers_import_error = e
else:
    _transformers_import_error = None

class P5ModelWrapper:
    """P5推荐模型的包装器类（最终修复版，支持智能加载）"""
    
    def __init__(self, model_path: str, device: str = 'cuda', t5_local_dir: Optional[str] = None, checkpoint: Optional[Dict] = None):
        """
        初始化P5模型包装器
        Args:
            model_path: P5模型标识符或权重文件路径（主要供日志记录）
            device: 设备类型 ('cuda' 或 'cpu')
            t5_local_dir: 本地t5模型目录
            checkpoint: (新增) 如果提供，则直接从此checkpoint加载，而不是从model_path加载文件
        """
        self.model_path = model_path
        self.device = device
        self.checkpoint = checkpoint
        self.logger = logging.getLogger(__name__)

        self.logger.info("🔍 模型初始化: 检查依赖库 transformers / sentencepiece ...")
        if _transformers_import_error is not None:
            raise _transformers_import_error
        if t5_local_dir and os.path.isdir(t5_local_dir):
            self.t5_source = t5_local_dir
        else:
            # 2. 检查环境变量
            env_path = os.environ.get('P5_T5_MODEL_DIR')
            if env_path and os.path.isdir(env_path):
                self.t5_source = env_path
            else:
                # 3. 自动计算项目内的默认路径
                try:
                    # PROJECT_ROOT 应该是 train_dual_memory.py 所在的目录
                    # train_dual_memory.py -> src/ -> model_wrapper.py
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    default_local_path = os.path.join(project_root, "hf_models", "t5-small")
                except NameError:
                    # 如果 __file__ 不可用，则从当前工作目录猜测
                    default_local_path = os.path.join(os.getcwd(), "hf_models", "t5-small")

                if os.path.isdir(default_local_path):
                    self.t5_source = default_local_path
                else:
                    # 4. 如果以上全部失败，才回退到在线下载（在您的情况下会报错）
                    self.t5_source = 't5-small'
                    self.logger.warning(f"⚠️ 未找到任何本地T5模型路径，将尝试在线下载 '{self.t5_source}'")

        self.logger.info("📦 加载T5分词器: %s", self.t5_source)
        self.tokenizer = T5Tokenizer.from_pretrained(self.t5_source)
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """智能加载P5模型，兼容新旧两种checkpoint格式，并支持Dual-memory artifact的恢复

        逻辑来源: evaluate_datasets.initialize_model（已迁移并适配到 wrapper）。
        """
        if self.checkpoint is None and os.path.isdir(self.model_path):
                    try:
                        self.logger.info(f"🎯 检测到HuggingFace模型目录，尝试从目录加载: {self.model_path}")
                        self.logger.info(f"🔁 重新加载 Tokenizer 以确保匹配: {self.model_path}")
                        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
                        # 使用 from_pretrained 加载模型，分词器已在 __init__ 中加载
                        model = T5ForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
                        
                        # 检查并同步词表大小 (与 evaluate_datasets.py 中的逻辑一致)
                        tok_size = len(self.tokenizer)
                        emb_size = model.get_input_embeddings().weight.size(0)

                        # [ 关键 ] 移除那个错误的 resize 逻辑 (HF from_pretrained 应该已经处理了)
                        # 我们只在 tok_size > emb_size 时才需要介入
                        if tok_size > emb_size:
                            self.logger.warning(f"⚠️ Tokenizer 词表大小 ({tok_size}) 大于 模型 ({emb_size}). 正在调整模型大小...")
                            model.resize_token_embeddings(tok_size)
                        elif emb_size > tok_size:
                            self.logger.warning(f"⚠️ 模型词表大小 ({emb_size}) 大于 Tokenizer ({tok_size}). 这可能是一个配置错误，但我们将继续...")

                        self.model = model
                        self.model.eval()
                        self.logger.info(f"✅ HuggingFace 模型加载完成, device={self.device}")
                        
                        # [ 关键 ] 加载成功后必须立即返回，跳过后续的 torch.load 逻辑
                        return 
                    except Exception as e:
                        self.logger.warning(f"⚠️ 无法将目录作为HuggingFace模型加载, 将回退到 checkpoint 加载... Error: {e}")

        if self.checkpoint is None:
            self.logger.info(f"🎯 从文件加载P5模型权重: {self.model_path}")
            if not os.path.exists(self.model_path):
                self.logger.warning(f"⚠️ 模型权重文件 {self.model_path} 不存在，将仅使用预训练 {self.t5_source} 基础模型")
                self.model = T5ForConditionalGeneration.from_pretrained(self.t5_source).to(self.device)
                self.model.eval()
                return
            # PyTorch 2.6 默认 weights_only=True 会导致老式checkpoint报 _pickle.UnpicklingError
            # 这里显式设置为 False（前提是来自可信来源的本地文件）
            self.checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        else:
            self.logger.info(f"🎯 使用传入 checkpoint 加载模型: {self.model_path}")

        checkpoint = self.checkpoint

        # small helpers copied/adapted from evaluate_datasets
        def _ensure_special_tokens(tokenizer_obj):
            if tokenizer_obj is None or not hasattr(tokenizer_obj, 'add_special_tokens'):
                return 0
            forget_id = tokenizer_obj.convert_tokens_to_ids('<forget>')
            retain_id = tokenizer_obj.convert_tokens_to_ids('<retain>')
            unk_id = getattr(tokenizer_obj, 'unk_token_id', None)
            tokens_to_add = []
            if forget_id is None or (unk_id is not None and forget_id == unk_id):
                tokens_to_add.append('<forget>')
            if retain_id is None or (unk_id is not None and retain_id == unk_id):
                tokens_to_add.append('<retain>')
            if not tokens_to_add:
                return 0
            special_tokens_dict = {'additional_special_tokens': tokens_to_add}
            try:
                added = tokenizer_obj.add_special_tokens(special_tokens_dict)
                if added > 0:
                    self.logger.info("🔁 补充特殊Token: 新增 %d 个 (%s)", added, ','.join(tokens_to_add))
                return int(added)
            except Exception as tok_err:
                self.logger.warning("添加特殊Token失败: %s", tok_err)
                return 0

        def _sync_vocab_size(model_obj, tokenizer_obj, min_expand: int = 0):
            if tokenizer_obj is None or model_obj is None:
                return 0
            desired_size = len(tokenizer_obj)
            if desired_size <= 0:
                return 0
            current_size = model_obj.get_input_embeddings().weight.size(0)
            if desired_size <= current_size and min_expand <= 0:
                return 0
            target_size = max(desired_size, current_size + max(min_expand, 0))
            if target_size == current_size:
                return 0
            model_obj.resize_token_embeddings(target_size)
            with torch.no_grad():
                embeddings = model_obj.get_input_embeddings().weight
                ref_slice = embeddings[:current_size] if current_size > 0 else None
                if ref_slice is not None and ref_slice.numel() > 0 and target_size > current_size:
                    mu = ref_slice.mean(dim=0)
                    sigma = ref_slice.std(dim=0)
                    sigma = sigma.clamp(min=1e-6)
                    embeddings[current_size:target_size] = mu + torch.randn_like(embeddings[current_size:target_size]) * sigma * 0.01
            self.logger.info("🔁 词表同步: 模型vocab=%d -> %d", current_size, target_size)
            return target_size - current_size

        def _align_lora_output_dims(editor: Optional[Any]) -> None:
            if editor is None:
                return
            for layer_name, module in editor.side_modules.items():
                if not hasattr(module, 'lora_B') or module.lora_B is None:
                    continue
                target_out = module.weight.shape[0]
                current_out = module.lora_B.shape[0]
                if current_out == target_out:
                    continue
                old_param = module.lora_B.data
                new_param = old_param.new_zeros((target_out, old_param.shape[1]))
                rows_copy = min(target_out, current_out)
                if rows_copy > 0:
                    new_param[:rows_copy] = old_param[:rows_copy]
                module.lora_B = nn.Parameter(new_param)
                self.logger.info("LoRA层输出维度调整: %s (%d -> %d)", layer_name, current_out, target_out)

        # Detect format and infer vocab size
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict_for_size_check = checkpoint['model_state_dict']
        else:
            state_dict_for_size_check = checkpoint

        vocab_size = None
        if isinstance(state_dict_for_size_check, dict) and 'shared.weight' in state_dict_for_size_check:
            try:
                vocab_size = int(state_dict_for_size_check['shared.weight'].shape[0])
                self.logger.info(f"从checkpoint推断到 vocab_size={vocab_size}")
            except Exception:
                vocab_size = None

        if vocab_size is None:
            vocab_size = None

        # Build base model with correct vocab if possible
        try:
            if vocab_size is not None:
                cfg = T5Config.from_pretrained(self.t5_source, vocab_size=vocab_size)
                base_model = T5ForConditionalGeneration(cfg)
            else:
                base_model = T5ForConditionalGeneration.from_pretrained(self.t5_source)
        except Exception:
            self.logger.warning("无法使用本地/远程T5配置，尝试直接使用默认T5-small实例化")
            base_model = T5ForConditionalGeneration.from_pretrained(self.t5_source)

        base_model = base_model.to(self.device)

        # If this is a dual-memory artifact (contains router_classifier_state_dict), perform WISE editor wrapping and LoRA application
        if isinstance(checkpoint, dict) and 'router_classifier_state_dict' in checkpoint:
            self.logger.info("检测到 Dual-memory artifact，恢复主记忆与侧记忆增量...")
            wise_config = checkpoint.get('wise_config', {})

            # Ensure tokenizer has special tokens
            added = _ensure_special_tokens(self.tokenizer)

            # Create WISE editor from dual_memory module if available
            try:
                from src.dual_memory import WISEUnlearningEditor as RealWISEEditor
                wise_editor = RealWISEEditor(base_model, self.tokenizer, wise_config)
            except Exception:
                self.logger.info("WISEUnlearningEditor 未在 src.dual_memory 中找到，使用本地占位实现")
                wise_editor = None

            # Load main weights
            try:
                base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                self.logger.info("✅ 主记忆权重加载成功")
            except Exception as e:
                self.logger.warning(f"加载主记忆权重时出现问题: {e}")

            try:
                _sync_vocab_size(base_model, self.tokenizer, min_expand=added)
                _align_lora_output_dims(wise_editor)
            except Exception as exc:
                self.logger.warning(f"词表/LoRA 调整失败: {exc}")

            # Load router classifier if present
            router_sd = checkpoint.get('router_classifier_state_dict', None)
            if router_sd and wise_editor is not None and getattr(wise_editor, 'router_classifier', None) is not None:
                try:
                    wise_editor.router_classifier.load_state_dict(router_sd)
                    wise_editor.router_classifier.eval()
                    self.logger.info("✅ 路由分类器加载成功")
                except Exception as _e:
                    self.logger.warning(f"路由分类器权重加载失败，将回退到 Δact-only 路由: {_e}")

            # Apply side deltas (LoRA or old-style)
            applied_side = False
            if 'lora_side_deltas' in checkpoint:
                lora_deltas = checkpoint['lora_side_deltas'] or {}
                loaded = 0

                def _copy_with_resize(target: torch.Tensor, source: torch.Tensor, tensor_name: str) -> None:
                    src_tensor = source.to(target.device)
                    if target.shape == src_tensor.shape:
                        target.copy_(src_tensor)
                        return
                    if target.dim() != src_tensor.dim():
                        raise RuntimeError(f"LoRA tensor维度不匹配[{tensor_name}]: target={target.shape}, source={src_tensor.shape}")
                    min_shape = tuple(min(t, s) for t, s in zip(target.shape, src_tensor.shape))
                    slices = tuple(slice(0, ms) for ms in min_shape)
                    target[slices] = src_tensor[slices]

                if wise_editor is not None:
                    for name, comp in lora_deltas.items():
                        if name in wise_editor.side_modules:
                            module = wise_editor.side_modules[name]
                            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                                with torch.no_grad():
                                    if 'lora_A' in comp:
                                        _copy_with_resize(module.lora_A, comp['lora_A'], f"{name}.lora_A")
                                    if 'lora_B' in comp:
                                        _copy_with_resize(module.lora_B, comp['lora_B'], f"{name}.lora_B")
                                    if 'scaling' in comp:
                                        module.scaling = comp['scaling']
                                loaded += 1
                    self.logger.info(f"✅ LoRA 侧记忆增量加载完成: {loaded}/{len(lora_deltas)}")
                    applied_side = True

            elif 'side_weight_deltas' in checkpoint and wise_editor is not None:
                for name, delta in checkpoint['side_weight_deltas'].items():
                    if name in wise_editor.side_modules:
                        module = wise_editor.side_modules[name]
                        if hasattr(module, 'side_weight'):
                            with torch.no_grad():
                                module.side_weight.add_(delta.to(module.side_weight.device))
                self.logger.info("✅ 侧记忆(旧格式)应用成功")
                applied_side = True

            # Restore thresholds and router feature norms if present
            if 'epsilon_threshold' in checkpoint and wise_editor is not None:
                wise_editor.epsilon_threshold = checkpoint.get('epsilon_threshold', wise_editor.epsilon_threshold)
            if 'router_prob_threshold' in checkpoint and checkpoint['router_prob_threshold'] is not None and wise_editor is not None:
                try:
                    wise_editor.router_prob_threshold = float(checkpoint['router_prob_threshold'])
                except Exception:
                    pass

            rfn = checkpoint.get('router_feature_norm', None)
            if rfn and wise_editor is not None:
                try:
                    mean = rfn.get('mean', None)
                    std = rfn.get('std', None)
                    eps_ref = rfn.get('epsilon_ref', None)
                    if mean is not None and std is not None:
                        mean_t = mean.to(self.device) if hasattr(mean, 'to') else torch.tensor(mean, device=self.device, dtype=torch.float32)
                        std_t = std.to(self.device) if hasattr(std, 'to') else torch.tensor(std, device=self.device, dtype=torch.float32)
                        wise_editor.router_feature_dataset = {'mean': mean_t, 'std': std_t, 'epsilon_ref': float(eps_ref) if eps_ref is not None else None}
                        if 'input_dim' in rfn:
                            wise_editor.router_input_dim = int(rfn['input_dim'])
                        if 'feature_mode' in rfn and rfn['feature_mode']:
                            wise_editor.router_feature_mode = str(rfn['feature_mode']).lower()
                        self.logger.info("✅ 恢复路由特征归一化信息")
                except Exception as _e:
                    self.logger.warning(f"恢复路由特征归一化失败: {_e}")

            # attach editor ref if present
            if wise_editor is not None:
                setattr(base_model, 'wise_editor_ref', wise_editor)

        else:
            # Standard model checkpoint path
            self.logger.info("检测到标准模型 checkpoint，执行标准加载流程...")
            state_dict_to_load = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            try:
                base_model.load_state_dict(state_dict_to_load, strict=False)
            except Exception as e:
                self.logger.warning(f"加载标准模型权重时出现问题: {e}")
            added = _ensure_special_tokens(self.tokenizer)
            try:
                _sync_vocab_size(base_model, self.tokenizer, min_expand=added)
                _align_lora_output_dims(None)
            except Exception as emb_err:
                self.logger.warning(f"原始模型扩展特殊Token嵌入失败: {emb_err}")

        # finalize
        self.model = base_model.to(self.device)
        self.model.eval()

        # sync editor devices if present
        wise_editor_sync = getattr(self.model, 'wise_editor_ref', None)
        if wise_editor_sync is not None:
            try:
                target_device = next(self.model.parameters()).device
                wise_editor_sync.device = target_device
                if getattr(wise_editor_sync, 'router_classifier', None) is not None:
                    wise_editor_sync.router_classifier.to(target_device)
            except Exception as _e:
                self.logger.warning(f"WISE编辑器设备同步警告: {_e}")

        # record some metadata
        params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"模型加载/恢复完成，device={self.device}, params={params}")

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer

    def load_from_checkpoint(self, checkpoint_path: str, local_t5_dir: Optional[str] = None) -> None:
        """
        Load model weights from a checkpoint file (new or old format). This wraps the existing
        `_load_model` behavior but allows specifying a different checkpoint without re-creating
        the wrapper.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # 同上，兼容 PyTorch 2.6 的默认行为变更
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        # re-run the loader logic
        self._load_model()

    def load_dual_runtime(self, artifact_path: str, device: Optional[str] = None, threshold: Optional[float] = None):
        """
        Load a DualMemory runtime from artifacts and return DualMemoryRuntime instance.
        This delegates to src.dual_memory.load_dual_memory_runtime to keep logic colocated.
        """
        try:
            from src.dual_memory import load_dual_memory_runtime
        except Exception as e:
            raise RuntimeError("dual_memory utilities unavailable: %s" % e)
        runtime = load_dual_memory_runtime(artifact_path, device=device or self.device, threshold=threshold)
        return runtime

    def get_dual_runtime(self, artifact_path: str, device: Optional[str] = None, threshold: Optional[float] = None):
        """Convenience alias for load_dual_runtime."""
        return self.load_dual_runtime(artifact_path, device=device, threshold=threshold)

    @classmethod
    def from_dual_memory_artifacts(cls, artifact_path: str, device: Optional[str] = None, threshold_override: Optional[float] = None):
        """Construct a P5ModelWrapper directly from dual-memory artifacts.

        This will load the DualMemoryRuntime (model + tokenizer + adapter) and
        attach it to the wrapper instance as `dual_runtime` for convenient access.
        """
        try:
            from src.dual_memory import load_dual_memory_runtime
        except Exception as e:
            raise RuntimeError(f"Unable to import dual_memory utilities: {e}")

        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        runtime = load_dual_memory_runtime(artifact_path, device=device, threshold=threshold_override)

        # Create a minimal wrapper and attach runtime
        wrapper = cls(model_path=artifact_path, device=device)
        wrapper.model = runtime.model
        wrapper.tokenizer = runtime.tokenizer
        wrapper.dual_runtime = runtime
        return wrapper

    def _extract_items_from_text(self, text: str) -> List[str]:
        """从生成的文本中提取物品ID"""
        item_pattern = re.findall(r'item[_\s]*(\d+)', text, re.IGNORECASE)
        return item_pattern
    
    # 修复：该函数不再使用，其逻辑已移至 evaluate_datasets.py
    # 保留此函数是为了确保 P5ModelWrapper 的完整性，但实际调用已更改
    def generate_simple_recommendation(self, prompt: str, max_items: int = 20) -> List[str]:
        """简化的推荐生成方法（已废弃）"""
        self.logger.warning("该函数已废弃，其逻辑已移至 evaluate_datasets.py 以确保 WISE 路由和生成流程的原子性。")
        return []

    def predict_router_output(self, prompt: str) -> float:
            """
            [CIU 关键修复版]
            预测路由器的Sigmoid输出值。确保前向传播方式与训练时一致。
            """
            wise_editor = getattr(self.model, 'wise_editor_ref', None)
            if not wise_editor:
                return 0.0

            self.model.eval()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # [关键修复] 构造一个与训练时结构相同的输入字典
            dummy_labels = torch.full_like(inputs['input_ids'], self.tokenizer.pad_token_id)
            inputs['labels'] = dummy_labels

            def capture_activations(use_side: bool):
                wise_editor.set_routing_state(use_side)
                wise_editor.captured_activations.clear()
                with torch.no_grad():
                    _ = self.model(**inputs)
                outputs = []
                for name in wise_editor.router_target_layers:
                    if name in wise_editor.captured_activations:
                        outputs.append(wise_editor.captured_activations[name][:, 0, :])
                if not outputs:
                    return None
                if len(outputs) == 1:
                    return outputs[0]
                return torch.stack(outputs, dim=0).mean(dim=0)

            main_act = capture_activations(False)
            side_act = capture_activations(True)
            wise_editor.set_routing_state(False)

            if main_act is None or side_act is None:
                return 0.0

            delta_norm = (side_act - main_act).norm(p=2, dim=-1, keepdim=True)
            epsilon = wise_editor.epsilon_threshold or 1.0

            if wise_editor.router_feature_mode in ("delta-only", "delta_only"):
                feature_vec = torch.cat([delta_norm, delta_norm - epsilon], dim=-1)
            else:
                main_mean = main_act.mean(dim=-1, keepdim=True)
                feature_vec = torch.cat([delta_norm, main_mean], dim=-1)

            dataset_info = getattr(wise_editor, 'router_feature_dataset', None)
            if dataset_info is not None:
                mean = dataset_info['mean'].to(feature_vec.device)
                std = dataset_info['std'].to(feature_vec.device)
                feature_vec = (feature_vec - mean) / std.clamp(min=1e-6)

            try:
                router_device = next(wise_editor.router_classifier.parameters()).device
            except StopIteration:
                router_device = self.device
            feature_vec = feature_vec.to(router_device)

            router_logits = wise_editor.router_classifier(feature_vec)
            probability = torch.sigmoid(router_logits).mean().item()

            return probability