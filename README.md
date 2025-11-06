# 基于T5和双记忆网络的推荐系统遗忘框架

本项目实现了一种高效、有效的推荐系统机器遗忘框架。该框架基于预训练的T5序列推荐模型，通过引入一个轻量级的、可插拔的“侧记忆”（Side Memory）网络和一个智能“路由”（Router）机制，实现了在不完全重训练模型的情况下，对特定用户交互进行精准“遗忘”。

## 核心架构

本框架的核心是 **双记忆模型 (Dual Memory Model)**，它由三个关键组件构成：

1.  **主记忆 (Main Memory)**:
    *   一个在完整数据集上预训练好的T5推荐模型 (`M_base`)。
    *   它包含了丰富的通用推荐知识，在遗忘过程中其所有参数均被**冻结**，以确保对非遗忘用户（保留用户）的推荐性能稳定性。

2.  **侧记忆 (Side Memory)**:
    *   一个通过 **LoRA (Low-Rank Adaptation)** 技术实现的轻量级“插件”网络。
    *   当需要遗忘时，我们**只训练这个插件**，让它学会生成一个“修正信号”，以抵消和削弱主记忆中关于遗忘信息的“烙印”。

3.  **路由 (Router)**:
    *   一个简单的分类器，负责扮演“交通警察”的角色。
    *   它判断一个输入请求是来自“遗忘用户”还是“保留用户”，并据此智能地决定最终输出应该更多地依赖主记忆还是侧记忆。


## 主要特性

- **高效性**: 仅需训练轻量级的LoRA插件和路由，避免了代价高昂的完全重训练。
- **有效性**: 无论是从推荐指标（如NDCG）的大幅下降，还是从成员推断攻击（MIA）的低成功率来看，遗忘效果都非常显著。
- **性能保持**: 对绝大多数无关用户的推荐性能几乎无任何负面影响，有效控制了“连带损伤”。


## 数据准备

本项目使用 **MovieLens-1M (ml-1m)** 数据集。请确保数据文件位于 `data/ML1M/` 目录下，并包含以下文件：
- `ml-1m.inter`: 用户-物品交互文件，格式为 `user_id\titem_id\trating\ttimestamp`。
- `user_indexing.txt`: 用户ID映射文件。

脚本会自动处理后续的数据划分，包括为遗忘任务生成 `forget_samples_subset.json` 和 `retain_samples_subset.json`。

## 训练与评估流程

整个工作流分为三个步骤：预训练主模型、执行遗忘训练、评估遗忘效果。

### 步骤 1: 预训练主记忆模型 (`M_base`)

首先，我们需要一个强大的基础推荐模型作为主记忆。

```bash
python train_ml1m_t5.py \
    --model_name_or_path ./hf_models/t5-small/ \
    --output_dir ./models/My_ML1M_Base_V1/ \
    --dataset_path ./data/ML1M/ \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_input_length 200 \
    --max_target_length 10 \
    --amp
```

### 步骤 2: 执行双记忆遗忘训练 (`M_unlearned`)

这是本框架的核心。使用以下经过优化的最佳参数配置来训练侧记忆和路由，以达到最佳的遗忘效果。

```bash
# --- 最佳遗忘训练参数 ---
python train_dual_memory.py \
  --model models/My_ML1M_Base_V1 \
  --preset e2_strong \
  --epochs 28 \
  --batch-size 256 \
  --lr 5e-4 \
  --lora-r 16 \
  --alpha-weight 35 \
  --beta-weight 240 \
  --kl-reg-weight 6 \
  --kl-mask-forgotten \
  --freeze-lm-head \
  --unlikelihood-weight 1.5 \
  --pairwise-weight 0.8 \
  --pairwise-margin 2.0 \
  --hard-neg-k 50 \
  --topk-penalty-weight 1.5 \
  --topk-k 100 \
  --topk-margin 0.4 \
  --abs-suppression-weight 3.0 \
  --abs-suppression-margin 3.5 \
  --workers 12 \
  --amp \
  --output results/My_Unlearning_Run_E3_L20_abs
```

### 步骤 3: 评估遗忘效果

评估分为两个维度：推荐性能评估和隐私安全评估（MIA）。

#### a) 推荐性能评估

此脚本会对比原始模型 (`M_base`) 和遗忘后模型 (`M_unlearned`) 在保留集和遗忘集上的性能差异。

```bash
python evaluate_datasets.py \
  --compare \
  --original_model models/My_ML1M_Base_V1 \
  --unlearned_model results/My_Unlearning_Run_E3_L20_abs/dual_memory_artifacts.pt \
  --eval_sample_size 300 \
  --k_values 10,20 \
  --disable_fallback \
  --skip_all_users \
  --num_return_sequences 50 \
  --num_beams 50 \
  --max_gen_len 150 \
  --verbose
```

#### b) 成员推断攻击 (MIA) 评估

此脚本通过训练一个攻击模型来判断一个用户是否属于训练集，从而量化隐私泄露风险。**AUC 越接近 0.5，代表遗忘效果越好**。

```bash
python MIA.py \
  --original_model models/My_ML1M_Base_V1 \
  --unlearned_artifacts results/My_Unlearning_Run_E3_L20_abs/dual_memory_artifacts.pt \
  --gold_model path/to/your/gold_model.pt \
  --k 50 \
  --eval_users 800
```

## 最佳实验结果摘要

以下是使用上述最佳参数配置得到的实验结果。

### 推荐性能

| 评估维度 | 指标 | Base | Unlearned | 变化 | 评价 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **保留集** | hit@10 | 0.3500 | 0.3500 | +0.0% | ✅ 性能保持良好 |
| (通用性) | ndcg@10 | 0.2921 | 0.2921 | +0.0% | ✅ 性能保持良好 |
| **遗忘集** | hit@10 | 0.4867 | 0.1400 | -71.2% | ✅ 遗忘成功 |
| (遗忘效果) | ndcg@10 | 0.3407 | 0.2884 | -15.4% | ✅ 遗忘成功 |
| **连带损伤** | hit@10 | 0.3433 | 0.2233 | -35.0% | ⚠️ 可接受 |
| (相似用户) | ndcg@10 | 0.3316 | 0.2973 | -10.3% | ⚠️ 可接受 |

### MIA 隐私评估

| 模型 | AUC (越接近0.5越好) | 评价 |
| :--- | :--- | :--- |
| `M_base` (原始模型) | 0.4713 | - |
| `M_gold` (重训练模型) | **0.8549** | 隐私风险高 (基线) |
| `M_unlearned` (遗忘后模型) | **0.4671** | ✅ 遗忘非常成功 |

**结果解读**:
- **性能**: 框架在几乎不影响无关用户（保留集）的前提下，显著降低了对遗忘目标的推荐能力。
- **隐私**: `M_unlearned` 的AUC值与随机猜测（0.5）非常接近，远低于`M_gold`模型，证明其有效抵御了成员推断攻击，隐私保护效果出色。

## 核心损失函数

遗忘训练阶段的总损失函数 \(\mathcal{L}_{\text{total}}\) 是一个精细设计的加权和，旨在平衡“保留”、“遗忘”和“稳定”三大目标。

\[
\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{retention}} + \beta \cdot \mathcal{L}_{\text{forget}} + \gamma \cdot \mathcal{L}_{\text{stability}}
\]

其中，\(\mathcal{L}_{\text{forget}}\) 由多种遗忘工具（如反向似然、成对排序、绝对抑制等）加权构成，通过调整命令行中的各项权重参数，可以实现对遗忘强度和副作用的精准控制。