#!/usr/bin/env python3
"""
MIA (Membership Inference Attack) adapter for P5/T5 dual-memory unlearning on ML-1M.

What this script does now:
- Loads ML-1M interactions (the same files used by train_dual_memory/evaluate_datasets).
- Generates recommendations from:
  * Original model (base P5 checkpoint)
  * Unlearned dual-memory artifacts (side memory used for forgotten users, main memory otherwise)
  * Optional "gold" retrained model (baseline upper bound)
- Builds NDCG-aware features for a membership inference attacker (Logistic Regression).
- Reports attacker Accuracy/AUC and forgetting-focused summaries (NDCG@K on forgotten targets, exposure drop, selectivity).

Usage example:
  python MIA.py \
    --original_model models/My_ML1M_Base_V1/model.safetensors \
    --unlearned_artifacts results/My_Unlearning_Run_E3_L20_abs/dual_memory_artifacts.pt \
    --k 50 --eval_users 800 --num_beams 20 --num_return_sequences 20 --disable_fallback

Optional gold (retrained) model:
  --gold_model models/My_ML1M_Retrain.pt
"""

import os
import re
import json
import time
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

from src.model_wrapper import P5ModelWrapper
from src.dual_memory import load_dual_memory_runtime

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
ML1M_DIR = os.path.join(PROJECT_ROOT, 'data', 'ML1M')
INTER_FILE = os.path.join(ML1M_DIR, 'ml-1m.inter')
USER_INDEX_FILE = os.path.join(ML1M_DIR, 'user_indexing.txt')
FORGET_FILE = os.path.join(PROJECT_ROOT, 'results', 'forget_samples_subset.json')

def _load_mappings() -> Dict[str, Dict[str, str]]:
    user_map = {line.strip().split()[0]: line.strip().split()[1] for line in open(USER_INDEX_FILE, 'r', encoding='utf-8')}
    mapped_to_user = {v: k for k, v in user_map.items()}
    return {'user_to_mapped': user_map, 'mapped_to_user': mapped_to_user}

def load_ml1m_splits(rating_threshold: int = 4):
    """Load ML1M interactions, return train_df, test_df, mappings.
    Split by timestamp 80/20 to match evaluate_datasets.py
    """
    if not os.path.exists(INTER_FILE):
        raise FileNotFoundError(f"交互文件不存在: {INTER_FILE}")
    maps = _load_mappings()
    df = pd.read_csv(INTER_FILE, sep='\t', header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'],
                     dtype={'user_id': str, 'item_id': str, 'rating': str, 'timestamp': str})
    df['user_id'] = df['user_id'].astype(str).str.strip()
    df['item_id'] = df['item_id'].astype(str).str.strip()
    df['mapped_user_id'] = df['user_id'].map(maps['user_to_mapped']).astype(str)
    df['mapped_item_id'] = df['item_id'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('timestamp')
    split_point = int(len(df) * 0.8)
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()
    return train_df, test_df, maps

def load_forget_requests() -> Dict[str, List[str]]:
    """Load unlearning requests: map mapped_user_id -> [item_ids_to_forget]."""
    if not os.path.exists(FORGET_FILE):
        return {}
    maps = _load_mappings()
    with open(FORGET_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    requests = defaultdict(list)
    for s in data:
        orig_uid = str(s.get('user_id'))
        mapped_uid = maps['user_to_mapped'].get(orig_uid)
        if not mapped_uid:
            continue
        items = [str(x) for x in (s.get('suppression_targets') or [])]
        if items:
            requests[str(mapped_uid)].extend(items)
    return requests

def _dcg_from_ranks(ranks: List[int]) -> float:
    return sum(1.0 / math.log2(r + 1) for r in ranks)

def _ndcg_from_hits(ranks: List[int], k: int) -> float:
    if not ranks:
        return 0.0
    dcg = _dcg_from_ranks(ranks)
    # ideal ranks are 1..min(len(ranks), k)
    L = min(len(ranks), k)
    idcg = _dcg_from_ranks(list(range(1, L + 1)))
    return float(dcg / idcg) if idcg > 0 else 0.0

def extract_features(user_id: str,
                     user_history: Set[str],
                     recommendation_list,
                     forget_targets: Optional[Set[str]] = None,
                     k: int = 50) -> List[float]:
    """
    提取 MIA 特征（对排名/曝光更敏感），加入 NDCG 指标：
     - 命中数、DCG@K、首命中RR、命中排名标准差、最小命中排名
     - NDCG@K(基于历史)
     - 忘记目标NDCG@K（若提供 forget_targets）
    """
    # 标准化推荐列表为 item_ids 和 scores（scores 目前对 T5 不可用，保留占位）
    if isinstance(recommendation_list, dict):
        sorted_items = sorted(recommendation_list.items(), key=lambda x: x[1], reverse=True)
        item_ids = [str(it) for it, _ in sorted_items]
        scores = [float(sc) for _, sc in sorted_items]
    else:
        item_ids = [str(it) for it in recommendation_list]
        scores = None

    if len(item_ids) > k:
        item_ids = item_ids[:k]
        if scores is not None:
            scores = scores[:k]
    elif len(item_ids) < k:
        pad = k - len(item_ids)
        item_ids.extend(['-1'] * pad)
        if scores is not None:
            scores.extend([0.0] * pad)

    # 命中历史的排名
    ranks_of_hits = []
    hit_scores = []
    for i, (it, sc) in enumerate(zip(item_ids, scores if scores is not None else [None] * len(item_ids))):
        if it in user_history:
            ranks_of_hits.append(i + 1)
            if sc is not None:
                hit_scores.append(sc)

    hit_count = len(ranks_of_hits)
    f_hit_count = float(hit_count)
    f_history_dcg = sum(1.0 / math.log2(r + 1) for r in ranks_of_hits) if hit_count > 0 else 0.0
    f_rr_first_hit = 1.0 / min(ranks_of_hits) if hit_count > 0 else 0.0
    f_std_dev_rank = float(np.std(ranks_of_hits)) if hit_count > 1 else 0.0
    f_min_rank = float(min(ranks_of_hits)) if hit_count > 0 else float(k + 1)
    f_avg_score = float(np.mean(scores)) if scores is not None and len(scores) > 0 else 0.0
    f_score_variance = float(np.var(scores)) if scores is not None and len(scores) > 1 else 0.0
    f_hit_avg_score = float(np.mean(hit_scores)) if hit_scores else 0.0
    f_hit_score_std = float(np.std(hit_scores)) if len(hit_scores) > 1 else 0.0
    f_ndcg_hist = _ndcg_from_hits(ranks_of_hits, k)

    f_ndcg_forget = 0.0
    if forget_targets:
        ranks_f = []
        fset = set(str(x) for x in forget_targets)
        for i, it in enumerate(item_ids):
            if it in fset:
                ranks_f.append(i + 1)
        f_ndcg_forget = _ndcg_from_hits(ranks_f, k)

    return [
        f_hit_count,
        f_history_dcg,
        f_rr_first_hit,
        f_std_dev_rank,
        f_min_rank,
        f_avg_score,
        f_score_variance,
        f_hit_avg_score,
        f_hit_score_std,
        f_ndcg_hist,
        f_ndcg_forget,
    ]

def prepare_training_data(retain_interactions: Dict[str, Set[str]],
                         test_interactions: Dict[str, Set[str]],
                         base_recommendations: Dict[str, List[str]],
                         forget_requests: Optional[Dict[str, List[str]]] = None,
                         k: int = 50):
    """
    准备攻击模型的训练数据
    :param retain_interactions: 保留集用户交互（正样本）
    :param test_interactions: 测试集用户交互（负样本）
    :param base_recommendations: 基础模型推荐结果
    :param k: Top-K值
    :return: 特征矩阵X和标签向量y
    """
    # 随机抽样一部分用户用于训练
    retain_users = list(retain_interactions.keys())
    test_users = list(test_interactions.keys())
    
    # 抽样相同样本数量的用户
    sample_size = min(len(retain_users), len(test_users), 1000)  # 限制样本数量以提高效率
    sampled_retain_users = random.sample(retain_users, sample_size)
    sampled_test_users = random.sample(test_users, sample_size)
    
    X = []  # 特征
    y = []  # 标签: 1表示成员，0表示非成员
    
    # 处理正样本（保留集用户）
    for user_id in sampled_retain_users:
        if user_id in base_recommendations:
            user_history = retain_interactions[user_id]
            rec_list = base_recommendations[user_id]
            forget_t = set(forget_requests.get(user_id, [])) if forget_requests else None
            features = extract_features(user_id, user_history, rec_list, forget_t, k)
            X.append(features)
            y.append(1)
    
    # 处理负样本（测试集用户）
    for user_id in sampled_test_users:
        if user_id in base_recommendations:
            user_history = test_interactions[user_id]
            rec_list = base_recommendations[user_id]
            forget_t = set(forget_requests.get(user_id, [])) if forget_requests else None
            features = extract_features(user_id, user_history, rec_list, forget_t, k)
            X.append(features)
            y.append(0)
    
    return np.array(X), np.array(y)

def train_attack_model(X, y, test_size=0.2, tune_hyperparameters=True, random_state=42):
    """
    改进的训练攻击模型函数，包含验证集、超参数调优（可选）和性能报告。

    Args:
        X (np.ndarray or torch.Tensor): 攻击模型的完整特征数据 (members + non-members)。
        y (np.ndarray or torch.Tensor): 对应的标签 (1 for members, 0 for non-members)。
        test_size (float): 用于内部验证集的比例。
        tune_hyperparameters (bool): 是否执行超参数搜索。
        random_state (int): 用于可复现性。

    Returns:
        tuple: (best_clf, scaler, report)
               best_clf: 训练好的最佳分类器。
               scaler: 在训练集上拟合好的StandardScaler。
               report (dict): 包含训练和验证性能指标的报告。
    """
    # 确保数据是 numpy array
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # --- 1. 划分训练集和内部验证集 ---
    X_attack_train, X_attack_val, y_attack_train, y_attack_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y # stratify 保证类别比例
    )
    print(f"攻击者训练数据: {len(X_attack_train)} 样本")
    print(f"攻击者验证数据: {len(X_attack_val)} 样本")

    # --- 2. 特征标准化 (仅在训练集上fit, 然后transform训练集和验证集) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_attack_train)
    X_val_scaled = scaler.transform(X_attack_val) # 使用相同的 scaler

    best_clf = None
    report = {}

    # --- 3. 模型训练与选择 ---
    if tune_hyperparameters:
        print("正在为逻辑回归执行超参数调优...")
        # 定义参数网格 (增加迭代次数以确保收敛)
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']} 
        
        # 使用GridSearchCV进行交叉验证调优
        grid_search = GridSearchCV(
            LogisticRegression(random_state=random_state, max_iter=200, class_weight='balanced'),
            param_grid,
            cv=5, # 5折交叉验证
            scoring='roc_auc', # 以AUC为主，稳定评估
            n_jobs=-1 # 使用所有CPU核心
        )
        grid_search.fit(X_train_scaled, y_attack_train)

        print(f"通过GridSearchCV找到的最佳超参数: {grid_search.best_params_}")
        best_clf = grid_search.best_estimator_ # 获取在整个训练集上用最佳参数训练好的模型
        report['best_params'] = grid_search.best_params_
        # 记录CV统计
        try:
            cv_res = grid_search.cv_results_
            means = cv_res.get('mean_test_score')
            stds = cv_res.get('std_test_score')
            if means is not None and stds is not None:
                report['cv_auc_mean'] = float(np.max(means)) if len(means) else None
                # std对应最佳参数索引
                best_idx = int(grid_search.best_index_)
                report['cv_auc_std'] = float(stds[best_idx]) if best_idx < len(stds) else None
        except Exception:
            pass

    else:
        # 如果不调优，直接训练默认模型
        print("使用默认参数 (C=1.0) 训练逻辑回归...")
        # 同样增加迭代次数并平衡类别权重
        clf = LogisticRegression(random_state=random_state, max_iter=2000, class_weight='balanced')
        clf.fit(X_train_scaled, y_attack_train)
        best_clf = clf
        report['best_params'] = 'default (C=1.0)'

    # --- 4. 在训练集和验证集上评估最终模型 ---
    y_train_pred_proba = best_clf.predict_proba(X_train_scaled)[:, 1]
    y_val_pred_proba = best_clf.predict_proba(X_val_scaled)[:, 1]
    
    y_train_pred_label = (y_train_pred_proba > 0.5).astype(int)
    y_val_pred_label = (y_val_pred_proba > 0.5).astype(int)

    # 检查验证集标签是否单一，以防AUC计算报错
    val_auc = 0.5
    if len(np.unique(y_attack_val)) > 1:
        val_auc = roc_auc_score(y_attack_val, y_val_pred_proba)
        
    train_auc = 0.5
    if len(np.unique(y_attack_train)) > 1:
          train_auc = roc_auc_score(y_attack_train, y_train_pred_proba)

    report['train'] = {
        'accuracy': accuracy_score(y_attack_train, y_train_pred_label),
        'auc': train_auc
    }
    report['validation'] = {
        'accuracy': accuracy_score(y_attack_val, y_val_pred_label),
        'auc': val_auc
    }

    print("\n攻击者性能报告:")
    print(f"  训练集 Accuracy: {report['train']['accuracy']:.4f}")
    print(f"  训练集 AUC: {report['train']['auc']:.4f}")
    print(f"  验证集 Accuracy: {report['validation']['accuracy']:.4f}")
    print(f"  验证集 AUC: {report['validation']['auc']:.4f}")

    # 检查过拟合
    overfitting_threshold = 0.1 # 可以调整这个阈值
    if report['train']['accuracy'] > report['validation']['accuracy'] + overfitting_threshold:
        print("警告: 检测到潜在过拟合 (训练集ACC >> 验证集ACC)")
    if report['train']['auc'] > report['validation']['auc'] + overfitting_threshold:
         print("警告: 检测到潜在过拟合 (训练集AUC >> 验证集AUC)")

    return best_clf, scaler, report

def evaluate_model(clf, scaler, X_test, y_test):
    """
    评估模型性能
    """
    # 特征标准化
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]  # 正类的概率
    
    acc = accuracy_score(y_test, y_pred)
    # 只有当y_test中包含两个类别时才计算AUC
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = 0.5  # 当只有一类时，AUC无定义，设为0.5
    
    return acc, auc

def _summarize_scores(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """Threshold-free and thresholded summaries for a binary score.
    Returns: {
      'pos_rate': float,
      'acc@0.5': float,
      'bal_acc@0.5': float,
      'best_acc': float,
      'best_acc_threshold': float,
      'youdenJ_acc': float,
      'youdenJ_threshold': float,
      'auc': float,
      'confusion@0.5': [tn, fp, fn, tp]
    }
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    assert y_true.shape[0] == y_scores.shape[0]
    res = {}
    pos_rate = float(np.mean(y_true)) if y_true.size else 0.0
    res['pos_rate'] = pos_rate
    # default 0.5 threshold
    y_hat = (y_scores > 0.5).astype(int)
    res['acc@0.5'] = float(accuracy_score(y_true, y_hat))
    res['bal_acc@0.5'] = float(balanced_accuracy_score(y_true, y_hat))
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    res['confusion@0.5'] = [int(tn), int(fp), int(fn), int(tp)]
    # AUC
    auc = 0.5
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_scores))
    res['auc'] = auc
    # Best accuracy over a grid of thresholds (avoid using labels directly to prevent optimistic bias)
    thresholds = np.unique(np.concatenate([np.linspace(0,1,201), y_scores]))
    best_acc = -1.0
    best_thr = 0.5
    for thr in thresholds:
        acc = accuracy_score(y_true, (y_scores > thr).astype(int))
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    res['best_acc'] = float(best_acc)
    res['best_acc_threshold'] = float(best_thr)
    # Youden's J (maximize TPR-FPR on ROC)
    try:
        fpr, tpr, thr = roc_curve(y_true, y_scores)
        j = tpr - fpr
        j_idx = int(np.argmax(j))
        j_thr = float(thr[j_idx]) if j_idx < len(thr) else 0.5
        y_hat_j = (y_scores > j_thr).astype(int)
        res['youdenJ_threshold'] = j_thr
        res['youdenJ_acc'] = float(accuracy_score(y_true, y_hat_j))
    except Exception:
        res['youdenJ_threshold'] = 0.5
        res['youdenJ_acc'] = res['acc@0.5']
    return res

def evaluate_forget_effectiveness(
    clf,
    scaler,
    forget_interactions: Dict[str, Set[str]],
    recommendations_mbase: Dict[str, List[str]],
    recommendations_munlearned: Dict[str, List[str]],
    recommendations_mgold: Optional[Dict[str, List[str]]],
    encoded_forget_interactions: Dict[str, Set[str]],
    encoded_test_interactions: Dict[str, Set[str]],
    encoded_retain_interactions: Dict[str, Set[str]],
    forget_requests: Dict[str, List[str]],
    k: int = 50,
):
    """
    评估遗忘效果
    :param clf: 训练好的攻击模型
    :param scaler: 特征标准化器
    :param forget_interactions: 遗忘集用户交互（原始ID）
    :param recommendations_mbase: 基础模型推荐结果（编码后ID）
    :param recommendations_munlearned: 遗忘后模型推荐结果（编码后ID）
    :param recommendations_mgold: 重训模型推荐结果（编码后ID）
    :param encoded_forget_interactions: 编码后的遗忘集用户交互
    :param encoded_test_interactions: 编码后的测试集用户交互（用于负样本）
    :param encoded_retain_interactions: 编码后的保留集用户交互
    :param k: Top-K值
    :return: 各模型在遗忘集上的ACC和AUC
    """
    forget_users = list(forget_interactions.keys())
    encoded_forget_users = list(encoded_forget_interactions.keys())
    encoded_test_users = list(encoded_test_interactions.keys())
    encoded_retain_users = list(encoded_retain_interactions.keys())
    
    results = {}
    
    # 随机选择测试集用户作为负样本
    test_users_sample = random.sample(encoded_test_users, 
                                    min(len(encoded_forget_users), len(encoded_test_users)))
    
    # 随机选择部分retain_set用户用于AUC计算
    retain_users_sample = random.sample(encoded_retain_users, 
                                      min(len(encoded_forget_users), len(encoded_retain_users)))
    
    # 评估基础模型 (M_base)
    # ACC计算：forget_set (标签1) vs test_set (标签0)
    X_mbase_acc_positive = []
    for user_id in encoded_forget_users:
        if user_id in recommendations_mbase:
            user_history = encoded_forget_interactions[user_id]
            rec_list = recommendations_mbase[user_id]
            features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
            X_mbase_acc_positive.append(features)
    
    X_mbase_acc_negative = []
    for user_id in test_users_sample:
        if user_id in recommendations_mbase:
            user_history = encoded_test_interactions[user_id]
            rec_list = recommendations_mbase[user_id]
            features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
            X_mbase_acc_negative.append(features)
    
    if X_mbase_acc_positive and X_mbase_acc_negative:
        X_mbase_acc = np.array(X_mbase_acc_positive + X_mbase_acc_negative)
        X_mbase_acc_scaled = scaler.transform(X_mbase_acc)
        y_pred_proba_mbase_acc = clf.predict_proba(X_mbase_acc_scaled)[:, 1]
        # 正样本标签为1，负样本标签为0
        y_true_acc = [1] * len(X_mbase_acc_positive) + [0] * len(X_mbase_acc_negative)
        acc_mbase = accuracy_score(y_true_acc, (y_pred_proba_mbase_acc > 0.5).astype(int))
        acc_summ_base = _summarize_scores(np.array(y_true_acc), y_pred_proba_mbase_acc)
        print(f"M_base预测概率 - 平均值: {np.mean(y_pred_proba_mbase_acc):.4f}, 标准差: {np.std(y_pred_proba_mbase_acc):.4f}")
        print(f"M_base预测概率范围: [{np.min(y_pred_proba_mbase_acc):.4f}, {np.max(y_pred_proba_mbase_acc):.4f}]")
        
        # AUC计算：forget_set + 部分retain_set (标签1) vs test_set (标签0)
        X_mbase_auc_positive_forget = X_mbase_acc_positive  # forget_set用户
        
        X_mbase_auc_positive_retain = []
        for user_id in retain_users_sample:
            if user_id in recommendations_mbase:
                user_history = encoded_retain_interactions[user_id]
                rec_list = recommendations_mbase[user_id]
                features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
                X_mbase_auc_positive_retain.append(features)
        
        X_mbase_auc_positive = X_mbase_auc_positive_forget + X_mbase_auc_positive_retain
        X_mbase_auc_negative = X_mbase_acc_negative  # test_set用户
        
        if X_mbase_auc_positive and X_mbase_auc_negative:
            X_mbase_auc = np.array(X_mbase_auc_positive + X_mbase_auc_negative)
            X_mbase_auc_scaled = scaler.transform(X_mbase_auc)
            y_pred_proba_mbase_auc = clf.predict_proba(X_mbase_auc_scaled)[:, 1]
            # 正样本标签为1，负样本标签为0
            y_true_auc = [1] * len(X_mbase_auc_positive) + [0] * len(X_mbase_auc_negative)
            auc_mbase = roc_auc_score(y_true_auc, y_pred_proba_mbase_auc)
        else:
            auc_mbase = 0.5
        results['M_base'] = {
            'ACC': acc_mbase,
            'AUC': auc_mbase,
            'pred_mean': float(np.mean(y_pred_proba_mbase_acc)),
            'acc_detail': acc_summ_base,
        }
    
    # 评估遗忘后模型 (M_unlearned)
    # ACC计算：forget_set (标签1) vs test_set (标签0)
    X_munlearned_acc_positive = []
    for user_id in encoded_forget_users:
        if user_id in recommendations_munlearned:
            user_history = encoded_forget_interactions[user_id]
            rec_list = recommendations_munlearned[user_id]
            features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
            X_munlearned_acc_positive.append(features)
    
    X_munlearned_acc_negative = []
    for user_id in test_users_sample:
        if user_id in recommendations_munlearned:
            user_history = encoded_test_interactions[user_id]
            rec_list = recommendations_munlearned[user_id]
            features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
            X_munlearned_acc_negative.append(features)
    
    if X_munlearned_acc_positive and X_munlearned_acc_negative:
        X_munlearned_acc = np.array(X_munlearned_acc_positive + X_munlearned_acc_negative)
        X_munlearned_acc_scaled = scaler.transform(X_munlearned_acc)
        y_pred_proba_munlearned_acc = clf.predict_proba(X_munlearned_acc_scaled)[:, 1]
        # 正样本标签为1，负样本标签为0
        y_true_acc = [1] * len(X_munlearned_acc_positive) + [0] * len(X_munlearned_acc_negative)
        acc_munlearned = accuracy_score(y_true_acc, (y_pred_proba_munlearned_acc > 0.5).astype(int))
        acc_summ_unl = _summarize_scores(np.array(y_true_acc), y_pred_proba_munlearned_acc)
        print(f"M_unlearned预测概率 - 平均值: {np.mean(y_pred_proba_munlearned_acc):.4f}, 标准差: {np.std(y_pred_proba_munlearned_acc):.4f}")
        print(f"M_unlearned预测概率范围: [{np.min(y_pred_proba_munlearned_acc):.4f}, {np.max(y_pred_proba_munlearned_acc):.4f}]")
        
        # AUC计算：forget_set + 部分retain_set (标签1) vs test_set (标签0)
        X_munlearned_auc_positive_forget = X_munlearned_acc_positive  # forget_set用户
        
        X_munlearned_auc_positive_retain = []
        for user_id in retain_users_sample:
            if user_id in recommendations_munlearned:
                user_history = encoded_retain_interactions[user_id]
                rec_list = recommendations_munlearned[user_id]
                features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
                X_munlearned_auc_positive_retain.append(features)
        
        X_munlearned_auc_positive = X_munlearned_auc_positive_forget + X_munlearned_auc_positive_retain
        X_munlearned_auc_negative = X_munlearned_acc_negative  # test_set用户
        
        if X_munlearned_auc_positive and X_munlearned_auc_negative:
            X_munlearned_auc = np.array(X_munlearned_auc_positive + X_munlearned_auc_negative)
            X_munlearned_auc_scaled = scaler.transform(X_munlearned_auc)
            y_pred_proba_munlearned_auc = clf.predict_proba(X_munlearned_auc_scaled)[:, 1]
            # 正样本标签为1，负样本标签为0
            y_true_auc = [1] * len(X_munlearned_auc_positive) + [0] * len(X_munlearned_auc_negative)
            auc_munlearned = roc_auc_score(y_true_auc, y_pred_proba_munlearned_auc)
        else:
            auc_munlearned = 0.5
        results['M_unlearned'] = {
            'ACC': acc_munlearned,
            'AUC': auc_munlearned,
            'pred_mean': float(np.mean(y_pred_proba_munlearned_acc)),
            'acc_detail': acc_summ_unl,
        }
    
    # 评估重训模型 (M_gold)
    # ACC和AUC计算：retain_set (标签1) vs forget_set + test_set (标签0)
    X_mgold_positive = []
    if recommendations_mgold is not None:
        for user_id in retain_users_sample:
            if user_id in recommendations_mgold:
                user_history = encoded_retain_interactions[user_id]
                rec_list = recommendations_mgold[user_id]
                features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
                X_mgold_positive.append(features)
    
    # 负样本：forget set用户
    X_mgold_negative_forget = []
    if recommendations_mgold is not None:
        for user_id in encoded_forget_users:
            if user_id in recommendations_mgold:
                user_history = encoded_forget_interactions[user_id]
                rec_list = recommendations_mgold[user_id]
                features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
                X_mgold_negative_forget.append(features)
    
    # 负样本：test set用户
    X_mgold_negative_test = []
    if recommendations_mgold is not None:
        for user_id in test_users_sample:
            if user_id in recommendations_mgold:
                user_history = encoded_test_interactions[user_id]
                rec_list = recommendations_mgold[user_id]
                features = extract_features(user_id, user_history, rec_list, set(forget_requests.get(user_id, [])), k)
                X_mgold_negative_test.append(features)
    
    # 合并负样本
    if recommendations_mgold is not None:
        X_mgold_negative = X_mgold_negative_forget + X_mgold_negative_test
        if X_mgold_positive and X_mgold_negative:
            X_mgold = np.array(X_mgold_positive + X_mgold_negative)
            X_mgold_scaled = scaler.transform(X_mgold)
            y_pred_proba_mgold = clf.predict_proba(X_mgold_scaled)[:, 1]
            y_true = [1] * len(X_mgold_positive) + [0] * len(X_mgold_negative)
            acc_mgold = accuracy_score(y_true, (y_pred_proba_mgold > 0.5).astype(int))
            auc_mgold = roc_auc_score(y_true, y_pred_proba_mgold)
            print(f"M_gold预测概率 - 平均值: {np.mean(y_pred_proba_mgold):.4f}, 标准差: {np.std(y_pred_proba_mgold):.4f}")
            print(f"M_gold预测概率范围: [{np.min(y_pred_proba_mgold):.4f}, {np.max(y_pred_proba_mgold):.4f}]")
            results['M_gold'] = {
                'ACC': acc_mgold,
                'AUC': auc_mgold,
                'pred_mean': float(np.mean(y_pred_proba_mgold)),
                'acc_detail': _summarize_scores(np.array(y_true), y_pred_proba_mgold),
            }
        elif X_mgold_positive or X_mgold_negative:
            X_mgold = np.array(X_mgold_positive if X_mgold_positive else X_mgold_negative)
            X_mgold_scaled = scaler.transform(X_mgold)
            y_pred_proba_mgold = clf.predict_proba(X_mgold_scaled)[:, 1]
            y_true = [1] * len(X_mgold_positive) if X_mgold_positive else [0] * len(X_mgold_negative)
            acc_mgold = accuracy_score(y_true, (y_pred_proba_mgold > 0.5).astype(int)) if len(np.unique(y_true)) > 1 else 0.5
            auc_mgold = roc_auc_score(y_true, y_pred_proba_mgold) if len(np.unique(y_true)) > 1 else 0.5
            print(f"M_gold预测概率 - 平均值: {np.mean(y_pred_proba_mgold):.4f}, 标准差: {np.std(y_pred_proba_mgold):.4f}")
            print(f"M_gold预测概率范围: [{np.min(y_pred_proba_mgold):.4f}, {np.max(y_pred_proba_mgold):.4f}]")
            results['M_gold'] = {
                'ACC': acc_mgold,
                'AUC': auc_mgold,
                'pred_mean': float(np.mean(y_pred_proba_mgold)),
                'acc_detail': _summarize_scores(np.array(y_true), y_pred_proba_mgold) if len(np.unique(y_true))>1 else None,
            }
    
    return results

def build_interaction_sets(train_df: pd.DataFrame, test_df: pd.DataFrame, rating_threshold: int = 4):
    """Build per-user interaction sets for retain (train>=threshold) and test (test>=threshold)."""
    retain_interactions: Dict[str, Set[str]] = defaultdict(set)
    test_interactions: Dict[str, Set[str]] = defaultdict(set)
    tdf = train_df[train_df['rating'] >= rating_threshold]
    for _, row in tdf.iterrows():
        retain_interactions[str(row['mapped_user_id'])].add(str(row['mapped_item_id']))
    sdf = test_df[test_df['rating'] >= rating_threshold]
    for _, row in sdf.iterrows():
        test_interactions[str(row['mapped_user_id'])].add(str(row['mapped_item_id']))
    return retain_interactions, test_interactions

def build_forget_interactions_from_requests(requests: Dict[str, List[str]]):
    """Build forget_interactions as {mapped_user_id: set(items_to_forget)}"""
    return {uid: set(items) for uid, items in requests.items()}

def _prompt_for_user(mapped_uid: str, train_df: pd.DataFrame, forget_items: Optional[List[str]] = None, history_window: int = 20) -> str:
    user_hist = train_df[train_df['mapped_user_id'] == mapped_uid]['mapped_item_id'].astype(str).tolist()
    extra = list(forget_items or [])
    full = user_hist + extra
    hist = " ".join(f"item_{it}" for it in full[-history_window:]) if full else "<empty>"
    return f"User {mapped_uid} recent history: {hist}. Recommend next item."

def generate_recommendations_for_users(model_like,
                                       tokenizer,
                                       user_ids: List[str],
                                       train_df: pd.DataFrame,
                                       forget_requests: Dict[str, List[str]],
                                       use_side_for_forget: bool,
                                       k: int,
                                       num_beams: int,
                                       num_return_sequences: int,
                                       disable_fallback: bool = True,
                                       device: Optional[str] = None,
                                       base_temperature: float = 1.0,
                                       side_temperature: float = 1.2,
                                       use_entropy_fallback: bool = True,
                                       conf_fallback_threshold: Optional[float] = 0.85,
                                       min_unique_ratio: float = 0.3) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Generate top-k recommendations. model_like can be a raw T5 model or a DualMemoryRuntime.
    If it's a DualMemoryRuntime, we switch adapter to side for forget users when use_side_for_forget=True.
    """
    is_dual = hasattr(model_like, 'adapter') and hasattr(model_like, 'generate')
    recs: Dict[str, List[str]] = {}
    item_pattern = re.compile(r'item_(\d+)', re.IGNORECASE)
    device = device or (model_like.device if hasattr(model_like, 'device') else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # lazy import to avoid hard dependency if transformers changes
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
        """Add a large negative penalty to a set of token ids every step.
        This is a soft guardrail to complement bad_words_ids (sequence-level block).
        """
        def __init__(self, banned_token_ids: Set[int], penalty: float = 20.0) -> None:
            self.banned = set(int(x) for x in banned_token_ids)
            self.penalty = float(max(0.0, penalty))
        def __call__(self, input_ids, scores):
            if not self.banned or self.penalty <= 0:
                return scores
            # scores shape: (batch*beam, vocab)
            if hasattr(scores, 'index_fill_'):
                # create a mask vector of size vocab
                try:
                    import torch as _torch
                    vocab = scores.size(-1)
                    if self.banned:
                        idx = _torch.tensor(list(self.banned), device=scores.device, dtype=_torch.long)
                        valid = (idx >= 0) & (idx < vocab)
                        idx = idx[valid]
                        if idx.numel() > 0:
                            scores.index_fill_(dim=-1, index=idx, value=(scores.min().item() - self.penalty))
                except Exception:
                    pass
            return scores

    def _build_processors(temp: float, penalty_tokens: Optional[Set[int]] = None, penalty_val: float = 0.0):
        procs = LogitsProcessorList()
        if temp and abs(float(temp) - 1.0) > 1e-6:
            procs.append(TemperatureLogitsProcessor(float(temp)))
        if penalty_tokens and penalty_val > 0:
            procs.append(TokenPenaltyProcessor(set(penalty_tokens), float(penalty_val)))
        return procs

    def _encode_bad_words(_tokenizer, forbidden_items: List[str]) -> Tuple[List[List[int]], Set[int]]:
        """Build bad_words_ids sequences and a flat set of token ids to penalize (first subword heuristic)."""
        seqs: List[List[int]] = []
        flat: Set[int] = set()
        for it in forbidden_items:
            text = f"item_{it}"
            ids = _tokenizer.encode(text, add_special_tokens=False)
            if ids:
                seqs.append(ids)
                flat.add(ids[0])
        return seqs, flat

    def _generate_with_temp(_model, _tokenizer, _prompt: str, _beams: int, _nrs: int, _temp: float, _device: str,
                             bad_words_ids: Optional[List[List[int]]] = None,
                             penalty_tokens: Optional[Set[int]] = None,
                             penalty_val: float = 0.0):
        inputs = _tokenizer(_prompt, return_tensors='pt')
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        processors = _build_processors(_temp, penalty_tokens=penalty_tokens, penalty_val=penalty_val)
        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_length=150,
                num_beams=_beams,
                num_return_sequences=_nrs,
                do_sample=False,
                early_stopping=True,
                logits_processor=processors,
                bad_words_ids=bad_words_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )
        # decode
        decoded = _tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        # compute first-step normalized entropy if available
        first_scores = None
        norm_entropy = None
        try:
            if out.scores and len(out.scores) > 0:
                first_scores = out.scores[0]
                # first_scores shape: (batch*beam, vocab)
                p = torch.softmax(first_scores, dim=-1)
                # entropy per row
                H = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1)
                # normalize by log(V)
                V = p.size(-1)
                norm_entropy = float((H / math.log(V)).mean().item())
        except Exception:
            norm_entropy = None
        return decoded, norm_entropy

    # diagnostics accumulators
    total_users = 0
    forget_users = 0
    routed_side_count = 0
    fallback_count = 0
    entropy_vals_forget: List[float] = []
    unique_ratio_vals_forget: List[float] = []

    for uid in user_ids:
        forget_items = forget_requests.get(uid, [])
        prompt = _prompt_for_user(uid, train_df, forget_items if use_side_for_forget else None)
        # 规范化 beam/返回序列，避免 HF 约束错误：num_return_sequences <= num_beams
        safe_beams = max(1, int(num_beams))
        safe_nrs = max(1, int(num_return_sequences))
        if safe_beams < safe_nrs:
            safe_beams = safe_nrs

        # Dual runtime path
        total_users += 1
        is_forget_user = False
        if is_dual:
            # route: forget -> side; else -> main
            is_forget_user = (use_side_for_forget and uid in forget_requests)
            if is_forget_user:
                forget_users += 1
            model_like.adapter.set_mode('side' if is_forget_user else 'main')
            # generate with temperature
            temp = side_temperature if is_forget_user else base_temperature
            bad_words_ids = None
            penalty_tokens = None
            penalty_val = 0.0
            if is_forget_user and bool(globals().get('_CERTIFIED_FORGETFUL_DECODING_ENABLED', False)):
                # this global knob will be set from outer scope via closure below
                try:
                    bad_words_ids, penalty_tokens = _encode_bad_words(model_like.tokenizer, list(map(str, forget_items)))
                    penalty_val = float(globals().get('_FORBIDDEN_PENALTY_VALUE', 0.0))
                except Exception:
                    bad_words_ids, penalty_tokens, penalty_val = None, None, 0.0
            decoded, norm_ent = _generate_with_temp(
                model_like.model,
                model_like.tokenizer,
                prompt,
                safe_beams,
                safe_nrs,
                temp,
                str(model_like.device),
                bad_words_ids=bad_words_ids,
                penalty_tokens=penalty_tokens,
                penalty_val=penalty_val,
            )
            # entropy-based fallback for forget users routed to side
            trigger_fallback = False
            if is_forget_user and bool(use_entropy_fallback):
                if norm_ent is not None and conf_fallback_threshold is not None:
                    if float(norm_ent) > float(conf_fallback_threshold):
                        trigger_fallback = True
            if is_forget_user:
                if norm_ent is not None:
                    entropy_vals_forget.append(float(norm_ent))
        else:
            # raw T5 model
            decoded, norm_ent = _generate_with_temp(
                model_like,
                tokenizer,
                prompt,
                safe_beams,
                safe_nrs,
                base_temperature,
                device,
            )

        seen = set()
        items: List[str] = []
        for t in decoded:
            found = item_pattern.findall(t)
            for it in found:
                if it not in seen:
                    items.append(it)
                    seen.add(it)
                if len(items) >= k:
                    break
            if len(items) >= k:
                break

        # additional uniqueness fallback check
        if is_dual and is_forget_user:
            unique_ratio = (len(items) / float(k)) if k > 0 else 0.0
            if unique_ratio < float(min_unique_ratio):
                trigger_fallback = True
            unique_ratio_vals_forget.append(float(unique_ratio))

        if is_dual and is_forget_user:
            routed_side_count += 1

        # reroute to main if triggered
        if is_dual and 'trigger_fallback' in locals() and trigger_fallback:
            model_like.adapter.set_mode('main')
            decoded_fb, _ = _generate_with_temp(
                model_like.model,
                model_like.tokenizer,
                prompt,
                safe_beams,
                safe_nrs,
                base_temperature,
                str(model_like.device),
            )
            # rebuild items from fallback decoded
            seen = set()
            items = []
            for t in decoded_fb:
                found = item_pattern.findall(t)
                for it in found:
                    if it not in seen:
                        items.append(it)
                        seen.add(it)
                    if len(items) >= k:
                        break
                if len(items) >= k:
                    break
            fallback_count += 1

        if len(items) < k and not disable_fallback:
            # optionally pad with popular items from train_df
            popular = list(train_df['mapped_item_id'].value_counts().index.astype(str))
            for p in popular:
                if p not in seen:
                    items.append(p)
                    seen.add(p)
                if len(items) >= k:
                    break
        # Certified check: ensure no forbidden item appears, record certificate
        if is_dual and is_forget_user and bool(globals().get('_EMIT_CERTIFICATES', False)):
            try:
                forb_set = set(map(str, forget_items))
                inter = [x for x in items if x in forb_set]
                import os as _os, json as _json, time as _time
                _os.makedirs('results/certificates', exist_ok=True)
                cert = {
                    'time': int(_time.time()),
                    'user_id': uid,
                    'routed': 'side',
                    'forbidden_items': list(forb_set),
                    'topk_items': items[:k],
                    'violation_count': len(inter),
                    'violations': inter,
                }
                with open(_os.path.join('results/certificates', f'cert_{uid}_{int(_time.time())}.json'), 'w', encoding='utf-8') as f:
                    _json.dump(cert, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        recs[uid] = items
    # reset adapter if dual
    if is_dual:
        model_like.adapter.set_mode('main')
    diag = {
        'total_users': float(total_users),
        'forget_users': float(forget_users),
        'routed_side_count': float(routed_side_count),
        'fallback_count': float(fallback_count),
        'avg_norm_entropy_forget': float(np.mean(entropy_vals_forget)) if entropy_vals_forget else float('nan'),
        'avg_unique_ratio_forget': float(np.mean(unique_ratio_vals_forget)) if unique_ratio_vals_forget else float('nan'),
    }
    return recs, diag

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MIA for P5/T5 dual-memory on ML-1M")
    parser.add_argument('--original_model', type=str, required=True, help='Path to base model checkpoint (e.g., models/.../model.safetensors)')
    parser.add_argument('--unlearned_artifacts', type=str, required=True, help='Path to dual_memory_artifacts.pt')
    parser.add_argument('--gold_model', type=str, default=None, help='Optional retrained model checkpoint')
    parser.add_argument('--k', type=int, default=50, help='Top-K for recommendations and features')
    parser.add_argument('--eval_users', type=int, default=800, help='Number of users per set to evaluate (retain/test/forget sample sizes)')
    parser.add_argument('--num_beams', type=int, default=20)
    parser.add_argument('--num_return_sequences', type=int, default=20)
    parser.add_argument('--disable_fallback', action='store_true', help='Disable popularity fallback padding')
    parser.add_argument('--base_temperature', type=float, default=1.0, help='Temperature for base/main memory generation (beam search logit scaling)')
    parser.add_argument('--side_temperature', type=float, default=1.2, help='Temperature for side memory generation (soften overconfident logits)')
    parser.add_argument('--use_entropy_fallback', action='store_true', help='Enable entropy-based confidence fallback to base for side-generated outputs')
    parser.add_argument('--conf_fallback_threshold', type=float, default=0.85, help='Normalized entropy threshold for triggering fallback (0-1, higher means more uncertain)')
    parser.add_argument('--min_unique_ratio', type=float, default=0.3, help='If unique recommended items / K below this, trigger fallback to base for forget users')
    parser.add_argument('--certified_forgetful_decoding', action='store_true', help='Enable certified decoding: block forgotten items during generation and emit certificates')
    parser.add_argument('--forbidden_penalty', type=float, default=20.0, help='Additional logit penalty for forbidden tokens (complements bad_words_ids)')
    parser.add_argument('--emit_certificates', action='store_true', help='Write per-user certificates verifying no forbidden items appear in Top-K')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducible sampling and training splits')
    args = parser.parse_args()

    # Set deterministic seeds for reproducibility across sampling and training
    try:
        import numpy as _np
        random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    print("正在加载 ML-1M 数据...")
    train_df, test_df, maps = load_ml1m_splits()
    forget_requests = load_forget_requests()
    retain_interactions, test_interactions = build_interaction_sets(train_df, test_df)
    forget_interactions = build_forget_interactions_from_requests(forget_requests)

    # 采样用户集
    forget_users_all = list(forget_interactions.keys())
    if len(forget_users_all) == 0:
        raise RuntimeError("未发现遗忘请求，请先运行训练脚本生成 forget_samples_subset.json")
    eval_size = min(args.eval_users, len(forget_users_all))
    eval_forget_users = random.sample(forget_users_all, eval_size)
    retain_users_all = [u for u in retain_interactions.keys() if u not in set(eval_forget_users)]
    test_users_all = list(test_interactions.keys())
    eval_retain_users = random.sample(retain_users_all, min(eval_size, len(retain_users_all)))
    eval_test_users = random.sample(test_users_all, min(eval_size, len(test_users_all)))

    print(f"评估用户数: forget={len(eval_forget_users)} retain={len(eval_retain_users)} test={len(eval_test_users)}")

    # 加载模型
    print("加载原始模型...")
    base_wrapper = P5ModelWrapper(args.original_model, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    base_model = base_wrapper.get_model()
    base_tok = base_wrapper.get_tokenizer()

    print("加载遗忘工件...")
    dual_runtime = load_dual_memory_runtime(args.unlearned_artifacts, device=('cuda' if torch.cuda.is_available() else 'cpu'))

    gold_model = None
    gold_tok = None
    if args.gold_model:
        print("加载重训模型 (gold)...")
        gold_wrapper = P5ModelWrapper(args.gold_model, device=('cuda' if torch.cuda.is_available() else 'cpu'))
        gold_model = gold_wrapper.get_model()
        gold_tok = gold_wrapper.get_tokenizer()

    # 生成推荐
    print("生成基础模型推荐...")
    users_for_base = list(set(eval_forget_users + eval_retain_users + eval_test_users))
    # Set global knobs for the local generation helpers
    globals()['_CERTIFIED_FORGETFUL_DECODING_ENABLED'] = bool(args.certified_forgetful_decoding)
    globals()['_FORBIDDEN_PENALTY_VALUE'] = float(args.forbidden_penalty)
    globals()['_EMIT_CERTIFICATES'] = bool(args.emit_certificates)
    base_recs, base_diag = generate_recommendations_for_users(
        model_like=base_model,
        tokenizer=base_tok,
        user_ids=users_for_base,
        train_df=train_df,
        forget_requests=forget_requests,
        use_side_for_forget=False,
        k=args.k,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        disable_fallback=args.disable_fallback,
        device=base_wrapper.device,
        base_temperature=args.base_temperature,
        side_temperature=args.side_temperature,
        use_entropy_fallback=args.use_entropy_fallback,
        conf_fallback_threshold=args.conf_fallback_threshold,
        min_unique_ratio=args.min_unique_ratio,
    )

    print("生成遗忘后模型推荐 (遗忘用户走侧记忆)...")
    users_for_unl = users_for_base
    unlearned_recs, unl_diag = generate_recommendations_for_users(
        model_like=dual_runtime,
        tokenizer=dual_runtime.tokenizer,
        user_ids=users_for_unl,
        train_df=train_df,
        forget_requests=forget_requests,
        use_side_for_forget=True,
        k=args.k,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        disable_fallback=args.disable_fallback,
        device=str(dual_runtime.device),
        base_temperature=args.base_temperature,
        side_temperature=args.side_temperature,
        use_entropy_fallback=args.use_entropy_fallback,
        conf_fallback_threshold=args.conf_fallback_threshold,
        min_unique_ratio=args.min_unique_ratio,
    )

    gold_recs = None
    if gold_model is not None:
        print("生成重训模型推荐...")
        users_for_gold = users_for_base
        gold_recs, gold_diag = generate_recommendations_for_users(
            model_like=gold_model,
            tokenizer=gold_tok,
            user_ids=users_for_gold,
            train_df=train_df,
            forget_requests=forget_requests,
            use_side_for_forget=False,
            k=args.k,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            disable_fallback=args.disable_fallback,
            device=gold_wrapper.device,
            base_temperature=args.base_temperature,
            side_temperature=args.side_temperature,
            use_entropy_fallback=args.use_entropy_fallback,
            conf_fallback_threshold=args.conf_fallback_threshold,
            min_unique_ratio=args.min_unique_ratio,
        )

    # 准备攻击者训练数据（基于基础模型）
    print("准备攻击者训练数据...")
    X, y = prepare_training_data(
        retain_interactions={u: retain_interactions[u] for u in eval_retain_users if u in retain_interactions},
        test_interactions={u: test_interactions[u] for u in eval_test_users if u in test_interactions},
        base_recommendations=base_recs,
        forget_requests=forget_requests,
        k=args.k,
    )
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")
    feature_names = [
        "命中数量", "历史DCG", "首个命中项倒数排名", "命中项排名标准差", "最高命中排名",
        "平均分数", "分数方差", "命中项平均分数", "命中项分数标准差", "NDCG@K(历史)", "NDCG@K(遗忘目标)"
    ]
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=100.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=100.0, neginf=0.0)

    print("训练攻击模型...")
    attack_model, scaler, r = train_attack_model(X_train, y_train)
    val_acc, val_auc = evaluate_model(attack_model, scaler, X_val, y_val)
    print(f"攻击模型在验证集上的性能: ACC={val_acc:.4f}, AUC={val_auc:.4f}")

    # 构造 encoded_* 交互子集供评估函数使用
    encoded_retain = {u: retain_interactions[u] for u in eval_retain_users if u in retain_interactions}
    encoded_test = {u: test_interactions[u] for u in eval_test_users if u in test_interactions}
    encoded_forget = {u: forget_interactions[u] for u in eval_forget_users if u in forget_interactions}

    print("评估遗忘效果 (MIA 攻击角度)...")
    forget_results = evaluate_forget_effectiveness(
        attack_model, scaler,
        forget_interactions=encoded_forget,
        recommendations_mbase=base_recs,
        recommendations_munlearned=unlearned_recs,
        recommendations_mgold=gold_recs,
        encoded_forget_interactions=encoded_forget,
        encoded_test_interactions=encoded_test,
        encoded_retain_interactions=encoded_retain,
        forget_requests=forget_requests,
        k=args.k,
    )

    print("\n遗忘效果评估结果 (MIA):")
    print("=" * 50)
    for model_name, metrics in forget_results.items():
        print(f"{model_name}:")
        print(f"  ACC: {metrics['ACC']:.4f}")
        print(f"  AUC: {metrics['AUC']:.4f}")
        print(f"  平均预测概率: {metrics['pred_mean']:.4f}")
        accd = metrics.get('acc_detail') or {}
        if accd:
            print(f"  [acc_detail] pos_rate={accd.get('pos_rate'):.3f}, acc@0.5={accd.get('acc@0.5'):.3f}, bal_acc@0.5={accd.get('bal_acc@0.5'):.3f}, best_acc={accd.get('best_acc'):.3f} @ thr={accd.get('best_acc_threshold'):.3g}")
            print(f"               youdenJ_acc={accd.get('youdenJ_acc'):.3f} @ thr={accd.get('youdenJ_threshold'):.3g}, auc={accd.get('auc'):.3f}, confusion@0.5={accd.get('confusion@0.5')}")
        print()

    # 附加：直接基于 NDCG 的遗忘强度展示（非攻击模型）
    def avg_ndcg_forget(rec_map: Dict[str, List[str]], users: List[str]) -> float:
        vals = []
        for u in users:
            recs = rec_map.get(u, [])
            fset = set(forget_requests.get(u, []))
            ranks = []
            for i, it in enumerate(recs[:args.k]):
                if it in fset:
                    ranks.append(i + 1)
            vals.append(_ndcg_from_hits(ranks, args.k))
        return float(np.mean(vals)) if vals else 0.0

    ndcg_base = avg_ndcg_forget(base_recs, eval_forget_users)
    ndcg_unl = avg_ndcg_forget(unlearned_recs, eval_forget_users)
    ndcg_gold = avg_ndcg_forget(gold_recs, eval_forget_users) if gold_recs is not None else None
    print("直接NDCG@K(遗忘目标集) 对比:")
    print(f"  Base:      {ndcg_base:.4f}")
    print(f"  Unlearned: {ndcg_unl:.4f}  (↓ {ndcg_base - ndcg_unl:+.4f})")
    if ndcg_gold is not None:
        print(f"  Gold:      {ndcg_gold:.4f}  (Base- Gold: {ndcg_base - ndcg_gold:+.4f})")

    # 保存结果
    ts = int(time.time())
    out = {
        'time': ts,
        'k': args.k,
        'eval_users': args.eval_users,
        'attack_val': {'acc': float(val_acc), 'auc': float(val_auc)},
        'mia_results': forget_results,
        'ndcg_forgotten': {
            'base': float(ndcg_base),
            'unlearned': float(ndcg_unl),
            'gold': float(ndcg_gold) if ndcg_gold is not None else None,
        },
        'diagnostics': {
            'base': base_diag,
            'unlearned': unl_diag,
            'gold': gold_diag if gold_recs is not None else None,
        },
        'config': {
            'original_model': args.original_model,
            'unlearned_artifacts': args.unlearned_artifacts,
            'gold_model': args.gold_model,
            'num_beams': args.num_beams,
            'num_return_sequences': args.num_return_sequences,
            'disable_fallback': bool(args.disable_fallback),
            'base_temperature': float(args.base_temperature),
            'side_temperature': float(args.side_temperature),
            'use_entropy_fallback': bool(args.use_entropy_fallback),
            'conf_fallback_threshold': float(args.conf_fallback_threshold),
            'min_unique_ratio': float(args.min_unique_ratio),
            'seed': int(args.seed),
        }
    }
    os.makedirs('results', exist_ok=True)
    out_path = os.path.join('results', f'mia_evaluation_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"📄 MIA评估摘要已保存: {out_path}")

if __name__ == "__main__":
    main()