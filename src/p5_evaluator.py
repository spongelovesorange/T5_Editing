"""
P5推荐系统官方评估器
基于OpenP5官方评估方法，支持标准的推荐系统指标
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class P5RecommendationEvaluator:
    """P5推荐系统评估器 - 基于官方评估方法"""
    
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_recommendations(self, 
                               predictions: List[Dict],
                               ground_truth: List[Dict],
                               metrics: List[str] = ['hit@5', 'hit@10', 'hit@20', 'ndcg@5', 'ndcg@10', 'ndcg@20'],
                               filtered: bool = True) -> Dict[str, float]:
        """
        评估推荐性能
        
        Args:
            predictions: 预测结果列表，每个元素包含user_id和recommended_items
            ground_truth: 真实标签列表，每个元素包含user_id和item_id
            metrics: 评估指标列表
            filtered: 是否过滤训练集中的物品
        
        Returns:
            评估结果字典
        """
        logger.info(f"开始P5推荐系统评估，指标: {metrics}")
        
        # 构建用户-物品映射
        user_predictions = self._build_user_predictions(predictions)
        user_ground_truth = self._build_user_ground_truth(ground_truth)
        
        # 计算相关性结果（用于 hit 和 ndcg）
        rel_results = self._compute_relevance_results(
            user_predictions, user_ground_truth, max([self._extract_k(m) for m in metrics])
        )
        
        # 计算各项指标
        results = {}
        for metric in metrics:
            if metric.lower().startswith('hit'):
                k = self._extract_k(metric)
                results[f'hit@{k}'] = self._hit_at_k(rel_results, k)
            elif metric.lower().startswith('ndcg'):
                k = self._extract_k(metric)
                results[f'ndcg@{k}'] = self._ndcg_at_k(rel_results, k)
            elif metric.lower().startswith('recall'):
                k = self._extract_k(metric)
                # 修正：基于 ground truth 大小计算 Recall@K
                results[f'recall@{k}'] = self._recall_at_k_from_maps(user_predictions, user_ground_truth, k)
        
        # 计算用户数量用于标准化
        total_users = len(user_ground_truth)
        
        # 整理结果（无需二次归一化，因为各指标计算函数已经正确归一化）
        normalized_results = {}
        for metric_name, value in results.items():
            normalized_results[metric_name] = value
        
        # 添加HR指标（等同于Hit Rate）
        if 'hit@10' in normalized_results:
            normalized_results['hit_rate@10'] = normalized_results['hit@10']
        if 'hit@20' in normalized_results:
            normalized_results['hit_rate@20'] = normalized_results['hit@20']
        
        logger.info("P5评估完成")
        self._log_metrics(normalized_results)
        
        return normalized_results
    
    def _build_user_predictions(self, predictions: List[Dict]) -> Dict[str, List[str]]:
        """构建用户预测映射"""
        user_preds = defaultdict(list)
        
        for pred in predictions:
            user_id = str(pred.get('user_id', ''))
            items = pred.get('recommended_items', [])
            
            # 确保items是列表
            if isinstance(items, str):
                items = [items]
            elif isinstance(items, (int, float)):
                items = [str(items)]
            
            # 转换为字符串并去重
            item_strs = []
            for item in items:
                item_str = str(item)
                if item_str not in item_strs:
                    item_strs.append(item_str)
            
            user_preds[user_id].extend(item_strs)
        
        return user_preds
    
    def _build_user_ground_truth(self, ground_truth: List[Dict]) -> Dict[str, List[str]]:
        """构建用户真实标签映射"""
        user_truth = defaultdict(list)
        
        for truth in ground_truth:
            user_id = str(truth.get('user_id', ''))
            item_id = str(truth.get('item_id', ''))
            
            if item_id and item_id not in user_truth[user_id]:
                user_truth[user_id].append(item_id)
        
        return user_truth
    
    def _compute_relevance_results(self, 
                                  user_predictions: Dict[str, List[str]],
                                  user_ground_truth: Dict[str, List[str]],
                                  max_k: int) -> List[List[int]]:
        """计算相关性结果（按照P5官方方法）"""
        rel_results = []
        
        for user_id in user_ground_truth:
            if user_id not in user_predictions:
                # 如果没有预测，填充0
                rel_results.append([0] * max_k)
                continue
            
            pred_items = user_predictions[user_id][:max_k]
            true_items = set(user_ground_truth[user_id])
            
            # 计算每个位置的相关性
            user_rel = []
            for item in pred_items:
                if item in true_items:
                    user_rel.append(1)
                else:
                    user_rel.append(0)
            
            # 填充到max_k长度
            while len(user_rel) < max_k:
                user_rel.append(0)
            
            rel_results.append(user_rel)
        
        return rel_results
    
    def _extract_k(self, metric: str) -> int:
        """从指标名称中提取k值"""
        try:
            return int(metric.split('@')[1])
        except (IndexError, ValueError):
            return 10  # 默认值
    
    def _hit_at_k(self, relevance: List[List[int]], k: int) -> float:
        """
        计算Hit@K（正确的归一化版本）
        Hit@K = 有推荐命中的用户数 / 总用户数
        """
        if not relevance:
            return 0.0
            
        correct = 0.0
        total_users = len(relevance)
        
        for row in relevance:
            rel = row[:k]
            if sum(rel) > 0:  # 如果在前K个推荐中有命中
                correct += 1
                
        return correct / total_users if total_users > 0 else 0.0
    
    def _ndcg_at_k(self, relevance: List[List[int]], k: int) -> float:
        """
        计算NDCG@K（修复版本）
        NDCG@K = DCG@K / IDCG@K
        其中IDCG@K是理想情况下的DCG@K（假设前K个推荐都是最相关的）
        """
        if not relevance:
            return 0.0
            
        total_ndcg = 0.0
        num_users = 0
        
        for row in relevance:
            # 计算DCG@K
            dcg = 0.0
            for i in range(min(k, len(row))):
                if row[i] > 0:  # 只有相关物品才计算
                    dcg += row[i] / math.log(i + 2, 2)
            
            # 计算IDCG@K（理想情况下的DCG@K）
            # 对于推荐系统，通常假设每个用户的相关物品评分都是1
            # IDCG@K = 理想情况下前K个位置都是相关物品(评分=1)的DCG
            num_relevant_items = sum(1 for score in row if score > 0)  # 用户实际的相关物品数
            ideal_relevant_at_k = min(k, num_relevant_items)  # 理想情况下前K个位置中的相关物品数
            
            idcg = 0.0
            for i in range(ideal_relevant_at_k):
                idcg += 1.0 / math.log(i + 2, 2)  # 假设理想情况下相关物品评分为1
            
            # 计算NDCG = DCG / IDCG
            if idcg > 0:
                total_ndcg += dcg / idcg
                num_users += 1
        
        return total_ndcg / num_users if num_users > 0 else 0.0
    
    def _recall_at_k_from_maps(self,
                               user_predictions: Dict[str, List[str]],
                               user_ground_truth: Dict[str, List[str]],
                               k: int) -> float:
        """
        计算 Recall@K：sum_users |Pred@K ∩ GT| / sum_users |GT|
        说明：必须使用 ground truth 的真实数量作为分母，而不是相关性行的和。
        """
        total_hits = 0
        total_relevant = 0
        for uid, gt_items in user_ground_truth.items():
            gt_set = set(gt_items)
            if not gt_set:
                continue
            preds = user_predictions.get(uid, [])[:k]
            hits = sum(1 for it in preds if it in gt_set)
            total_hits += hits
            total_relevant += len(gt_set)
        if total_relevant == 0:
            return 0.0
        return total_hits / total_relevant
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """记录评估指标"""
        logger.info("📊 P5评估指标:")
        for metric_name, value in sorted(metrics.items()):
            logger.info(f"  {metric_name}: {value:.4f}")
    
    def evaluate_unlearning_effectiveness(self, 
                                        before_metrics: Dict[str, float],
                                        after_metrics: Dict[str, float],
                                        target_metrics: List[str] = None) -> Dict[str, Any]:
        """
        评估遗忘效果
        
        Args:
            before_metrics: 遗忘前的指标
            after_metrics: 遗忘后的指标
            target_metrics: 目标指标列表
        
        Returns:
            遗忘效果分析结果
        """
        if target_metrics is None:
            target_metrics = ['hit@10', 'hit@20', 'ndcg@10', 'ndcg@20', 'recall@10', 'recall@20']
        
        effectiveness = {}
        
        for metric in target_metrics:
            # 检查指标是否存在
            before_value = before_metrics.get(metric, 0.0)
            after_value = after_metrics.get(metric, 0.0)
            
            # 计算变化
            absolute_change = after_value - before_value
            relative_change = (absolute_change / before_value * 100) if before_value > 0 else 0.0
            
            effectiveness[metric] = {
                'before': before_value,
                'after': after_value,
                'absolute_change': absolute_change,
                'relative_change': relative_change
            }
        
        return effectiveness
    
    def compute_unlearning_score(self, 
                                retain_effectiveness: Dict[str, Any],
                                unlearn_effectiveness: Dict[str, Any],
                                target_metrics: List[str] = None) -> float:
        """
        计算遗忘效果综合分数
        
        理想情况：
        - 保留集性能变化尽量小（接近0）
        - 遗忘集性能显著下降（负值）
        
        分数计算：遗忘集性能下降程度 - 保留集性能变化程度
        """
        if target_metrics is None:
            target_metrics = ['hit@10', 'hit@20', 'ndcg@10', 'ndcg@20']
        
        total_score = 0.0
        valid_metrics = 0
        
        for metric in target_metrics:
            if metric in retain_effectiveness and metric in unlearn_effectiveness:
                # 保留集相对变化的绝对值（越小越好）
                retain_change = abs(retain_effectiveness[metric]['relative_change'])
                
                # 遗忘集相对变化的绝对值（越大越好，因为期望性能下降）
                unlearn_change = abs(unlearn_effectiveness[metric]['relative_change'])
                
                # 计算单个指标的遗忘效果分数
                # 如果遗忘集性能下降且保留集性能保持稳定，则分数为正
                metric_score = unlearn_change - retain_change
                total_score += metric_score
                valid_metrics += 1
        
        return total_score / valid_metrics if valid_metrics > 0 else 0.0
    
    def generate_comparison_report(self, 
                                    before_retain: Dict[str, float],
                                    after_retain: Dict[str, float],
                                    before_unlearn: Dict[str, float],
                                    after_unlearn: Dict[str, float]) -> str:
            """[CIU 升级版] 生成详细的交互级遗忘对比报告"""
            
            report_lines = []
            report_lines.append("=" * 110)
            report_lines.append("P5模型 [交互级遗忘] 效果详细报告 (CIU Scheme)")
            report_lines.append("=" * 110)
            report_lines.append("说明: '保留效用'衡量对用户其他偏好的推荐能力(越高越好)，'遗忘效能'衡量对特定物品的遗忘程度(越低越好)。")
            
            # 表头
            header = f"{'指标':^25} | {'遗忘前':^15} | {'遗忘后':^15} | {'变化(%)':^12} || {'遗忘前':^15} | {'遗忘后':^15} | {'变化(%)':^12}"
            subheader = f"{' ':^25} | {'--- 保留集 (保留效用) ---':^46} || {'--- 遗忘集 (保留效用) ---':^46}"
            report_lines.append(subheader)
            report_lines.append(header)
            report_lines.append("-" * 110)
            
            # 关键指标
            key_metrics_retain = ['hit@10_retain', 'ndcg@10_retain', 'recall@10_retain']
            
            for metric in key_metrics_retain:
                # 保留集
                br, ar = before_retain.get(metric, 0), after_retain.get(metric, 0)
                retain_change = ((ar - br) / br * 100) if br > 1e-6 else 0
                
                # 遗忘集
                bu, au = before_unlearn.get(metric, 0), after_unlearn.get(metric, 0)
                unlearn_change_retain = ((au - bu) / bu * 100) if bu > 1e-6 else 0
                
                report_lines.append(
                    f"{metric:^25} | {br:^15.4f} | {ar:^15.4f} | {retain_change:^12.1f} || {bu:^15.4f} | {au:^15.4f} | {unlearn_change_retain:^12.1f}"
                )
            
            report_lines.append("-" * 110)
            
            # 遗忘效能的单独报告
            header_forget = f"{'指标':^25} | {'遗忘前 (召回率)':^20} | {'遗忘后 (召回率)':^20} | {'下降率 (%)':^15}"
            subheader_forget = f"{' ':^25} | {'--- 遗忘集 (遗忘效能) ---':^60}"
            report_lines.append("\n" + subheader_forget)
            report_lines.append(header_forget)
            report_lines.append("-" * 85)
            
            key_metrics_forget = ['recall@10_forgotten', 'recall@20_forgotten']
            avg_forget_reduction = []

            for metric in key_metrics_forget:
                if metric in before_unlearn and metric in after_unlearn:
                    bu_f, au_f = before_unlearn.get(metric, 0), after_unlearn.get(metric, 0)
                    forget_reduction = ((bu_f - au_f) / bu_f * 100) if bu_f > 1e-6 else 0
                    avg_forget_reduction.append(forget_reduction)
                    report_lines.append(f"{metric:^25} | {bu_f:^20.4f} | {au_f:^20.4f} | {forget_reduction:^15.1f}")
            
            report_lines.append("-" * 85)
            
            # 计算综合评估
            retain_changes = [abs(((after_retain.get(m,0)-before_retain.get(m,0))/before_retain.get(m,1e-6)*100)) for m in key_metrics_retain if m in before_retain]
            forget_utility_changes = [abs(((after_unlearn.get(m,0)-before_unlearn.get(m,0))/before_unlearn.get(m,1e-6)*100)) for m in key_metrics_retain if m in before_unlearn]
            
            avg_retain_perf_change = np.mean(retain_changes) if retain_changes else 0
            avg_forget_utility_change = np.mean(forget_utility_changes) if forget_utility_changes else 0
            avg_forget_efficacy = np.mean(avg_forget_reduction) if avg_forget_reduction else 0

            report_lines.append(f"\n🎯 遗忘效果综合评估:")
            report_lines.append(f"  - 保留集性能稳定性: 平均变化 {avg_retain_perf_change:.2f}% (越小越好)")
            report_lines.append(f"  - 遗忘集通用性能影响: 平均变化 {avg_forget_utility_change:.2f}% (越小越好)")
            report_lines.append(f"  - 遗忘集特定项遗忘率: 平均下降 {avg_forget_efficacy:.2f}% (越大越好)")
            
            score = avg_forget_efficacy - (avg_retain_perf_change + avg_forget_utility_change) / 2
            report_lines.append(f"  - 综合分数 (遗忘率 - 平均性能影响): {score:.2f}")

            if score > 50:
                report_lines.append("\n✅ 遗忘效果优秀！模型精准地遗忘了特定交互，同时几乎没有影响其他推荐的质量。")
            elif score > 20:
                report_lines.append("\n👍 遗忘效果良好，但对通用推荐性能有轻微影响。")
            else:
                report_lines.append("\n⚠️ 遗忘效果一般或不佳，请检查超参数或训练轮数。")

            report_lines.append("=" * 110)
            
            return "\n".join(report_lines)
