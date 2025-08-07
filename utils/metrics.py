"""评估指标计算"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def calculate_ic(predictions, targets, method='spearman'):
    """
    计算信息系数(IC)

    Args:
        predictions: 预测值
        targets: 真实值
        method: 'spearman' 或 'pearson'

    Returns:
        IC值
    """
    # 处理数据
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(targets, 'values'):
        targets = targets.values

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # 对齐长度
    min_len = min(len(predictions), len(targets))
    predictions = predictions[:min_len]
    targets = targets[:min_len]

    # 移除NaN
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    if valid_mask.sum() < 2:
        return 0.0

    # 计算相关系数
    if method == 'spearman':
        corr, _ = spearmanr(predictions[valid_mask], targets[valid_mask])
    else:
        corr, _ = pearsonr(predictions[valid_mask], targets[valid_mask])

    return corr if not np.isnan(corr) else 0.0


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods=252):
    """
    计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        periods: 年化因子（日频数据为252）

    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate / periods

    if len(excess_returns) < 2:
        return 0.0

    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        return 0.0

    sharpe_ratio = np.sqrt(periods) * mean_excess_return / std_excess_return

    return sharpe_ratio


def calculate_max_drawdown(cumulative_returns):
    """
    计算最大回撤

    Args:
        cumulative_returns: 累计收益率序列

    Returns:
        最大回撤
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)

    return abs(max_drawdown)