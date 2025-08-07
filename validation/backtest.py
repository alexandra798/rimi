"""回测模块 - 修复版"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging


from alpha.evaluator import FormulaEvaluator

logger = logging.getLogger(__name__)


def backtest_formulas(formulas, X_test, y_test):
    """
    回测已发现的公式

    Parameters:
    - formulas: 要测试的公式列表
    - X_test: 测试特征数据
    - y_test: 测试目标数据

    Returns:
    - results: 公式及其IC值的字典
    """
    evaluator = FormulaEvaluator()
    results = {}

    for formula in formulas:
        # 使用统一的评估函数
        feature = evaluator.evaluate(formula, X_test)

        # 对齐数据
        valid_indices = ~(feature.isna() | y_test.isna())
        feature_clean = feature[valid_indices]
        y_test_clean = y_test[valid_indices]

        # 计算IC
        if len(feature_clean) > 1:
            ic, _ = spearmanr(feature_clean, y_test_clean)
            results[formula] = ic if not np.isnan(ic) else 0
        else:
            results[formula] = 0
            logger.warning(f"Insufficient data for formula: {formula}")

    return results