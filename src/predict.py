"""
预测与提交文件生成模块（学习版）

功能：
1. 加载训练好的 sklearn 模型（joblib）
2. 读取测试集特征并按训练列对齐
3. 执行预测（支持训练时使用 log1p 目标）
4. 生成 Kaggle 提交文件 submission.csv

使用方法：
    python -m src.predict
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    MODEL_FILEPATH,
    SUBMISSION_FILEPATH,
    TEST_FILEPATH,
    TEST_PROCESSED_FILEPATH,
    TRAIN_REPORT_FILEPATH,
)

# =============================================================================
# 第一阶段：加载模型与元信息
# =============================================================================


def loadModel(modelPath: str = MODEL_FILEPATH) -> Any:
    """加载训练好的 sklearn 模型对象（可为 Pipeline/Ensemble）。"""
    if not os.path.exists(modelPath):
        raise FileNotFoundError(
            f"未找到模型文件: {modelPath}，请先运行 `python -m src.train`"
        )
    model = load(modelPath)
    return model


def loadTrainReport(reportPath: str = TRAIN_REPORT_FILEPATH) -> dict[str, Any]:
    """加载训练报告，读取训练时特征列和参数（如 useLogTarget）。"""
    if not os.path.exists(reportPath):
        raise FileNotFoundError(
            f"未找到训练报告: {reportPath}，请先运行 `python -m src.train`"
        )
    with open(reportPath, encoding="utf-8") as f:
        report = json.load(f)
    return report


# =============================================================================
# 第二阶段：读取并对齐测试集
# =============================================================================


def loadTestFeatures() -> pd.DataFrame:
    """
    读取处理后的测试集特征。

    注意：
        - 使用的是特征工程后的 TEST_PROCESSED_FILEPATH
        - 不应包含目标列 SalePrice
    """
    if not os.path.exists(TEST_PROCESSED_FILEPATH):
        raise FileNotFoundError(
            f"未找到处理后测试集: {TEST_PROCESSED_FILEPATH}，请先运行 `python -m src.featureEngineering`"
        )

    xTestDf = pd.read_csv(TEST_PROCESSED_FILEPATH)
    if "SalePrice" in xTestDf.columns:
        xTestDf = xTestDf.drop(columns=["SalePrice"], errors="ignore")
    return xTestDf


def alignTestColumns(xTestDf: pd.DataFrame, featureCols: list[str]) -> pd.DataFrame:
    """
    将测试集列对齐到训练列。

    原因：
        - one-hot 后 train/test 列可能不同
        - 训练报告里保存了训练时实际使用的 featureCols
    """
    aligned = xTestDf.reindex(columns=featureCols, fill_value=0)
    return aligned


def loadSubmissionIds() -> pd.Series:
    """
    从原始测试集读取 Id 列，用于生成提交文件。

    说明：
        - 处理后的测试集通常已经去掉 Id
        - Kaggle 提交必须包含 Id
    """
    if not os.path.exists(TEST_FILEPATH):
        raise FileNotFoundError(f"未找到原始测试集: {TEST_FILEPATH}")

    testRaw = pd.read_csv(TEST_FILEPATH)
    if "Id" not in testRaw.columns:
        raise ValueError("原始测试集缺少 Id 列，无法生成 submission")
    return testRaw["Id"]


# =============================================================================
# 第三阶段：预测与提交
# =============================================================================


def predictSalePrice(
    model: Any,
    xTestAligned: pd.DataFrame,
    useLogTarget: bool = True,
) -> np.ndarray:
    """
    进行房价预测。

    若训练时 useLogTarget=True：
        - 训练标签是 log1p(y)
        - 预测后需要 expm1 还原到原始价格空间
    """
    pred = model.predict(xTestAligned.values.astype(np.float64))

    if useLogTarget:
        pred = np.expm1(pred)

    # 房价不能为负，做安全截断
    pred = np.clip(pred, a_min=0, a_max=None)
    return pred


def buildSubmission(ids: pd.Series, pred: np.ndarray) -> pd.DataFrame:
    """构建 Kaggle 提交 DataFrame。"""
    submission = pd.DataFrame({"Id": ids.values, "SalePrice": pred})
    return submission


def saveSubmission(
    submissionDf: pd.DataFrame, filepath: str = SUBMISSION_FILEPATH
) -> None:
    """保存提交文件。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    submissionDf.to_csv(filepath, index=False)
    print(f"提交文件已保存: {filepath}")


# =============================================================================
# 第四阶段：主流程入口
# =============================================================================


def main() -> None:
    """命令行入口。"""
    print("=" * 60)
    print("House Prices 预测与提交文件生成")
    print("=" * 60)

    print("[1] 加载模型与训练报告...")
    model = loadModel()
    report = loadTrainReport()

    featureCols = report.get("featureCols", [])
    useLogTarget = bool(report.get("useLogTarget", True))
    if not featureCols:
        raise ValueError("train_report.json 中缺少 featureCols，无法做列对齐")

    print("[2] 读取测试特征并列对齐...")
    xTestDf = loadTestFeatures()
    xTestAligned = alignTestColumns(xTestDf, featureCols)
    print(f"测试集原始维度: {xTestDf.shape}, 对齐后维度: {xTestAligned.shape}")

    print("[3] 执行预测...")
    pred = predictSalePrice(model, xTestAligned, useLogTarget=useLogTarget)
    print(
        f"预测统计: min={pred.min():.2f}, max={pred.max():.2f}, mean={pred.mean():.2f}"
    )

    print("[4] 生成并保存提交文件...")
    ids = loadSubmissionIds()
    submissionDf = buildSubmission(ids, pred)
    saveSubmission(submissionDf)


if __name__ == "__main__":
    main()
