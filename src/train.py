"""
模型训练模块（学习版）

功能：
1. 读取特征工程后的训练数据
2. 使用 sklearn 做 K 折交叉验证评估（RMSE / RMSLE）
3. 训练最终线性回归模型（Pipeline: StandardScaler + Ridge）
4. 保存模型与训练报告

使用方法：
    python -m src.train
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    MODEL_FILEPATH,
    N_SPLITS,
    RANDOM_STATE,
    RIDGE_ALPHA,
    TARGET_COL,
    TRAIN_PROCESSED_FILEPATH,
    TRAIN_REPORT_FILEPATH,
    USE_LOG_TARGET,
)

# =============================================================================
# 第一阶段：评估指标
# =============================================================================


def rmse(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    计算 RMSE（Root Mean Squared Error）。

    含义：
        误差平方后取平均，再开根号。
        数值越小越好，单位与房价单位一致（美元）。
    """
    return float(np.sqrt(mean_squared_error(yTrue, yPred)))


def rmsle(yTrue: np.ndarray, yPred: np.ndarray) -> float:
    """
    计算 RMSLE（Root Mean Squared Logarithmic Error）。

    说明：
        - Kaggle House Prices 常见关注指标之一是对数空间误差。
        - 对高价房与低价房的相对误差更公平。
        - 预测值若出现负数不合法，先 clip 到 0。
    """
    yPred = np.clip(yPred, a_min=0, a_max=None)
    return float(
        np.sqrt(
            np.mean(
                (np.log1p(yPred) - np.log1p(np.clip(yTrue, a_min=0, a_max=None))) ** 2
            )
        )
    )


# =============================================================================
# 第二阶段：模型构建
# =============================================================================


def buildModel(alpha: float = 1.0) -> Pipeline:
    """
    构建 sklearn 模型管道。

    Pipeline 结构：
        1) StandardScaler：标准化特征，避免某些数值范围过大主导优化。
        2) Ridge：线性回归 + L2 正则，降低过拟合风险。

    Args:
        alpha: Ridge 正则强度，越大约束越强。
    """
    model = Pipeline(
        steps=[
            # 对每个特征做 z-score 标准化：(x - mean) / std
            ("scaler", StandardScaler()),
            # 线性模型：在最小二乘基础上加入 L2 正则项
            ("regressor", Ridge(alpha=alpha, random_state=42)),
        ]
    )
    return model


def loadTrainData(targetCol: str = "SalePrice") -> tuple[pd.DataFrame, pd.Series]:
    """
    读取处理后的训练数据。

    数据来源：
        config.TRAIN_PROCESSED_FILEPATH

    约束：
        - 必须包含目标列 targetCol
        - 其余列默认都作为特征列
    """
    # 1) 校验文件是否存在（避免静默失败）
    if not os.path.exists(TRAIN_PROCESSED_FILEPATH):
        raise FileNotFoundError(
            f"未找到处理后训练集: {TRAIN_PROCESSED_FILEPATH}，请先运行特征工程。"
        )

    # 2) 读取 CSV
    trainDf = pd.read_csv(TRAIN_PROCESSED_FILEPATH)

    # 3) 校验目标列存在性
    if targetCol not in trainDf.columns:
        raise ValueError(f"训练数据缺少目标列: {targetCol}")

    # 4) X/y 拆分
    xDf = trainDf.drop(columns=[targetCol])
    y = trainDf[targetCol]
    return xDf, y


# =============================================================================
# 第三阶段：交叉验证
# =============================================================================


def crossValidate(
    xDf: pd.DataFrame,
    y: pd.Series,
    nSplits: int = 5,
    randomState: int = 42,
    useLogTarget: bool = True,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """
    K 折交叉验证。

    流程：
        - 将数据分成 K 份（默认 5 折）
        - 每次取 1 份做验证，剩余 K-1 份做训练
        - 重复 K 次，汇总均值与标准差

    Args:
        xDf: 特征 DataFrame
        y: 目标 Series
        nSplits: 折数
        randomState: 随机种子
        useLogTarget: 是否对目标做 log1p
        alpha: Ridge 正则强度
    """
    # 转 numpy：训练/预测效率更稳定
    x = xDf.values.astype(np.float64)
    yArr = y.values.astype(np.float64)

    # shuffle=True 让每折分布更均匀；random_state 保证可复现
    kf = KFold(n_splits=nSplits, shuffle=True, random_state=randomState)
    foldMetrics: list[dict[str, float]] = []

    # 枚举每一折：trainIdx 是训练样本索引，validIdx 是验证样本索引
    for i, (trainIdx, validIdx) in enumerate(kf.split(x), start=1):
        xTrain, xValid = x[trainIdx], x[validIdx]
        yTrain, yValid = yArr[trainIdx], yArr[validIdx]

        # 每一折都重新创建并训练模型，防止信息泄漏
        model = buildModel(alpha=alpha)

        if useLogTarget:
            # 在训练阶段对 y 做 log1p，可缓解长尾分布
            yTrainLog = np.log1p(np.clip(yTrain, a_min=0, a_max=None))
            model.fit(xTrain, yTrainLog)

            # 预测后把对数空间结果映射回原空间
            predLog = model.predict(xValid)
            pred = np.expm1(predLog)
        else:
            model.fit(xTrain, yTrain)
            pred = model.predict(xValid)

        # 记录本折指标
        foldRmse = rmse(yValid, pred)
        foldRmsle = rmsle(yValid, pred)
        foldMetrics.append({"rmse": foldRmse, "rmsle": foldRmsle})

        print(f"Fold {i}/{nSplits} - RMSE: {foldRmse:.4f}, RMSLE: {foldRmsle:.4f}")

    meanRmse = float(np.mean([m["rmse"] for m in foldMetrics]))
    stdRmse = float(np.std([m["rmse"] for m in foldMetrics]))
    meanRmsle = float(np.mean([m["rmsle"] for m in foldMetrics]))
    stdRmsle = float(np.std([m["rmsle"] for m in foldMetrics]))

    # 返回每折结果 + 汇总结果，便于后续记录与比较
    return {
        "folds": foldMetrics,
        "summary": {
            "mean_rmse": meanRmse,
            "std_rmse": stdRmse,
            "mean_rmsle": meanRmsle,
            "std_rmsle": stdRmsle,
        },
    }


# =============================================================================
# 第四阶段：全量训练
# =============================================================================


def trainFinalModel(
    xDf: pd.DataFrame,
    y: pd.Series,
    useLogTarget: bool = True,
    alpha: float = 1.0,
) -> Pipeline:
    """
    在全量训练集上训练最终模型。

    逻辑：
        - 先根据超参构建模型
        - 再用全部样本拟合
        - 返回已训练模型对象
    """
    x = xDf.values.astype(np.float64)
    yArr = y.values.astype(np.float64)
    model = buildModel(alpha=alpha)

    if useLogTarget:
        yFit = np.log1p(np.clip(yArr, a_min=0, a_max=None))
    else:
        yFit = yArr

    model.fit(x, yFit)
    return model


def saveArtifacts(model: Pipeline, report: dict[str, Any]) -> None:
    """
    保存模型与训练报告。

    输出：
        - 模型：outputs/model/ridge_pipeline_v1.joblib
        - 报告：outputs/train/train_report.json
    """
    # 1) 确保模型目录存在
    os.makedirs(os.path.dirname(MODEL_FILEPATH), exist_ok=True)

    # 2) 保存模型对象（joblib 适合 sklearn）
    dump(model, MODEL_FILEPATH)

    # 3) 保存训练报告
    os.makedirs(os.path.dirname(TRAIN_REPORT_FILEPATH), exist_ok=True)
    with open(TRAIN_REPORT_FILEPATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"模型已保存: {MODEL_FILEPATH}")
    print(f"训练报告已保存: {TRAIN_REPORT_FILEPATH}")


# =============================================================================
# 第五阶段：主流程入口
# =============================================================================


def main() -> None:
    """
    命令行入口。

    执行顺序：
        [1] 读数据
        [2] 交叉验证
        [3] 全量训练
        [4] 保存产物
    """
    print("=" * 60)
    print("House Prices 模型训练")
    print("=" * 60)

    # 训练配置（学习版先写死，后续可改成 argparse）
    targetCol = TARGET_COL
    randomState = RANDOM_STATE
    nSplits = N_SPLITS
    useLogTarget = USE_LOG_TARGET
    alpha = RIDGE_ALPHA

    print("[1] 加载处理后训练数据...")
    xDf, y = loadTrainData(targetCol=targetCol)
    print(f"特征维度: {xDf.shape}, 目标样本数: {len(y)}")

    print("[2] K 折交叉验证...")
    cvReport = crossValidate(
        xDf=xDf,
        y=y,
        nSplits=nSplits,
        randomState=randomState,
        useLogTarget=useLogTarget,
        alpha=alpha,
    )
    print("CV Summary:")
    print(cvReport["summary"])

    print("[3] 训练最终模型...")
    model = trainFinalModel(xDf=xDf, y=y, useLogTarget=useLogTarget, alpha=alpha)

    print("[4] 保存模型与报告...")
    # 报告中记录关键训练参数，方便复现实验
    report = {
        "modelType": "Pipeline(StandardScaler + Ridge)",
        "alpha": alpha,
        "useLogTarget": useLogTarget,
        "featureCols": xDf.columns.tolist(),
        "targetCol": targetCol,
        "nSplits": nSplits,
        "randomState": randomState,
        "cv": cvReport,
    }
    saveArtifacts(model=model, report=report)


if __name__ == "__main__":
    main()
