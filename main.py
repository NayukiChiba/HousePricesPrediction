"""
House Prices Prediction 一键运行入口

支持三种模式（通过命令行参数控制）：
    all       - 完整流程：特征工程 → 训练 → 预测（默认）
    fe        - 仅执行特征工程
    train     - 仅执行训练（需先完成特征工程）
    predict   - 仅执行预测（需先完成训练）

使用方法：
    python main.py
    python main.py all
    python main.py fe
    python main.py train
    python main.py predict
"""

import argparse
import sys
import time
from pathlib import Path

# 保证 src 包和 config 可被直接导入
sys.path.insert(0, str(Path(__file__).resolve().parent))


def runFeatureEngineering() -> None:
    """执行特征工程步骤。"""
    import pandas as pd

    from config import (
        TEST_FILEPATH,
        TEST_PROCESSED_FILEPATH,
        TRAIN_FILEPATH,
        TRAIN_PROCESSED_FILEPATH,
    )
    from src.featureEngineering import runFeatureEngineering as _run

    print("[1/3] 特征工程")
    print("-" * 40)

    trainDf = pd.read_csv(TRAIN_FILEPATH)
    testDf = pd.read_csv(TEST_FILEPATH)

    trainProcessedDf, testProcessedDf = _run(trainDf, testDf)

    import os

    os.makedirs(os.path.dirname(TRAIN_PROCESSED_FILEPATH), exist_ok=True)
    trainProcessedDf.to_csv(TRAIN_PROCESSED_FILEPATH, index=False)
    testProcessedDf.to_csv(TEST_PROCESSED_FILEPATH, index=False)

    print(f"训练集: {trainProcessedDf.shape}  →  {TRAIN_PROCESSED_FILEPATH}")
    print(f"测试集: {testProcessedDf.shape}  →  {TEST_PROCESSED_FILEPATH}")


def runTrain() -> None:
    """执行模型训练步骤。"""
    from src.train import (
        N_SPLITS,
        RANDOM_STATE,
        RIDGE_ALPHA,
        TARGET_COL,
        USE_LOG_TARGET,
        crossValidate,
        loadTrainData,
        saveArtifacts,
        trainFinalModel,
    )

    print("[2/3] 模型训练")
    print("-" * 40)

    xDf, y = loadTrainData(targetCol=TARGET_COL)
    print(f"特征维度: {xDf.shape}, 样本数: {len(y)}")

    print("K 折交叉验证...")
    cvReport = crossValidate(
        xDf=xDf,
        y=y,
        nSplits=N_SPLITS,
        randomState=RANDOM_STATE,
        useLogTarget=USE_LOG_TARGET,
        alpha=RIDGE_ALPHA,
    )
    summary = cvReport["summary"]
    print(
        f"CV RMSLE: {summary['mean_rmsle']:.4f} ± {summary['std_rmsle']:.4f}  |  "
        f"CV RMSE: {summary['mean_rmse']:.1f} ± {summary['std_rmse']:.1f}"
    )

    print("全量训练...")
    model = trainFinalModel(
        xDf=xDf, y=y, useLogTarget=USE_LOG_TARGET, alpha=RIDGE_ALPHA
    )
    # 将特征列名和 log 目标标志写入报告，供 predict 步骤列对齐使用
    cvReport["featureCols"] = xDf.columns.tolist()
    cvReport["useLogTarget"] = USE_LOG_TARGET
    saveArtifacts(model, cvReport)
    print("模型已保存。")


def runPredict() -> None:
    """执行预测并生成提交文件步骤。"""
    from src.predict import (
        alignTestColumns,
        buildSubmission,
        loadModel,
        loadSubmissionIds,
        loadTestFeatures,
        loadTrainReport,
        predictSalePrice,
        saveSubmission,
    )

    print("[3/3] 预测与提交生成")
    print("-" * 40)

    model = loadModel()
    report = loadTrainReport()

    featureCols = report.get("featureCols", [])
    useLogTarget = bool(report.get("useLogTarget", True))
    if not featureCols:
        raise ValueError("train_report.json 中缺少 featureCols，请先重新训练")

    xTestDf = loadTestFeatures()
    xTestAligned = alignTestColumns(xTestDf, featureCols)
    print(f"测试集维度: {xTestDf.shape}  →  对齐后: {xTestAligned.shape}")

    pred = predictSalePrice(model, xTestAligned, useLogTarget=useLogTarget)
    print(f"预测区间: [{pred.min():.0f}, {pred.max():.0f}]  均值: {pred.mean():.0f}")

    ids = loadSubmissionIds()
    submissionDf = buildSubmission(ids, pred)
    saveSubmission(submissionDf)
    print("提交文件已生成。")


STEPS = {
    "fe": runFeatureEngineering,
    "train": runTrain,
    "predict": runPredict,
}

PIPELINES = {
    "all": ["fe", "train", "predict"],
    "fe": ["fe"],
    "train": ["train"],
    "predict": ["predict"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="House Prices Prediction 流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python main.py\n  python main.py train\n  python main.py predict",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=list(PIPELINES.keys()),
        help="运行模式（默认: all）",
    )
    args = parser.parse_args()

    steps = PIPELINES[args.mode]
    totalStart = time.time()

    print("=" * 60)
    print(f"House Prices Prediction  —  模式: {args.mode}")
    print("=" * 60)

    for step in steps:
        stepStart = time.time()
        STEPS[step]()
        elapsed = time.time() - stepStart
        print(f"  完成，耗时 {elapsed:.1f}s\n")

    totalElapsed = time.time() - totalStart
    print("=" * 60)
    print(f"全部完成，总耗时 {totalElapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
