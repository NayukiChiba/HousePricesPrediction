"""
特征工程模块

功能：
1. 缺失值处理
2. 类别特征编码
3. 数值特征变换
4. 新特征构造
5. 特征选择

使用方法：
    python -m src.featureEngineering
"""

import os
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import OUTPUTS_DIR, TEST_FILEPATH, TRAIN_FILEPATH


class FeatureEngineer:
    """
    房价预测特征工程器

    Args:
        targetCol: 目标列名，默认 SalePrice
        idCol: 主键列名，默认 Id
        randomState: 随机种子，保证可复现
    """

    def __init__(
        self,
        targetCol: str = "SalePrice",
        idCol: str = "Id",
        randomState: int = 42,
    ) -> None:
        """初始化特征工程配置。"""
        # 目标列
        self.targetCol = targetCol
        # id列
        self.idCol = idCol
        # 随机种子
        self.randomState = randomState

        # 在fit中学习得到的特征工程规则
        self.numericFillValues: dict[str, float] = {}
        self.categoricalFillValues: dict[str, str] = {}
        self.categoricalLevels: dict[str, list[str]] = {}
        self.logTransformCols: list[str] = []

        # 训练完成后的特征列顺序（不包含id和target）
        self.featureCols: list[str] = []
        # 是否已在训练集上调用fit
        self.isFitted: bool = False

    def fit(self, trainDf: pd.DataFrame) -> "FeatureEngineer":
        """
        在训练集上学习特征工程规则（如填充统计量、编码映射等）。

        Args:
            trainDf: 训练数据

        Returns:
            当前实例（支持链式调用）
        """
        df = trainDf.copy()

        # 训练应该排除目标列
        # 忽略错误以防目标列不存在
        featureDf = df.drop(columns=[self.targetCol], errors="ignore")

        # 数值列和类别列
        numericCols = featureDf.select_dtypes(include=[np.number]).columns.tolist()
        categoricalCols = featureDf.select_dtypes(exclude=[np.number]).columns.tolist()

        # 数值填充: 中位数
        self.numericFillValues = {col: featureDf[col].median() for col in numericCols}

        # 类别填充: 众数
        self.categoricalFillValues = {}
        for col in categoricalCols:
            # mode() 返回一个 Series，可能有多个众数，因此取第一个
            # dropna=True 确保忽略缺失值计算众数
            modeSeries = featureDf[col].mode(dropna=True)
            # 使用iloc[0]获取第一个众数，如果modeSeries为空（即全是缺失值），则使用"Missing"作为填充值
            self.categoricalFillValues[col] = (
                str(modeSeries.iloc[0]) if not modeSeries.empty else "Missing"
            )

            # 类别水平编码: Label Encoding
            self.categoricalLevels = {}
            for col in categoricalCols:
                vals = featureDf[col].dropna().astype(str).unique().tolist()
                self.categoricalLevels[col] = sorted(vals)  # 按字母顺序排序类别水平

            # 偏态列记录：偏态特征通常需要对数变换等处理
            self.logTransformCols = []
            for col in numericCols:
                # 计算偏态系数（skewness），绝对值大于0.75且最小值非负的数值列可能需要对数变换
                colSeries = featureDf[col].dropna()
                if colSeries.empty:
                    continue
                skewness = colSeries.skew()
                # 仅当偏态系数存在且绝对值大于0.75且最小值非负时，才考虑对数变换
                if pd.notna(skewness) and abs(skewness) > 0.75 and colSeries.min() >= 0:
                    self.logTransformCols.append(col)

            self.isFitted = True

            # TODO: 用训练集跑一次完整的流程，确保所有步骤的配置都正确
            pass

    def transform(self, df: pd.DataFrame, isTrain: bool = False) -> pd.DataFrame:
        """
        按已学习规则对数据进行特征工程变换。

        Args:
            df: 待变换数据（训练集或测试集）
            isTrain: 是否为训练集（决定是否保留目标列）

        Returns:
            变换后的特征数据
        """
        if not self.isFitted:
            raise RuntimeError("请先在训练集上调用 fit() 方法学习特征工程规则。")
        return self._applyPipeline(df, isTrain, alignColumns=True)

    def fitTransform(self, trainDf: pd.DataFrame) -> pd.DataFrame:
        """
        在训练集上执行 fit + transform。

        Args:
            trainDf: 训练数据

        Returns:
            训练集特征工程结果
        """
        self.fit(trainDf)
        return self.transform(trainDf, isTrain=True)

    def handleMissingValues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值。

        Args:
            df: 输入数据

        Returns:
            缺失值处理后的数据
        """
        out = df.copy()
        # 在数值列中填充中位数
        for col, fillVal in self.numericFillValues.items():
            if col in out.columns:
                out[col] = out[col].fillna(fillVal)
        # 在非数值列中填充众数
        for col, fillVal in self.categoricalFillValues.items():
            if col in out.columns:
                out[col] = out[col].fillna(fillVal)

        return out

    def encodeCategoricalFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        编码类别特征。
        执行类别编码（如Label/One-Hot/Target Encoding），并记录编码映射。

        Args:
            df: 输入数据

        Returns:
            编码后的数据
        """
        out = df.copy()
        categoricalCols = out.select_dtypes(exclude=[np.number]).columns.tolist()

        # id / target 列不编码
        categoricalCols = [
            col for col in categoricalCols if col not in [self.idCol, self.targetCol]
        ]

        # 如果非数值列中没有需要编码的列，直接返回
        if len(categoricalCols) == 0:
            return out

        # 把类别列转换为字符串类型，确保一致性
        for col in categoricalCols:
            out[col] = out[col].astype(str)

        # Label Encoding
        # drop_first=False 保留所有类别水平，避免信息丢失
        out = pd.get_dummies(out, columns=categoricalCols, drop_first=False, dtype=int)
        return out

    def transformNumericFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        变换数值特征（如对数变换、标准化等）。

        Args:
            df: 输入数据

        Returns:
            数值变换后的数据
        """
        out = df.copy()

        for col in self.logTransformCols:
            if col in out.columns:
                # 对数变换前确保数值非负，添加1以避免对数0的情况
                out[col] = np.log1p(out[col])

        return out

    def buildCustomFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构造业务特征。

        Args:
            df: 输入数据

        Returns:
            新增特征后的数据
        """
        out = df.copy()

        # TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
        if all(col in out.columns for col in ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]):
            out["TotalSF"] = (
                out["TotalBsmtSF"].fillna(0)
                + out["1stFlrSF"].fillna(0)
                + out["2ndFlrSF"].fillna(0)
            )

        # TotalBathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
        if all(
            col in out.columns
            for col in ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
        ):
            out["TotalBathrooms"] = (
                out["FullBath"].fillna(0)
                + 0.5 * out["HalfBath"].fillna(0)
                + out["BsmtFullBath"].fillna(0)
                + 0.5 * out["BsmtHalfBath"].fillna(0)
            )

        # TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
        if all(
            col in out.columns
            for col in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
        ):
            out["TotalPorchSF"] = (
                out["OpenPorchSF"].fillna(0)
                + out["EnclosedPorch"].fillna(0)
                + out["3SsnPorch"].fillna(0)
                + out["ScreenPorch"].fillna(0)
            )

        # HouseAge = YrSold - YearBuilt
        if all(col in out.columns for col in ["YrSold", "YearBuilt"]):
            out["HouseAge"] = out["YrSold"].fillna(0) - out["YearBuilt"].fillna(0)

        # RemodelAge = YrSold - YearRemodAdd
        if all(col in out.columns for col in ["YrSold", "YearRemodAdd"]):
            out["RemodelAge"] = out["YrSold"].fillna(0) - out["YearRemodAdd"].fillna(0)

        # GarageAge = YrSold - GarageYrBlt
        if all(col in out.columns for col in ["YrSold", "GarageYrBlt"]):
            out["GarageAge"] = out["YrSold"].fillna(0) - out["GarageYrBlt"].fillna(0)

        return out

    def selectFeatures(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        选择最终用于建模的特征。

        Args:
            df: 输入数据

        Returns:
            特征选择后的数据
        """
        out = df.copy()
        dropCols: list[str] = []
        for col in out.columns:
            if col in {self.idCol, self.targetCol}:
                continue
            if out[col].isna().all():
                dropCols.append(col)
                continue
            if out[col].nunique(dropna=False) <= 1:
                dropCols.append(col)
        if dropCols:
            out = out.drop(columns=dropCols, errors="ignore")

        return out

    def saveFeatureMetadata(self, outputDir: str | None = None) -> None:
        """
        保存特征工程元信息（列名、映射、参数等）。

        Args:
            outputDir: 输出目录，默认写入 OUTPUTS_DIR/featureEngineering
        """
        if outputDir is None:
            outputDir = os.path.join(OUTPUTS_DIR, "featureEngineering")
        os.makedirs(outputDir, exist_ok=True)

        metadata: dict[str, any] = {
            "targetCol": self.targetCol,
            "idCol": self.idCol,
            "randomState": self.randomState,
            "numericFillValues": self.numericFillValues,
            "categoricalFillValues": self.categoricalFillValues,
            "categoricalLevels": self.categoricalLevels,
            "logTransformCols": self.logTransformCols,
            "featureCols": self.featureCols,
            "isFitted": self.isFitted,
        }

        metaPath = os.path.join(outputDir, "feature_metadata.json")
        with open(metaPath, "w") as f:
            import json

            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _applyPipeline(
        self, df: pd.DataFrame, isTrain: bool, alignColumns: bool = False
    ) -> pd.DataFrame:
        """
        内部方法：按顺序应用特征工程步骤。

        Args:
            df: 输入数据
            isTrain: 是否为训练集（决定是否保留目标列）
            alignColumns: 是否对齐训练集和测试集的列（仅测试集时使用）
        Returns:
            处理后的数据
        """
        out = df.copy()
        out = self.handleMissingValues(out)
        out = self.encodeCategoricalFeatures(out)
        out = self.transformNumericFeatures(out)
        out = self.buildCustomFeatures(out)
        out = self.selectFeatures(out)

        # 训练和测试不保留ID
        if self.idCol in out.columns:
            out = out.drop(columns=[self.idCol], errors="ignore")

        # 测试集灭有目标列, 如果isTrain=True则保留目标列，否则丢弃
        if not isTrain and self.targetCol in out.columns:
            out = out.drop(columns=[self.targetCol], errors="ignore")

        # 按照训练集的列顺序对齐测试集
        if alignColumns and len(self.featureCols) > 0:
            if isTrain and self.targetCol in out.columns:
                y = out[self.targetCol].copy()
                x = out.drop(columns=[self.targetCol], errors="ignore")
                x = x.reindex(columns=self.featureCols, fill_value=0)
                out = pd.concat([x, y], axis=1)
            else:
                out = out.reindex(columns=self.featureCols, fill_value=0)
        return out


def runFeatureEngineering(
    trainDf: pd.DataFrame,
    testDf: pd.DataFrame,
    targetCol: str = "SalePrice",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    运行完整特征工程流程。

    Args:
        trainDf: 原始训练集
        testDf: 原始测试集
        targetCol: 目标列名

    Returns:
        (trainProcessedDf, testProcessedDf)
    """
    fe = FeatureEngineer(targetCol=targetCol)
    print("[FeatureEngineering] 开始训练集 fit + transform ...")
    trainProcessedDf = fe.fitTransform(trainDf)
    print("[FeatureEngineering] 开始测试集 transform ...")
    testProcessedDf = fe.transform(testDf, isTrain=False)

    outputDir = os.path.join(OUTPUTS_DIR, "featureEngineering")
    fe.saveFeatureMetadata(outputDir=outputDir)

    return trainProcessedDf, testProcessedDf


def main() -> None:
    """命令行入口。"""
    print("=" * 60)
    print("房价预测特征工程")
    print("=" * 60)

    outputDir = os.path.join(OUTPUTS_DIR, "featureEngineering")
    os.makedirs(outputDir, exist_ok=True)

    print("[1] 读取数据...")
    trainDf = pd.read_csv(TRAIN_FILEPATH)
    testDf = pd.read_csv(TEST_FILEPATH)

    print("[2] 执行特征工程...")
    trainProcessedDf, testProcessedDf = runFeatureEngineering(trainDf, testDf)

    print("[3] 保存结果...")
    trainOutPath = os.path.join(outputDir, "train_processed.csv")
    testOutPath = os.path.join(outputDir, "test_processed.csv")

    trainProcessedDf.to_csv(trainOutPath, index=False)
    testProcessedDf.to_csv(testOutPath, index=False)

    print(f"训练集处理后形状: {trainProcessedDf.shape}")
    print(f"测试集处理后形状: {testProcessedDf.shape}")
    print(f"训练集输出: {trainOutPath}")
    print(f"测试集输出: {testOutPath}")


if __name__ == "__main__":
    main()
