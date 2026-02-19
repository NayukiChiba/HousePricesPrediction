"""
探索性数据分析（EDA）模块

功能：
1. 数据概览与基本信息
2. 目标变量分析
3. 缺失值分析
4. 数值特征分析
5. 类别特征分析
6. 异常值检测
7. 特征工程建议

使用方法：
    python -m src.eda
"""

import os
import sys
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from config import OUTPUTS_DIR, TEST_FILEPATH, TRAIN_FILEPATH

# 中文字符
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# =============================================================================
# 第一阶段：数据概览
# =============================================================================


def loadData(
    trainPath: str | None = TRAIN_FILEPATH, testPath: str | None = TEST_FILEPATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练集和测试集

    Args:
        trainPath: 训练集路径，默认使用 config 中的路径
        testPath: 测试集路径，默认使用 config 中的路径

    Returns:
        (trainDf, testDf) 元组
    """
    try:
        trainDf = pd.read_csv(trainPath)
        testDf = pd.read_csv(testPath)
        return trainDf, testDf
    except Exception as e:
        raise OSError(f"加载数据失败: {e}")


def getBasicInfo(df: pd.DataFrame) -> dict[str, Any]:
    """
    获取数据基本信息

    Args:
        df: 数据框

    Returns:
        包含以下信息的字典：
        - shape: 数据维度 (行数, 列数)
        - dtypes: 各列数据类型统计 (如 {'int64': 35, 'object': 43})
        - numericCols: 数值型列名列表
        - categoricalCols: 类别型列名列表
        - memoryUsage: 内存占用 (MB)

    提示：
        - df.shape 获取维度
        - df.dtypes.value_counts() 统计类型
        - df.select_dtypes(include=[np.number]).columns 获取数值列
        - df.memory_usage(deep=True).sum() / 1024**2 获取内存占用
    """
    # 获取全部列的基本信息
    print("数据基本信息:")
    print(df.info())

    # 获取基本信息
    numericCols = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricalCols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 构建字典
    result = {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "numericCols": numericCols,
        "categoricalCols": categoricalCols,
        "memoryUsage": df.memory_usage(deep=True).sum() / 1024**2,
    }

    # 打印基本信息
    print(f"数据维度: {result['shape']}")
    print("数据类型分布:")
    for dtype, count in result["dtypes"].items():
        print(f"  {dtype}: {count} 列")
    print(f"数值型特征数量: {len(result['numericCols'])}")
    print(f"类别型特征数量: {len(result['categoricalCols'])}")

    return result


# =============================================================================
# 第二阶段：目标变量分析
# =============================================================================


def analyzeTarget(df: pd.DataFrame, targetCol: str = "SalePrice") -> dict[str, Any]:
    """
    分析目标变量分布

    Args:
        df: 训练数据
        targetCol: 目标列名

    Returns:
        包含以下信息的字典：
        - stats: 描述性统计字典，包含：
            - mean: 均值
            - median: 中位数
            - std: 标准差
            - skewness: 偏度
            - kurtosis: 峰度
        - isSkewed: 是否偏态（|偏度| > 0.5）
        - suggestLogTransform: 是否建议对数变换（右偏且偏度 > 1）

    提示：
        - df[targetCol].describe() 获取基本统计
        - df[targetCol].skew() 计算偏度
        - df[targetCol].kurtosis() 计算峰度
        - 绘图：plt.figure() + sns.histplot() + sns.kdeplot()
        - Q-Q 图：stats.probplot(df[targetCol], plot=plt)
    """
    # SalePrice 初步定为int64
    # 计算统计量
    target = df[targetCol]
    skewness = target.skew()
    kurtosis = target.kurtosis()

    stats = {
        "mean": target.mean(),
        "median": target.median(),
        "std": target.std(),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

    # 判断偏态和是否需要对数变换
    isSkewed = abs(skewness) > 0.5
    suggestLogTransform = skewness > 1  # 右偏且偏度 > 1

    # 打印信息
    print("描述性统计:")
    print(target.describe())
    print(f"偏度: {skewness:.2f}")
    print(f"峰度: {kurtosis:.2f}")

    if isSkewed:
        print(f"⚠️ 目标变量存在偏态 (偏度={skewness:.2f})")
    if suggestLogTransform:
        print("💡 建议进行对数变换")

    return {
        "stats": stats,
        "isSkewed": isSkewed,
        "suggestLogTransform": suggestLogTransform,
    }


def plotTargetDistribution(df: pd.DataFrame, targetCol: str = "SalePrice") -> None:
    """
    绘制目标变量分布图

    Args:
        df: 训练数据
        targetCol: 目标列名

    绘制内容：
        1. 左图：原始分布（直方图 + KDE）
        2. 右图：对数变换后的分布

    提示：
        - fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        - np.log1p() 进行对数变换（处理 0 值）
        - sns.histplot(data, kde=True, ax=ax)
    """
    # 绘图, 主要是sns.histplot() + sns.kdeplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[targetCol], kde=True, ax=axes[0])
    axes[0].set_title("目标变量分布")
    sns.histplot(np.log1p(df[targetCol]), kde=True, ax=axes[1])
    axes[1].set_title("目标变量对数变换后分布")
    plt.tight_layout()
    filename = "target_distribution.png"
    VIZ_DIR = os.path.join(OUTPUTS_DIR, "eda")
    os.makedirs(VIZ_DIR, exist_ok=True)
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)

    # QQ图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(df[targetCol], plot=axes[0])
    axes[0].set_title("原始 Q-Q 图")
    stats.probplot(np.log1p(df[targetCol]), plot=axes[1])
    axes[1].set_title("对数变换后 Q-Q 图")
    plt.tight_layout()
    filename = "target_qqplot.png"
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)


# =============================================================================
# 第三阶段：缺失值分析
# =============================================================================


def analyzeMissing(df: pd.DataFrame) -> pd.DataFrame:
    """
    分析缺失值情况

    Args:
        df: 数据框

    Returns:
        DataFrame，包含列：
        - column: 列名
        - missingCount: 缺失数量
        - missingPercent: 缺失比例 (0-100)
        - dtype: 数据类型
        按 missingPercent 降序排列，只返回有缺失的列

    提示：
        - df.isnull().sum() 统计缺失数量
        - 注意区分「真缺失」和「NA 表示无此特征」
          例如 PoolQC 缺失可能表示没有泳池，而非数据缺失
    """
    # 缺失布尔矩阵
    missingCount = df.isnull().sum()

    # 缺失比例
    missingPercent = (missingCount / len(df)) * 100

    # 构建DataFrame
    missingDf = pd.DataFrame(
        {
            "columns": missingCount.index,
            "missingCount": missingCount.values,
            "missingPercent": missingPercent.values,
            "dtype": df.dtypes.values,
        }
    )

    # 筛选，只需要缺失 > 0的列
    missingDf = missingDf[missingDf["missingCount"] > 0]

    # 排序, 并且重置索引
    missingDf = missingDf.sort_values(by="missingPercent", ascending=False).reset_index(
        drop=True
    )

    return missingDf


def plotMissingValues(df: pd.DataFrame, topN: int = 20) -> None:
    """
    绘制缺失值可视化图

    Args:
        df: 数据框
        topN: 显示缺失最多的前 N 个特征

    提示：
        - 使用 analyzeMissing() 获取缺失值统计
        - sns.barplot() 绘制水平条形图
        - 添加百分比标签
    """
    missing = analyzeMissing(df)

    # 取前 topN 个
    plotDf = missing.head(topN)

    # 如果没有缺失值, 直接返回
    if len(plotDf) == 0:
        print("没有缺失值需要可视化")
        return

    # 绘制水平条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="missingPercent",
        y="columns",
        data=plotDf,
        ax=ax,
        hue="columns",  # 添加这行
        palette="viridis",
        legend=False,  # 添加这行，不显示图例
    )

    # 添加百分比标签
    for i, (percent, count) in enumerate(
        zip(plotDf["missingPercent"], plotDf["missingCount"])
    ):
        ax.text(percent + 0.5, i, f"{percent:.1f}% ({count})", va="center")

    ax.set_xlabel("缺失比例 (%)")
    ax.set_ylabel("特征")
    ax.set_title(f"缺失值分布（前 {topN} 个）")

    plt.tight_layout()

    # 保存图表
    filename = "missing_values.png"
    VIZ_DIR = os.path.join(OUTPUTS_DIR, "eda")
    os.makedirs(VIZ_DIR, exist_ok=True)
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)


# =============================================================================
# 第四阶段：数值特征分析
# =============================================================================


def analyzeNumericFeatures(
    df: pd.DataFrame, targetCol: str = "SalePrice", topN: int = 10
) -> dict[str, Any]:
    """
    分析数值特征与目标变量的关系

    Args:
        df: 训练数据
        targetCol: 目标列名
        topN: 返回相关性最高的前 N 个特征

    Returns:
        包含以下信息的字典：
        - correlations: Series，所有数值特征与目标的相关系数（降序）
        - topFeatures: 相关性最高的 topN 个特征名列表
        - multicollinearity: 特征间高度相关的特征对列表
          格式：[(feat1, feat2, corr), ...]，|corr| > 0.8

    提示：
        - df.select_dtypes(include=[np.number]) 选择数值列
        - df.corr() 计算相关矩阵
        - corrMatrix[targetCol].sort_values(ascending=False)
    """
    # 获取数值列
    numericCols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 计算相关矩阵
    corrMatrix = df[numericCols].corr()
    # 获取与目标变量的相关系数，并排序
    correlations = corrMatrix[targetCol].sort_values(ascending=False)
    # 获取相关性最高的 topN 个特征（排除目标列本身）
    topFeatures = correlations.index[1 : topN + 1].tolist()  # 排除目标列本身
    # 检测多重共线性，找出相关性 > 0.8 的特征对
    multicollinearity = []
    for i in range(len(numericCols)):
        for j in range(i + 1, len(numericCols)):
            feat1 = numericCols[i]
            feat2 = numericCols[j]
            corr = corrMatrix.loc[feat1, feat2]
            if abs(corr) > 0.8:
                multicollinearity.append((feat1, feat2, corr))
    return {
        "correlations": correlations,
        "topFeatures": topFeatures,
        "multicollinearity": multicollinearity,
    }


def plotCorrelationHeatmap(
    df: pd.DataFrame, features: list[str] | None = None, targetCol: str = "SalePrice"
) -> None:
    """
    绘制相关性热力图

    Args:
        df: 数据框
        features: 要绘制的特征列表，None 则使用 top 10 相关特征
        targetCol: 目标列名

    提示：
        - sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', center=0)
        - 设置 fmt='.2f' 显示两位小数
    """
    if features is None:
        features = analyzeNumericFeatures(df, targetCol)["topFeatures"]
    # 包含目标列和选定特征
    colsToPlot = [targetCol] + features
    corrMatrix = df[colsToPlot].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrMatrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("相关性热力图")
    plt.tight_layout()
    filename = "correlation_heatmap.png"
    VIZ_DIR = os.path.join(OUTPUTS_DIR, "eda")
    os.makedirs(VIZ_DIR, exist_ok=True)
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)


def plotScatterWithTarget(
    df: pd.DataFrame, features: list[str], targetCol: str = "SalePrice", ncols: int = 3
) -> None:
    """
    绘制特征与目标变量的散点图

    Args:
        df: 数据框
        features: 要绘制的特征列表
        targetCol: 目标列名
        ncols: 每行显示的图数量

    提示：
        - nrows = (len(features) + ncols - 1) // ncols 计算行数
        - fig, axes = plt.subplots(nrows, ncols, figsize=(...))
        - sns.scatterplot(x=feat, y=targetCol, data=df, ax=ax, alpha=0.5)
    """
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        sns.scatterplot(x=feature, y=targetCol, data=df, ax=axes[i], alpha=0.5)
        axes[i].set_title(f"{feature} vs {targetCol}")
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    filename = "scatter_with_target.png"
    VIZ_DIR = os.path.join(OUTPUTS_DIR, "eda")
    os.makedirs(VIZ_DIR, exist_ok=True)
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)


# =============================================================================
# 第五阶段：类别特征分析
# =============================================================================


def analyzeCategoricalFeatures(
    df: pd.DataFrame, targetCol: str = "SalePrice"
) -> dict[str, Any]:
    """
    分析类别特征

    Args:
        df: 训练数据
        targetCol: 目标列名

    Returns:
        包含以下信息的字典：
        - cardinality: Dict，每个类别特征的唯一值数量
          格式：{feature_name: unique_count}
        - highCardinality: 高基数特征列表（唯一值 > 10）
        - lowCardinality: 低基数特征列表（唯一值 <= 10）
        - targetMeanByCategory: Dict，每个类别特征各取值的目标均值
          格式：{feature_name: {category: mean_target}}

    提示：
        - df.select_dtypes(include=['object']) 选择类别列
        - df[col].nunique() 获取唯一值数量
        - df.groupby(col)[targetCol].mean() 计算各类别的目标均值
    """
    # 想要分析一个类别的特征, 首先需要获取所有类别特征的列名
    # 这里的类别就是非数值的特征, 也就是 object 类型的特征
    categoricalCols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # 去除基数, 基数过高的特征不适合做类别分析, 因为每个类别的样本数量太少了
    cardinality = {col: df[col].nunique() for col in categoricalCols}
    # 高基数就是唯一值数量大于 10 的特征, 低基数就是唯一值数量小于等于 10 的特征
    highCardinality = [col for col, count in cardinality.items() if count > 10]
    lowCardinality = [col for col, count in cardinality.items() if count <= 10]

    targetMeanByCategory = {}
    for col in lowCardinality:
        targetMeanByCategory[col] = df.groupby(col)[targetCol].mean().to_dict()

    return {
        "cardinality": cardinality,
        "highCardinality": highCardinality,
        "lowCardinality": lowCardinality,
        "targetMeanByCategory": targetMeanByCategory,
    }


def plotCategoricalVsTarget(
    df: pd.DataFrame, features: list[str], targetCol: str = "SalePrice", ncols: int = 2
) -> None:
    """
    绘制类别特征与目标变量的箱线图

    Args:
        df: 数据框
        features: 要绘制的类别特征列表
        targetCol: 目标列名
        ncols: 每行显示的图数量

    提示：
        - 按目标变量均值对类别排序后绘制
        - order = df.groupby(feat)[targetCol].mean().sort_values().index
        - sns.boxplot(x=feat, y=targetCol, data=df, order=order, ax=ax)
        - plt.xticks(rotation=45) 旋转 x 轴标签
    """
    nrows = (len(features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()
    for i, feature in enumerate(features):
        # 获取目标变量均值排序的类别顺序
        order = df.groupby(feature)[targetCol].mean().sort_values().index
        sns.boxplot(x=feature, y=targetCol, data=df, order=order, ax=axes[i])
        axes[i].set_title(f"{feature} vs {targetCol}")
        axes[i].tick_params(axis="x", rotation=45)
    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    filename = "categorical_vs_target.png"
    VIZ_DIR = os.path.join(OUTPUTS_DIR, "eda")
    os.makedirs(VIZ_DIR, exist_ok=True)
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)


# =============================================================================
# 第六阶段：异常值检测
# =============================================================================


def detectOutliers(
    df: pd.DataFrame, features: list[str], method: str = "iqr", threshold: float = 1.5
) -> dict[str, list[int]]:
    """
    检测异常值

    Args:
        df: 数据框
        features: 要检测的特征列表
        method: 检测方法
            - 'iqr': 四分位距法，threshold 为 IQR 倍数（默认 1.5）
            - 'zscore': Z-score 法，threshold 为 Z 值阈值（默认 3）
        threshold: 阈值参数

    Returns:
        字典，key 为特征名，value 为异常值的索引列表

    提示：
        IQR 方法：
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            异常：< Q1 - threshold * IQR 或 > Q3 + threshold * IQR

        Z-score 方法：
            z = (df[col] - df[col].mean()) / df[col].std()
            异常：|z| > threshold
    """
    outlierIndices = {}
    for feature in features:
        if method == "iqr":
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lowerBound = Q1 - threshold * IQR
            upperBound = Q3 + threshold * IQR
            outliers = df[
                (df[feature] < lowerBound) | (df[feature] > upperBound)
            ].index.tolist()
            outlierIndices[feature] = outliers
        elif method == "zscore":
            z = (df[feature] - df[feature].mean()) / df[feature].std()
            outliers = df[np.abs(z) > threshold].index.tolist()
            outlierIndices[feature] = outliers
        else:
            raise ValueError(f"未知的异常值检测方法: {method}")
    return outlierIndices


def plotOutliers(
    df: pd.DataFrame,
    feature: str,
    targetCol: str = "SalePrice",
    outlierIndices: list[int] | None = None,
) -> None:
    """
    绘制异常值散点图

    Args:
        df: 数据框
        feature: 特征名
        targetCol: 目标列名
        outlierIndices: 异常值索引列表，None 则自动检测

    提示：
        - 正常点用蓝色，异常点用红色标记
        - plt.scatter() 分别绘制正常点和异常点
    """
    if outlierIndices is None:
        outlierIndices = detectOutliers(df, [feature])  # 自动检测异常值
    normalData = df.drop(index=outlierIndices)
    outlierData = df.loc[outlierIndices]
    plt.figure(figsize=(8, 6))
    plt.scatter(
        normalData[feature],
        normalData[targetCol],
        color="blue",
        alpha=0.5,
        label="正常点",
    )
    plt.scatter(
        outlierData[feature],
        outlierData[targetCol],
        color="red",
        alpha=0.5,
        label="异常点",
    )
    plt.xlabel(feature)
    plt.ylabel(targetCol)
    plt.title(f"{feature} vs {targetCol}（异常值标记）")
    plt.legend()
    plt.tight_layout()
    filename = f"outliers_{feature}.png"
    VIZ_DIR = os.path.join(OUTPUTS_DIR, "eda", "outliers")
    os.makedirs(VIZ_DIR, exist_ok=True)
    filepath = os.path.join(VIZ_DIR, filename)
    plt.savefig(filepath)


# =============================================================================
# 第七阶段：特征工程建议
# =============================================================================


def suggestFeatureEngineering(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    基于 EDA 结果给出特征工程建议

    Args:
        df: 数据框

    Returns:
        建议列表，每个建议是一个字典，包含：
        - type: 建议类型
            - 'combine': 组合特征
            - 'transform': 变换特征
            - 'encode': 编码建议
            - 'drop': 删除建议
        - description: 具体描述
        - features: 涉及的特征列表
        - code: 示例代码（可选）

    常见建议（House Prices 数据集）：
        1. 组合特征：
           - TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
           - TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
           - TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch

        2. 年龄/时间特征：
           - HouseAge = YrSold - YearBuilt
           - RemodAge = YrSold - YearRemodAdd
           - GarageAge = YrSold - GarageYrBlt

        3. 对数变换：
           - 对右偏特征（如 LotArea, GrLivArea）做 log1p 变换

        4. 删除特征：
           - 缺失率 > 80% 的特征
           - 与目标相关性极低（|r| < 0.05）的特征
    """
    suggestions = []

    # 组合特征建议
    suggestions.append(
        {
            "type": "combine",
            "description": "创建总面积特征 TotalSF",
            "features": ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"],
            "code": "df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']",
        }
    )
    suggestions.append(
        {
            "type": "combine",
            "description": "创建总浴室数特征 TotalBath",
            "features": ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
            "code": (
                "df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + "
                "df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']"
            ),
        }
    )
    suggestions.append(
        {
            "type": "combine",
            "description": "创建总门廊面积特征 TotalPorchSF",
            "features": ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"],
            "code": (
                "df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + "
                "df['3SsnPorch'] + df['ScreenPorch']"
            ),
        }
    )

    # 年龄/时间特征建议
    suggestions.append(
        {
            "type": "transform",
            "description": "创建房屋年龄特征 HouseAge",
            "features": ["YrSold", "YearBuilt"],
            "code": "df['HouseAge'] = df['YrSold'] - df['YearBuilt']",
        }
    )
    suggestions.append(
        {
            "type": "transform",
            "description": "创建翻新年龄特征 RemodAge",
            "features": ["YrSold", "YearRemodAdd"],
            "code": ("df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']"),
        }
    )
    suggestions.append(
        {
            "type": "transform",
            "description": "创建车库年龄特征 GarageAge",
            "features": ["YrSold", "GarageYrBlt"],
            "code": ("df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']"),
        }
    )
    return suggestions


# =============================================================================
# 主函数：运行完整 EDA 流程
# =============================================================================


def main():
    """主函数"""
    print("=" * 60)
    print("House Prices EDA 探索性数据分析")
    print("=" * 60)

    train, test = loadData()
    print("\n[1] 数据概览")
    getBasicInfo(train)

    analyzeTarget(train)
    plotTargetDistribution(train)

    print("\n[2] 缺失值分析")
    print(analyzeMissing(train))
    plotMissingValues(train)

    print("\n[3] 数值特征分析")
    numericAnalysis = analyzeNumericFeatures(train)
    print("相关性最高的特征:", numericAnalysis["topFeatures"])
    plotCorrelationHeatmap(train, numericAnalysis["topFeatures"])
    plotScatterWithTarget(train, numericAnalysis["topFeatures"])

    print("\n[4] 类别特征分析")
    categoricalAnalysis = analyzeCategoricalFeatures(train)
    print("高基数特征:", categoricalAnalysis["highCardinality"])
    print("低基数特征:", categoricalAnalysis["lowCardinality"])
    plotCategoricalVsTarget(train, categoricalAnalysis["lowCardinality"])

    print("\n[5] 异常值检测")
    outliers = detectOutliers(train, numericAnalysis["topFeatures"])
    for feature, indices in outliers.items():
        print(f"{feature} 的异常值索引: {indices}")
        plotOutliers(train, feature, outlierIndices=indices)

    print("\n[6] 特征工程建议")
    suggestions = suggestFeatureEngineering(train)
    for suggestion in suggestions:
        print(f"建议类型: {suggestion['type']}")
        print(f"描述: {suggestion['description']}")
        print(f"涉及特征: {suggestion['features']}")
        print(f"示例代码:\n{suggestion['code']}\n")
        print("-" * 40)


if __name__ == "__main__":
    main()
