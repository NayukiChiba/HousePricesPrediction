"""
全局配置文件
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUTS_DIR, "model")
TRAIN_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "train")
SUBMISSION_DIR = os.path.join(OUTPUTS_DIR, "submissions")

# 数据集部分
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
TRAIN_PROCESSED_FILENAME = "train_processed.csv"
TEST_PROCESSED_FILENAME = "test_processed.csv"

TRAIN_FILEPATH = os.path.join(DATASETS_DIR, "raw", TRAIN_FILENAME)
TEST_FILEPATH = os.path.join(DATASETS_DIR, "raw", TEST_FILENAME)
TRAIN_PROCESSED_FILEPATH = os.path.join(
    DATASETS_DIR, "processed", TRAIN_PROCESSED_FILENAME
)
TEST_PROCESSED_FILEPATH = os.path.join(
    DATASETS_DIR, "processed", TEST_PROCESSED_FILENAME
)

# 训练/预测相关文件
MODEL_FILENAME = "ridge_pipeline_v1.joblib"
TRAIN_REPORT_FILENAME = "train_report.json"
SUBMISSION_FILENAME = "submission_v1.csv"

MODEL_FILEPATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
TRAIN_REPORT_FILEPATH = os.path.join(TRAIN_OUTPUT_DIR, TRAIN_REPORT_FILENAME)
SUBMISSION_FILEPATH = os.path.join(SUBMISSION_DIR, SUBMISSION_FILENAME)

# 训练超参数（学习版默认值）
TARGET_COL = "SalePrice"
RANDOM_STATE = 42
N_SPLITS = 5
USE_LOG_TARGET = True
RIDGE_ALPHA = 1.0
