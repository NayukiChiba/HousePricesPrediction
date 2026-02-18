"""
全局配置文件
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUTS_DIR, "model")

# 数据集部分
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
TRAIN_PROCESSED_FILENAME = "train_processed.csv"
TEST_PROCESSED_FILENAME = "test_processed.csv"

TRAIN_FILEPATH = os.path.join(DATASETS_DIR, TRAIN_FILENAME)
TEST_FILEPATH = os.path.join(DATASETS_DIR, TEST_FILENAME)
TRAIN_PROCESSED_FILEPATH = os.path.join(OUTPUTS_DIR, TRAIN_PROCESSED_FILENAME)
TEST_PROCESSED_FILEPATH = os.path.join(OUTPUTS_DIR, TEST_PROCESSED_FILENAME)

