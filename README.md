# HousePricesPrediction（学习版）

这是一个 **Kaggle 房价预测练习项目**。目标是：
1. 会做 [`EDA`](src/eda.py)
2. 会做 [`FeatureEngineer`](src/featureEngineering.py:27)
3. 会把流程跑通并生成提交文件

---

## 推荐项目结构（简单够用）

```text
HousePricesPrediction/
├── README.md
├── requirements.txt
├── config.py
├── .gitignore
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── outputs/
│   ├── eda/
│   ├── featureEngineering/
│   ├── models/
│   └── submissions/
│
└── src/
    ├── eda.py
    ├── featureEngineering.py
    ├── train.py
    └── predict.py
```

---

## 每个文件干什么（学习视角）

- [`config.py`](config.py)
  - 放路径和常量（数据路径、输出路径、随机种子）。

- [`src/eda.py`](src/eda.py)
  - 看数据分布、缺失、相关性、异常值。
  - 输出图到 [`outputs/eda/`](outputs/eda)。

- [`src/featureEngineering.py`](src/featureEngineering.py)
  - 缺失值处理、类别编码、构造特征。
  - 输出处理后的数据到 [`data/processed/`](data/processed)。

- [`src/train.py`](src/train.py)
  - 读处理后训练集，训练模型并保存到 [`outputs/models/`](outputs/models)。

- [`src/predict.py`](src/predict.py)
  - 读模型 + 测试集，输出提交文件到 [`outputs/submissions/`](outputs/submissions)。

---

## 你现在的学习顺序（建议）

1. 跑 [`python -m src.eda`](src/eda.py)
2. 跑 [`python -m src.featureEngineering`](src/featureEngineering.py)
3. 再写并跑 [`python -m src.train`](src/train.py)
4. 最后写并跑 [`python -m src.predict`](src/predict.py)

---

## 进阶时再加的东西（先不用急）

- [`notebooks/`](notebooks)：实验记录
- [`tests/`](tests)：单元测试
- [`.github/workflows/`](.github/workflows)：自动化检查

先把主流程跑通，再补这些最稳。
