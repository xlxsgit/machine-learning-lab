<div align="center">

# 🔬 Machine Learning Lab

<!-- Badge Definitions -->
[contributors-image]: https://img.shields.io/github/contributors/xlxsgit/machine-learning-lab
[contributors-url]: https://github.com/xlxsgit/machine-learning-lab/graphs/contributors
[commit-image]: https://img.shields.io/github/last-commit/xlxsgit/machine-learning-lab
[commit-url]: https://github.com/xlxsgit/machine-learning-lab
[license-image]: https://img.shields.io/github/license/xlxsgit/machine-learning-lab.svg
[license-url]: https://github.com/xlxsgit/machine-learning-lab/blob/master/LICENSE

[python-image]: https://img.shields.io/badge/python-3.12+-blue.svg
[python-url]: https://www.python.org/downloads/
[jupyter-image]: https://img.shields.io/badge/Jupyter-Notebook-orange.svg
[jupyter-url]: https://jupyter.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[sklearn-image]: https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org/
[xgboost-image]: https://img.shields.io/badge/XGBoost-3776AB?style=flat&logo=xgboost&logoColor=white
[xgboost-url]: https://xgboost.ai/
[pandas-image]: https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white
[pandas-url]: https://pandas.pydata.org/
[numpy-image]: https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[matplotlib-image]: https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=matplotlib&logoColor=white
[matplotlib-url]: https://matplotlib.org/
[seaborn-image]: https://img.shields.io/badge/Seaborn-3776AB?style=flat
[seaborn-url]: https://seaborn.pydata.org/

<!-- Project Status Badges -->
<div align="center">

[![Contributors][contributors-image]][contributors-url]
[![Last Commit][commit-image]][commit-url]
[![License][license-image]][license-url]

</div>

<!-- Technology Stack Badges -->
<div align="center">

[![Python][python-image]][python-url]
[![Jupyter][jupyter-image]][jupyter-url]
[![PyTorch][pytorch-image]][pytorch-url]
[![Scikit-learn][sklearn-image]][sklearn-url]
[![XGBoost][xgboost-image]][xgboost-url]
[![Pandas][pandas-image]][pandas-url]
[![NumPy][numpy-image]][numpy-url]
[![Matplotlib][matplotlib-image]][matplotlib-url]
[![Seaborn][seaborn-image]][seaborn-url]

</div>

一个AI与机器学习资源库，将体系化地汇集数理基础、算法原理推导及面向实践的代码实现。

</div>

<!-- Table of Contents -->
<details>
  <summary>📋 目录</summary>

<!-- TOC -->
* [🔬 Machine Learning Lab](#-machine-learning-lab)
  * [📖 安装指南](#-安装指南)
  * [📐 数学基础](#-数学基础)
  * [✨ 算法锦集](#-算法锦集)
  * [📎 常用公开数据集](#-常用公开数据集)
  * [📝 许可证](#-许可证)
<!-- TOC -->
</details>

---

## 📖 安装指南
### 环境要求
- 🐍 **Python** 3.8+
- 📦 **pip** (Python 包管理器)
- 🔧 **Git**

### 安装步骤
1. **克隆仓库**
```bash
git clone https://github.com/xlxsgit/machine-learning-lab.git
cd machine-learning-lab
```
2. **创建并激活虚拟环境(可选)**

强烈建议使用虚拟环境以隔离依赖。
```bash
# 使用 venv 创建虚拟环境
python -m venv ml_env
# 使用 conda 创建虚拟环境
conda create -n ml_env python=3.12

# 在 Linux/macOS 上激活：
source ml_env/bin/activate
# 在 Windows 上激活：
ml_env\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

---

## 📐 数学基础
在开始机器学习之旅前，牢固掌握其背后的数学原理是至关重要的。本部分旨在系统性地整理和回顾机器学习领域所依赖的核心数学概念与工具，为后续的算法理解和实践打下坚实的理论根基。

- [线性代数](documents/foundation-of-mathematics/Linear-algebra.md)
- [微积分](documents/foundation-of-mathematics/Calculus.md)
- [数学优化](documents/foundation-of-mathematics/Mathematical-optimization.md)
- 概率论
- [信息论](documents/foundation-of-mathematics/Information-theory.md)

---

## ✨ 算法锦集
本部分是项目的核心，致力于系统地整理和实现各类经典的机器学习算法。从基础的线性模型到复杂的深度学习网络，我们将通过理论推导与代码实践相结合的方式，深入探索每种算法的原理、优缺点及其适用场景。

| 类别 | 分支 | 算法/方法 | Notebook | 理论                                                                                   |
|------|------|-----------|----------|--------------------------------------------------------------------------------------|
| 📈 统计 | ⚓ 假设检验 | 方差分析 | [ANOVA-on-Iris](notebooks/statistics/hypothesis-testing/ANOVA-on-Iris.ipynb) | [Analysis-of-Variance](documents/theoretical-derivation/Analysis-of-Variance.md) |
|      | 🎲 抽样方法 | 逆变换抽样 | [Inverse-Transform-Sampling](notebooks/statistics/sampling/Inverse-Transform-Sampling.ipynb) |                                                                                      |
| 🤖 机器学习 | 〰️ 线性模型 | 逻辑回归 | [Logistic-on-Wine](notebooks/machine-learning/linear/Logistic-on-Wine.ipynb) |                                                                                      |
|      |      | 支持向量机 |  | [Support-Vector-Machine](documents/theoretical-derivation/Support-Vector-Machine.md) |
|      | 🌴 树模型 | 决策树 | [DecisionTree-on-Iris](notebooks/machine-learning/trees/DecisionTree-on-Iris.ipynb) |                                                                                      |
|      |      | XGBoost | [XGBoost-on-Iris](notebooks/machine-learning/trees/Xgboost-on-Iris.ipynb) |                                                                                      |
|      | 🧩 无监督方法 | K均值 | [KMeans-on-Synthetic](notebooks/machine-learning/unsupervised/KMeans-on-Synthetic.ipynb) |                                                                                      |
|      |      | DBSCAN | [DBSCAN-on-Synthetic](notebooks/machine-learning/unsupervised/DBSCAN-on-Synthetic.ipynb) |                                                                                      |
| 🧠 深度学习 | 💠 卷积神经网络 | CNN | [CNN-on-MNIST](notebooks/deep-learning/cnn/CNN-on-MNIST.ipynb) |                                                                                      |  |
|      | ♻️ 循环神经网络 |  |  |                                                                                      |
|      | 🎨 生成模型 |  |  |                                                                                      |

---

## 📎 常用公开数据集
高质量的数据是机器学习研究与实践的基石。本部分汇集了在学术界和工业界广泛使用的公开数据集，涵盖了回归、分类、聚类等多种任务类型，为模型的训练、验证和测试提供了标准化的基准。

| 类型 | 库                      | API                        | 描述                 |
|----|------------------------|----------------------------|--------------------|
| 回归 | `sklearn.datasets`     | `make_regression`          | 模拟数据               |
|    |                        | `load_diabetes`            | Diabetes           |
|    |                        | `fetch_california_housing` | California Housing |
| 分类 | `sklearn.datasets`     | `make_classification`      | 模拟数据          |
|    |                        | `load_iris`                | Iris               |
|    |                        | `load_wine`                | Wine Quality       |
|    |                        | `load_digits`              | Digits             |
|    |                        | `load_breast_cancer`       | Breast Cancer      |
|    | `torchvision.datasets` | `MNIST`                    | Handwritten Digits |
|    |                        | `CIFAR-10`                 | Object Recognition |
| 聚类 | `sklearn.datasets`     | `make_blobs`               | 模拟数据          | |
|    |                        | `make_circles`             | 模拟数据          |

---

## 📝 许可证
本项目采用 [**MIT**](./LICENSE) 许可证。允许为教育或研究目的而自由使用、修改和分发。

---
