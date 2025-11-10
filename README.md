# 🔬 Machine Learning Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

A comprehensive, notebook-based machine learning laboratory for exploring algorithms, simulating datasets, and performing end-to-end experiments with professional visualization and analysis.

---

## 🌟 Overview

This project provides a professional yet flexible workflow to:

* 📥 **Load & Prepare Datasets** using publicly available datasets from scikit-learn, PyTorch, and other sources
* 📊 **Perform EDA** (Exploratory Data Analysis) and data preprocessing
* 🤖 **Train & Validate Models** using classical, statistical, and deep learning algorithms
* 📈 **Evaluate & Visualize** model performance with comprehensive metrics
* 📚 **Document Insights** in clear, reproducible Jupyter Notebooks

Each notebook is designed as a **self-contained learning module** with theoretical background, practical implementation, and result validation. Data simulation is used only when necessary to demonstrate specific concepts.

---

## 📊 Featured Notebooks

### 📈 1_Statistical_Analysis
> (More notebooks continuously being added)

### 🤖 2_Machine_Statistical_Learning

#### 📉 Linear Models
| Task | Algorithm | Notebook |
|------|-----------|----------|
| Classification | Logistic Regression | [🍷 Logistic Classification on Wine](notebooks/2_Machine_Statistical_Learning/linear_models/Logistic_Classification_on_Wine.ipynb) |
| Classification | Support Vector Machine (SVM) | [🌸 SVM Classification on Iris](notebooks/2_Machine_Statistical_Learning/linear_models/SVM_Classification_on_Iris.ipynb) |

#### 🌲 Tree Models
| Task | Algorithm | Notebook |
|------|-----------|----------|
| Classification | Random Forest | [🌿 Random Forest Classification on Wine](notebooks/2_Machine_Statistical_Learning/tree_models/Random_Forest_Classification_on_Wine.ipynb) |
| Regression | XGBoost | [🏗️ XGBoost Regression on California](notebooks/2_Machine_Statistical_Learning/tree_models/XGBoost_Regression_on_California.ipynb) |

### 🧠 3_Deep_Learning

#### 🖼️ CNN Models
| Task | Algorithm | Notebook |
|------|-----------|----------|
| Classification | Convolutional Neural Network (CNN) | [🔢 CNN Classification on MNIST](notebooks/3_Deep_Learning/2_cnn_models/CNN_Classification_on_MNIST.ipynb) |

#### 🔄 RNN Models
| Task | Algorithm | Notebook |
|------|-----------|----------|
| TS Forecasting | Long Short-Term Memory (LSTM) | [✈️ LSTM Forecasting on Airline](notebooks/3_Deep_Learning/3_rnn_models/LSTM_Forecasting_on_Airline.ipynb) |

#### 🎨 Generative Models
| Task | Algorithm | Notebook |
|------|-----------|----------|
| Image Generation | Generative Adversarial Network (GAN) | [🎭 GAN Generation on CIFAR-10](notebooks/3_Deep_Learning/5_generative_models/GAN_Gneration_on_CIFAR10.ipynb) |

---


## ⚙️ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/xlxsgit/machine-learning-lab.git
cd machine-learning-lab

# 2. Create and activate virtual environment (recommended)
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Lab
jupyter lab
```
**Note:** This project is developed with Python 3.12, but is generally compatible with other Python 3.x versions.

---

## 🚀 Usage

Navigate to the `notebooks/` directory and open any notebook using these recommended environments:

- [**JupyterLab**](https://jupyter.org/) - Run `jupyter lab` in terminal
- [**VS Code**](https://code.visualstudio.com/) - With Python and Jupyter extensions installed
- [**PyCharm**](https://www.jetbrains.com/pycharm/) - Professional Edition with built-in Jupyter support

Other compatible environments include Google Colab, Jupyter Notebook, and any IDE with Jupyter integration.

---

## 🛠️ Tech Stack

### Core Libraries
- [**Python**](https://www.python.org/) - Primary programming language
- [**scikit-learn**](https://scikit-learn.org/) - Machine learning algorithms  
- [**PyTorch**](https://pytorch.org/) - Deep learning framework

### Data Manipulation & Analysis
- [**Pandas**](https://pandas.pydata.org/) - Data structures and analysis
- [**NumPy**](https://numpy.org/) - Numerical computing

### Visualization
- [**Matplotlib**](https://matplotlib.org/) - Plotting and visualization
- [**Seaborn**](https://seaborn.pydata.org/) - Statistical data visualization

### Development Environment
- [**Jupyter Notebook/Lab**](https://jupyter.org/) - Interactive computing

---

## 📝 License

This project is licensed under the [MIT License](./LICENSE).
You are free to use, modify, and distribute it for educational or research purposes.

---

## ⭐ Support
If you find this project useful for your learning or research, please consider:

- Giving a Star ⭐ on GitHub to show your support
- Sharing with others who might benefit
- Contributing new algorithms and implementations

---