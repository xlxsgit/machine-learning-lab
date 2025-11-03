# 🧪 Machine Learning Lab

A notebook-based machine learning lab for exploring algorithms, simulating datasets, and performing end-to-end experiments with visualization and analysis.

This repository serves as a modular learning framework — each notebook focuses on a specific **task** (e.g., classification, regression, clustering) and **algorithm** (e.g., Random Forest, K-Means, PCA), implemented using Python and `scikit-learn`.

---

## 🌟 Overview

This project provides a professional yet flexible workflow to:

* Generate or simulate datasets directly with `scikit-learn` or `NumPy`
* Perform EDA (Exploratory Data Analysis) and preprocessing
* Train models using classical or statistical algorithms
* Evaluate results and visualize model performance
* Present the whole process in clear Jupyter Notebooks

Each notebook is designed as a **self-contained learning demo**.

---

## 📁 Project Structure

```
machine-learning-lab/
│
├── data/               # Generated or example datasets
├── models/             # Saved model files (optional)
├── notebooks/          # Jupyter notebooks for each algorithm/task
├── requirements.txt    # Required dependencies
├── README.md           # Project documentation
└── LICENSE             # License file (MIT)
```

*(The notebooks folder will be continuously updated with new algorithm demos.)*

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/machine-learning-lab.git
cd machine-learning-lab

# 2. Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

You can run any notebook directly using JupyterLab or VSCode:

```bash
jupyter lab
```

Then open one of the notebooks under the `notebooks/` directory.

Each notebook contains:

* Data simulation and preprocessing
* Model training
* Evaluation and visualization
* Professional comments and insights

---

## 🧠 Algorithms Included

### 📊 Statistics
* [ANOVA Analysis (One-Way and Two-Way)](notebooks/Statistics/ANOVA_Analysis.ipynb)
* [Statistical Hypothesis Testing (Single and Two Normal Population Mean and Variance Tests)](notebooks/Statistics/Statistical_Hypothesis_Testing.ipynb)

### 🤖 Machine Learning
* **Classification:** [Logistic Regression](notebooks/ML/Classification_Logistic_Regression.ipynb), [SVM](notebooks/ML/Classification_SVM.ipynb)
* **Regression:** [Linear Regression](notebooks/ML/Regression_Linear_Regression.ipynb)
* **Clustering:** [K-Means](notebooks/ML/Clustering_KMeans.ipynb)

### 🔄 Ensemble Methods
* **Classification:** [AdaBoost](notebooks/Ensemble/Classification_AdaBoost.ipynb)
* **Regression:** [Random Forest](notebooks/Ensemble/Regression_Random_Forest.ipynb), [XGBoost](notebooks/Ensemble/Regression_XGBoost.ipynb)

### 🧠 Deep Learning
* **Neural Networks:** [MLP (Multi-Layer Perceptron)](notebooks/DL/Regression_MLP.ipynb)
* **Computer Vision:** [CNN for image classification](notebooks/DL/Classification_CNN.ipynb), [GAN for image generation](notebooks/DL/Classification_GAN.ipynb)
* **Time Series:** [LSTM for forecasting](notebooks/DL/TS Forecasting_LSTM.ipynb)
* **NLP:** [Transformer models](notebooks/DL/NLP_Transformer.ipynb)

---
## Built with:

* [Python](https://www.python.org/)
* [scikit-learn](https://scikit-learn.org/)
* [PyTorch](https://pytorch.org/)
* [Pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

---

## 📝 License

This project is licensed under the [MIT License](./LICENSE).
You are free to use, modify, and distribute it for educational or research purposes.

---

⭐ **If you find this project useful, consider giving it a star on GitHub!**
