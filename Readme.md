# 🤖 Generative Artificial Intelligence — Lecture Notes and Applied Notebooks

**Haydar Kilic | Faculty of Engineering, Artificial Intelligence Engineering**

This repository contains Jupyter Notebooks that reinforce the theoretical lecture slides of the *Generative Artificial Intelligence* course using Python. Each notebook derives the formulas discussed in class from first principles, visualizes them, and applies them to real-world data scenarios.

---

## 📚 Contents

| Lecture   | Topic                               | Notebook                     |
| --------- | ----------------------------------- | ---------------------------- |
| Lecture 1 | Fundamentals of Generative Modeling | [`GAI_Lecture1_Notebook.ipynb`] |

> The table will be updated as new lectures are added.

---

## 🗂 Lecture 1 — Fundamentals of Generative Modeling

### Topics Covered

**Section 1 — Fundamental Concepts**

* Handwritten digit recognition: 28×28 pixel vector representation, training/test/validation split
* Polynomial regression and curve fitting (Vandermonde matrix, Least Squares)
* Overfitting / Underfitting and RMS error analysis
* Ridge Regularization (L2 penalty, λ hyperparameter)

**Section 2 — Probability Theory**

* Joint, marginal, and conditional probability distributions
* Bayes’ theorem — medical diagnosis and the *base rate fallacy*
* Gaussian (Normal) distribution: PDF, CDF, numerical verification
* Maximum Likelihood Estimation (MLE) and bias
* Bayesian updating: prior → posterior using coin toss experiments

**Section 3 — Decision Theory**

* Minimum-error decision boundaries and posterior probabilities
* Reject Option and threshold θ
* Asymmetric loss matrix (medical diagnosis scenario)
* Comparison of generative, discriminative, and discriminant models

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/HAYDARKILIC/generative_artificial_intelligence
cd generative_artificial_intelligence

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

---

## 📦 Requirements

```
numpy>=1.24
matplotlib>=3.5
scipy>=1.9
scikit-learn>=1.0
jupyter>=1.0
```

> The `requirements.txt` file is included in the repository.

---

## 🏗 Project Structure

```
uretken-yapay-zeka/
├── README.md
├── requirements.txt
├── GAI_Lecture1_Notebook.ipynb   # Lecture 1 — Fundamental Concepts
└── (notebooks for future lectures will be added)
```

---


*Generative Artificial Intelligence — Haydar Kılıç, Artificial Intelligence Engineering*
