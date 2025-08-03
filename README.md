# Insider Attack Detection from Logon/Logoff Events Using Machine Learning

A machine learning–based case study to detect **insider threats** through the analysis of user logon/logoff activity patterns. This project compares various classification and anomaly detection algorithms trained on structured behavioral data. Developed as part of the CS F266 Study Project at BITS Pilani, Dubai Campus.

---

##  Objective

Insider threats are among the most difficult cybersecurity problems to detect, as they originate from users with legitimate access. By analyzing logon/logoff behaviors, we aim to identify patterns that distinguish normal activity from potential threats.

This project explores machine learning models to:
- Detect anomalies and attacks based on user session behaviors
- Compare multiple algorithms in terms of accuracy and performance
- Build a reusable, modular toolkit for cybersecurity threat analysis

---

##  Project Highlights

- Developed 8 machine learning models targeting insider threat detection
- Applied both classification and anomaly detection techniques
- Preprocessed behavioral data from logon/logoff sessions
- Comparative study using metrics like accuracy, precision, and recall
- Modular code structure for easy testing and extension
- Demonstrates practical applications of supervised and unsupervised learning in cybersecurity

---

##  Files Included

| File                          | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| `LOGON1.csv`                  | Simulated dataset of user logon/logoff events                |
| `Decision-Trees.py`           | Decision Tree classifier implementation                      |
| `Isolation-Forest.py`         | Anomaly detection using Isolation Forest                     |
| `KNN.py`                      | K-Nearest Neighbors classification model                     |
| `Logistic-Regression.py`      | Logistic Regression model for binary classification          |
| `Naive-Bayes-Classifier.py`   | Naïve Bayes model for logon/logoff analysis                  |
| `Random-Forest-Classifier.py` | Random Forest ensemble model for insider threat detection    |
| `SVM.py`                      | Support Vector Machine classifier                            |
| `XGBoost.py`                  | Gradient-boosted trees for improved performance              |

---

##  Problem Context

This project explores machine learning models to:
- Detect anomalies and attacks based on user session behaviors
- Compare multiple algorithms in terms of accuracy and performance
- Build a reusable, modular toolkit for cybersecurity threat analysis

---

##  Dataset Description

**LOGON1.csv** contains simulated user session activity, including:
- User IDs  
- Logon timestamps  
- Logoff timestamps  
- Session durations  
- Labels for normal or malicious sessions

All models read from this dataset to train and validate their predictions.

---

##  Evaluation Metrics

Each model prints and optionally visualizes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification reports

---

##  Future Scope

- Expand dataset to include file access, system commands, and privilege changes
- Add time-based features to detect slow and long-term threats
- Implement deep learning models such as LSTM for time-series user behavior
- Deploy as a web-based dashboard for real-time alerts and visualization
- Integrate with live system logs or SIEM platforms for operational use

---

## ❗ Usage Terms

**This repository is for academic and demonstration purposes only.  
Reproduction, reuse, or redistribution of this work without permission is not allowed.**

##  How to Run

### 1. Clone this repository

```bash
git clone https://github.com/yourusername/insider-threat-ml-detector.git
cd insider-threat-ml-detector
