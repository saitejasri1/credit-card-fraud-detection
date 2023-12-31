🕵️‍♂️ Fraud Detection in Credit Card Transactions 💳

Welcome to our repository dedicated to detecting fraudulent activities in credit card transactions using advanced machine learning techniques. This project applies various Gradient Boosting algorithms to identify fraud cases from a highly imbalanced dataset.

📚 Dataset

The dataset includes transactions made by credit cards, where each transaction is labeled as fraudulent or legitimate. Key features include:

- Time of transactions
- Transaction Amount
- 28 anonymized features (V1 to V28)
- Class (1: Fraud, 0: No Fraud)
- 🧰 Tools and Libraries

Python 🐍
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for data visualization
- Scikit-learn for model building and evaluation
- Imbalanced-learn for handling imbalanced data
- XGBoost, LightGBM, CatBoost, and other boosting algorithms

🔍 Exploratory Data Analysis

- Analysis of transaction amounts and time distribution
- Visualization of fraud vs. no fraud transactions
- Correlation analysis using heatmaps

📉 Visualizations

- Transaction Class Distribution: A bar chart showing the distribution of fraudulent and non-fraudulent transactions.
- Time and Amount Distributions: Histograms and scatter plots for transaction time and amount, divided by class.
- Correlation Heatmap: A heatmap to visualize the correlation between different features. 

🤖 Model Building and Evaluation

- Feature scaling and data resampling for balanced dataset
- Model training using Gradient Boosting algorithms like XGBoost, LightGBM, CatBoost
- Evaluation using classification report, confusion matrix, and accuracy scores

🚀 How to Run

Clone the repository
Install dependencies
Run the Jupyter notebooks or Python scripts
