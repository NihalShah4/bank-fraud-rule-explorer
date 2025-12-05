🕵️ Bank Fraud Rule Explorer
A complete end-to-end fraud analytics project demonstrating anomaly detection, feature engineering, rule exploration, and an interactive Streamlit dashboard.


⭐ Why this project matters
Banks, fintechs, and payment companies rely heavily on anomaly detection and rule-based monitoring to detect fraud, money laundering, structuring, and unusual customer behavior.

This project mimics a real fraud investigation workflow, making it perfect to showcase your skills for:
Fraud Analyst
AML Analyst / FIU Investigator
Risk Analyst
Data Analyst / Data Scientist
Fintech roles (Cash App, Stripe, Brex, Revolut, etc.)

🚀 Features
✔ Synthetic dataset of 15,000 realistic transactions
Includes:
country, merchant category, channel
timestamps
transaction velocity
embedded fraud behaviors (bursting, high-risk MCCs, foreign-online-crypto, extreme amounts)

✔ Advanced feature engineering
Time-based features
Customer-level aggregates
Log transforms
One-hot encoding

✔ Anomaly detection using Isolation Forest
Model outputs:
Anomaly score
Fraud likelihood flags
Recall / precision / hit-rate metrics

✔ Interactive Streamlit dashboard
Explore suspicious transactions
Filter by country, merchant, channel
Adjust anomaly score threshold
KPI metrics (recall, precision, fraud rate)
Visual patterns: flag rate by merchant & country

🧱 Tech Stack
Python
Pandas / NumPy
Scikit-learn
Streamlit
Matplotlib
GitHub Desktop

📂 Project Structure
bank-fraud-rule-explorer/
│
├── data/
│   ├── sample_transactions.csv
│   └── transactions_with_scores.csv
│
├── src/
│   ├── generate_synthetic_data.py
│   ├── preprocessing.py
│   ├── anomaly_model.py
│   └── metrics.py (optional extension)
│
├── app/
│   └── streamlit_app.py
│
├── venv/ (ignored)
├── README.md
└── requirements.txt

▶ How to Run
1. Create environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. Generate synthetic data
python src/generate_synthetic_data.py

3. Train anomaly model
python src/anomaly_model.py

4. Launch dashboard
streamlit run app/streamlit_app.py

📊 Example Dashboard
<img width="2560" height="1600" alt="Screenshot 2025-12-05 015413" src="https://github.com/user-attachments/assets/fbcc1118-518d-484d-8658-21d92c7baf52" />

🧩 Future Enhancements
Auto-generating fraud rules using LLM (GPT)
SAR-like case narrative generator
Customer-level risk scoring
Fraud ring detection using graph analytics
KPI benchmarking (false positive reduction)

👤 Author
Nihal Shah
Data Science & Fraud Analytics
New York, USA
GitHub: https://github.com/NihalShah4
