# UK-Listed FinTech Dashboard 
An interactive Streamlit dashboard for UK-listed fintech companies; 
Pulling and visualising financial KPIs from Yahoo Finance.

# Requirment
Python 3.10+  

# Streamlit dashboard
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uk-fintech-app-dashboard.streamlit.app/)

The dashboard visualises revenue, net margin and leverage (debt-to-equity) trends for UK-listed fintech companies.

# Run Locally
Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/A1brahim/uk-fintech-streamlit-dashboard.git
cd uk-fintech-streamlit-dashboard

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py

-- -- -- -- -- 
Your browser will open at http://localhost:8501.
