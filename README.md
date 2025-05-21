Setup Instructions
Clone the Repository:
  git clone https://github.com/meronheron/causalInference.git
  cd causalInference
Create and Activate Virtual Environment:
  python -m venv causal_env
  .\causal_env\Scripts\activate  # Windows
  source causal_env/bin/activate  # Linux/Mac
Install Dependencies:
  pip install pandas numpy causalml scikit-learn seaborn matplotlib streamlit openpyxl
run separate py codes if needed(dml_inference.py and psm_inference.py):
  python psm_inference.py
  python dml_inference.py
Run the Streamlit UI:
  streamlit run causal_ui.py
