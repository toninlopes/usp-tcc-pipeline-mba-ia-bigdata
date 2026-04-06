import sys
import os

# Garante que a raiz do projeto esteja no sys.path independente de onde o
# streamlit é invocado — necessário para resolver `shared.*` e `pipeline.*`
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

# Define your pages
pg1 = st.Page("../03_eda/dashboard.py", title="Analytics", icon="📊")
pg2 = st.Page("classifier.py", title="Classification", icon="🏷️")
pg3 = st.Page("../03_eda/eda.py", title="Exploração dos Dados", icon="🔍")

# Initialize navigation
pg = st.navigation([pg1, pg2, pg3])
pg.run()
