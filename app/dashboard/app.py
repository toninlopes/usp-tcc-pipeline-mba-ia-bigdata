import sys
import os

# 'app.py' conflicts with the 'app/' package: any sys.path entry that resolves
# to app/dashboard/ (including '' from PYTHONPATH=.) would let Python find
# app.py before app/__init__.py. Fix: strip dashboard dir and any existing
# project-root entry, then pin project root at position 0.
_dashboard_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_dashboard_dir))
sys.path = [p for p in sys.path if p not in (_dashboard_dir, _project_root)]
sys.path.insert(0, _project_root)

import streamlit as st

pg_annotation = st.Page("pages/annotation.py", title="Anotação", icon="🏷️")
pg_eda = st.Page("pages/eda.py", title="Analytics", icon="📊")
pg_exploration = st.Page("pages/exploration.py", title="Exploração dos Dados", icon="🔍")
pg_preprocessing = st.Page("pages/preprocessing.py", title="Pré-processamento", icon="🔬")
pg_split = st.Page("pages/dataset_split.py", title="Split do Dataset", icon="✂️")
pg_fine_tuning = st.Page("pages/fine_tuning.py", title="Fine-tuning BERTimbau", icon="🧠")
pg_processing = st.Page("pages/processing.py", title="Processamento", icon="🤖")
pg_evaluation = st.Page("pages/evaluation.py", title="Avaliação", icon="📈")

pg = st.navigation([
    pg_annotation,
    pg_eda,
    pg_exploration,
    pg_preprocessing,
    pg_split,
    pg_fine_tuning,
    pg_processing,
    pg_evaluation,
])
pg.run()