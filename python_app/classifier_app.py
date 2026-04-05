import streamlit as st

# Define your pages
pg1 = st.Page("classifier/dashboard.py", title="Analytics", icon="📊")
pg2 = st.Page("classifier/classifier.py", title="Classification", icon="🏷️")
pg3 = st.Page("classifier/eda.py", title="Exploração dos Dados", icon="🔍")

# Initialize navigation
pg = st.navigation([pg1, pg2, pg3])
pg.run()
