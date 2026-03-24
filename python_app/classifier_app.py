import streamlit as st

# Define your pages
pg1 = st.Page("classifier/dashboard.py", title="Analytics", icon="📊")
pg2 = st.Page("classifier/classifier.py", title="Classification", icon="🏷️")

# Initialize navigation
pg = st.navigation([pg1, pg2])
pg.run()
