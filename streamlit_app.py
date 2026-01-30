import streamlit as st

st.set_page_config(page_title="IST 488 Labs", layout="centered")

st.title("IST 488 Labs")
st.write("Use the sidebar to navigate between Lab 1 and Lab 2.")

# Create navigation pages
lab1 = st.Page("Labs/Lab1.py", title="Lab 1")
lab2 = st.Page("Labs/Lab2.py", title="Lab 2")

pg = st.navigation([lab1, lab2])
pg.run()



