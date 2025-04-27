import streamlit as st
import requests

st.title("Shopping Assistant - RAG Based")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask your shopping query:")

if st.button("Ask"):
    if query:
        response = requests.post(
            "http://your-ec2-ip:8000/query",
            params={"question": query}
        )
        answer = response.json()["response"]
        st.session_state.history.append((query, answer))

for user_query, bot_response in st.session_state.history:
    st.markdown(f"**You:** {user_query}")
    st.markdown(f"**Assistant:** {bot_response}")

