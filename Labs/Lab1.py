import streamlit as st
from openai import OpenAI

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)


# Lab 2 page
st.set_page_config(page_title="Lab 2", layout="centered")
st.title("Lab 2: Document Q&A")

# IMPORTANT (Part B):
# We are NOT asking the user for an API key in the UI anymore.
# The key should be stored as an environment variable / secret.
# OpenAI() will automatically read OPENAI_API_KEY from the environment.
client = OpenAI()

st.write(
    "Upload a document and ask a question about it.\n\n"
    "Note: The API key is loaded from the server environment (not typed in here)."
)

uploaded_file = st.file_uploader("Upload a document (.txt)", type=("txt",))

question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:
    document = uploaded_file.read().decode("utf-8", errors="ignore")

    messages = [
        {
            "role": "user",
            "content": f"Here's a document:\n\n{document}\n\n---\n\n{question}",
        }
    ]

    stream = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=messages,
        stream=True,
    )

    st.write_stream(stream)
