import streamlit as st
from openai import OpenAI
# Show title and description.
st.title("MY Document question answering")
st.write(
"Upload a document below and ask a question about it – GPT will answer! "
"To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)
# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
st.info("Please add your OpenAI API key to continue.", icon="#")
st.stop()
else:
try:
client = OpenAI(api_key=openai_api_key)
client.models.list()
st.success("API key is valid!")
except Exception:
st.error("Invalid or blocked API key. Please check it and try again.")
st.stop()
# Let the user upload a file
uploaded_file = st.file_uploader(
"Upload a document (.txt or .md)", type=("txt", "md")
)
# Ask for a question
question = st.text_area(
"Now ask a question about the document!",
placeholder="Can you give me a short summary?",
disabled=not uploaded_file,
)
# Only proceed if both are provided
if uploaded_file and question:
document = uploaded_file.read().decode()
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