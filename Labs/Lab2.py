import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Lab 2", layout="centered")
st.title("Lab 2: Document Q&A")

client = OpenAI()

summary_type = st.sidebar.radio(
    "Summary type",
    [
        "100 words",
        "2 connecting paragraphs",
        "5 bullet points",
    ],
)

use_advanced = st.sidebar.checkbox("Use advanced model")

model = "gpt-4.1-nano"
if use_advanced:
    model = "gpt-4.1-mini"


def summary_instruction(choice):
    if choice == "100 words":
        return "Summarize the document in about 100 words."
    elif choice == "2 connecting paragraphs":
        return "Summarize the document in 2 connecting paragraphs."
    else:
        return "Summarize the document in exactly 5 bullet points."

uploaded_file = st.file_uploader("Upload a document (.txt)", type=("txt",))

question = st.text_area(
    "Optional question (leave blank to just summarize)",
    disabled=not uploaded_file,
)

if uploaded_file and st.button("Run"):
    document = uploaded_file.read().decode("utf-8", errors="ignore")

    prompt = summary_instruction(summary_type)
    if question.strip():
        prompt += f"\n\nThen answer this question: {question}"

    messages = [
        {
            "role": "user",
            "content": f"Here is the document:\n\n{document}\n\n{prompt}",
        }
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    st.write_stream(stream)


