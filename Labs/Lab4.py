import sys

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "pysqlite3 is missing. Make sure 'pysqlite3-binary' is in requirements.txt."
    )


import streamlit as st
from openai import OpenAI
import time
from pathlib import Path
from PyPDF2 import PdfReader
import chromadb


MODEL_NAME = "gpt-4o-mini"   # you can change to "gpt-5-mini" if you want
MAX_TOKENS = 800  # token budget for the input buffer (rough estimate)

st.title("Lab 4 – RAG Pipeline with Vector DB (ChromaDB)")

# -----------------------------
# Token helpers Lab 3

def rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def rough_tokens_messages(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += rough_tokens(m.get("role", ""))
        total += rough_tokens(m.get("content", ""))
    return total

def build_token_buffer(all_messages: list[dict], max_tokens: int) -> list[dict]:
    if not all_messages:
        return []

    system_msg = all_messages[0]  # keep system prompt
    kept = [system_msg]
    used = rough_tokens_messages(kept)

    for msg in reversed(all_messages[1:]):
        msg_tokens = rough_tokens_messages([msg])
        if used + msg_tokens > max_tokens:
            break
        kept.insert(1, msg)
        used += msg_tokens

    return kept


# OpenAI client
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Messages init (keep your style)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful chatbot. Explain things so a 10-year-old can understand. "
                "Be clear, simple, and friendly.\n\n"
                "If you use information from retrieved course documents, say so briefly."
            ),
        }
    ]

if "waiting_for_more_info" not in st.session_state:
    st.session_state.waiting_for_more_info = False


# RAG / ChromaDB functions
DATA_FOLDER = "Labs/Lab4-Data"
CHROMA_PATH = "./ChromaDB_for_Lab4"
COLLECTION_NAME = "Lab4Collection"
EMBED_MODEL = "text-embedding-3-small"

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def embed_text(text: str) -> list[float]:
    # NOTE: embedding model has limits; for class docs this is usually fine.
    resp = st.session_state.client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding

def create_or_load_vectordb():
    """
    Creates a persistent ChromaDB collection and loads PDFs into it.
    Stored in st.session_state so we only build embeddings once.
    """
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    # If it's empty, load PDFs (first run)
    existing_count = collection.count()
    if existing_count == 0:
        pdf_dir = Path(DATA_FOLDER)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if not pdf_files:
            st.error(
                f"No PDFs found in {DATA_FOLDER}. Create the folder and add the 7 provided PDFs."
            )
            return collection

        with st.spinner("Loading PDFs + generating embeddings (first-time setup)..."):
            for pdf in pdf_files:
                file_name = pdf.name
                text = extract_text_from_pdf(str(pdf))

                # Embed and store (ID = filename)
                emb = embed_text(text)

                collection.add(
                    ids=[file_name],
                    documents=[text],
                    embeddings=[emb],
                    metadatas=[{"source": file_name}]
                )

        st.success("Vector DB built successfully!")
    return collection

def retrieve_context(query: str, k: int = 3) -> tuple[str, list[str]]:
    """
    Returns (context_text, source_ids)
    """
    q_emb = embed_text(query)
    results = st.session_state.Lab4_VectorDB.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    docs = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    context = "\n\n---\n\n".join(docs) if docs else ""
    return context, ids

# Sidebar controls

st.sidebar.header("Lab 4 Controls")

# Build vector DB once
if "Lab4_VectorDB" not in st.session_state:
    st.session_state.Lab4_VectorDB = create_or_load_vectordb()

rebuild = st.sidebar.button("Rebuild Vector DB (costs embeddings)")
if rebuild:
    # Danger: deletes local persisted DB folder and rebuilds
    import shutil
    if Path(CHROMA_PATH).exists():
        shutil.rmtree(CHROMA_PATH)
    st.session_state.Lab4_VectorDB = create_or_load_vectordb()

test_mode = st.sidebar.checkbox("Part A: Test Retrieval Mode", value=False)


# Display chat history (your Lab 3)
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# Part A test mode 

if test_mode:
    st.subheader("Vector DB Test (Part A)")
    q = st.text_input("Test search (e.g., Generative AI / Text Mining / Data Science Overview)")
    if q:
        _, ids = retrieve_context(q, k=3)
        st.write("Top 3 returned documents (filenames):")
        for i, doc_id in enumerate(ids, start=1):
            st.write(f"{i}. {doc_id}")

    st.info("When done testing, uncheck this box and use the chatbot below.")
    st.stop()


# Chat input (your Lab 3 + RAG)
prompt = st.chat_input("Ask a course question...")

if prompt:
    user_text = prompt.strip()
    user_lower = user_text.lower()

    if st.session_state.waiting_for_more_info:
        if user_lower in ["yes", "y"]:
            st.session_state.messages.append(
                {"role": "user", "content": "Yes, please give me more info."}
            )
        elif user_lower in ["no", "n"]:
            st.session_state.waiting_for_more_info = False
            msg = "Okay! What can I help you with?"
            with st.chat_message("assistant"):
                st.write(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()
        else:
            msg = "Please type Yes or No. Do you want more info?"
            with st.chat_message("assistant"):
                st.write(msg)
            st.stop()
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

    # --- RAG retrieval ---
    context, sources = retrieve_context(user_text, k=3)

    # Build message buffer like your Lab 3
    messages_for_model = build_token_buffer(st.session_state.messages, MAX_TOKENS)

    # Inject retrieved context as an EXTRA system message (doesn't permanently pollute history)
    rag_system = {
        "role": "system",
        "content": (
            "You have access to retrieved course document context below.\n"
            "Use it to answer the user's question. If the context is not relevant or missing, say so.\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"SOURCES (filenames): {', '.join(sources) if sources else 'None'}"
        )
    }

    # Put the RAG system message right after the original system message
    if messages_for_model and messages_for_model[0]["role"] == "system":
        messages_for_model = [messages_for_model[0], rag_system] + messages_for_model[1:]
    else:
        messages_for_model = [rag_system] + messages_for_model

    tokens_sent = rough_tokens_messages(messages_for_model)
    st.sidebar.write(f"Estimated tokens sent: {tokens_sent} / {MAX_TOKENS}")

    completion = st.session_state.client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages_for_model,
        temperature=0,
    )
    reply = completion.choices[0].message.content

    # Add a quick RAG transparency line + your Part C loop question
    if sources:
        reply = reply + f"\n\n(Used RAG sources: {', '.join(sources)})"

    reply = reply + "\n\nDo you want more info?"

    def stream_text(text: str):
        for ch in text:
            yield ch
            time.sleep(0.01)

    with st.chat_message("assistant"):
        st.write_stream(stream_text(reply))

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.waiting_for_more_info = True
