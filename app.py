# --- SQLite shim (only for Linux/Mac; safe on Streamlit Cloud) ---
import sys
if sys.platform != "win32":
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except Exception:
        # If not available, Chroma may still work with system sqlite
        pass

# --- Std imports ---
import os
import uuid
import streamlit as st
import dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

# -------------------- Constants --------------------
MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20240620",
]

st.set_page_config(
    page_title="RAG LLM app",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------- Header -----------------------
st.markdown(
    """
    <h2 style="text-align:center;">🧠📚🔍 <i>Do your LLM even RAG bro?</i> 🤖💬</h2>
    """,
    unsafe_allow_html=True,
)

# -------------------- Session init -----------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

# -------------------- Sidebar: API keys ------------
with st.sidebar:
    default_openai = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    default_anthropic = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")

    with st.popover("🔑 OpenAI"):
        openai_api_key = st.text_input(
            "Introduce your OpenAI API Key (https://platform.openai.com/)",
            value=default_openai,
            type="password",
            key="openai_api_key",
        )

    with st.popover("🔑 Anthropic"):
        anthropic_api_key = st.text_input(
            "Introduce your Anthropic API Key (https://console.anthropic.com/)",
            value=default_anthropic,
            type="password",
            key="anthropic_api_key",
        )

# Figure out which models the user can actually use (based on keys provided)
missing_openai = not bool(openai_api_key)
missing_anthropic = not bool(anthropic_api_key)

available_models = [
    m for m in MODELS
    if (m.startswith("openai/") and not missing_openai)
    or (m.startswith("anthropic/") and not missing_anthropic)
]

if not available_models:
    st.warning("🔑 Please add at least one API key in the sidebar to continue.")
    st.stop()

# -------------------- Sidebar: Model + RAG controls -------------
with st.sidebar:
    st.divider()
    st.selectbox("🤖 Select a Model", options=available_models, key="model")

    # Is vector DB available?
    is_vector_db_loaded = ("vector_db" in st.session_state) and (st.session_state.vector_db is not None)

    cols = st.columns(2)
    with cols[0]:
        st.toggle(
            "Use RAG",
            value=is_vector_db_loaded,
            key="use_rag",
            disabled=not is_vector_db_loaded,
        )
    with cols[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    st.header("📚 RAG Sources")

    st.file_uploader(
        label="📂 Upload a document",
        type=["pdf", "txt", "docx", "doc", "json", "md", "pptx", "ppt"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )

    st.text_input(
        label="🌐 Introduce a URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )

    # Show what’s loaded (guard against None)
    try:
        count = 0 if not is_vector_db_loaded else len(st.session_state.rag_sources)
        with st.expander(f"📄 Documents in DB ({count})"):
            if is_vector_db_loaded:
                metadatas = st.session_state.vector_db.get().get("metadatas", [])
                st.write(metadatas)
            else:
                st.write([])
    except Exception:
        with st.expander("📄 Documents in DB (0)"):
            st.write([])

# -------------------- Build the LLM client safely ---------------
model_str = st.session_state.get("model")
if not model_str:
    st.error("No model selected. Add an API key and pick a model in the sidebar.")
    st.stop()

provider = model_str.split("/")[0]
model_name = model_str.split("/")[-1]

if provider == "openai":
    llm_stream = ChatOpenAI(
        api_key=openai_api_key,
        model_name=model_name,
        temperature=0.3,
        streaming=True,
    )
elif provider == "anthropic":
    llm_stream = ChatAnthropic(
        api_key=anthropic_api_key,
        model_name=model_name,
        temperature=0.3,
        streaming=True,
    )
else:
    st.error(f"Unsupported provider: {provider}")
    st.stop()

# -------------------- Render chat history -----------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------- Handle new input --------------------------
if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Convert to LangChain message objects
        lc_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user"
            else AIMessage(content=msg["content"])
            for msg in st.session_state.messages
        ]

        if st.session_state.use_rag:
            st.write_stream(stream_llm_rag_response(llm_stream, lc_messages))
        else:
            st.write_stream(stream_llm_response(llm_stream, lc_messages))
