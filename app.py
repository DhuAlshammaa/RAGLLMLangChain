import sys
if sys.platform != "win32":  # Linux/Mac (Streamlit Cloud)
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

from rag_methods import (load_doc_to_db,
                         load_url_to_db,
                         stream_llm_response,
                         stream_llm_rag_response

)
dotenv.load_dotenv()

MODELS =[
    # "openai/ol-mini"
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-20240620",
    ]

st.set_page_config(
    page_title="RAG LLM app?",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""
    <h2 style="text-align: center;">
        🧠📚🔍 <i>Do your LLM even RAG bro?</i> 🤖💬
    </h2>
""")

# --- Initial Setup ---

# Generates a unique session ID for each user using uuid.uuid4().
# Useful to differentiate users or track individual chat sessions.
# 1. Set a unique session ID if not already present
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initializes an empty list to store document sources that will be used for RAG (Retrieval Augmented Generation).
# This could hold URLs, file names, or metadata about loaded content.
# 2. Initialize a list to track RAG sources

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []



# Sets up an initial chat message from the assistant.
# This will be the first message displayed in the chat UI.
# Helps the user know the system is ready for input.
# 3. Set up initial conversation with a default assistant message

#if you dont have cloud 
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there! How can I assist you today?"
        }
    ]

# if you have cloud

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "user", "content" :"Hello"},
        {
            "role": "assistant", "content" : "Hi there! How can I assist you today?"}
    ]


# Side bar LLM API Tokens

# --- Sidebar LLM API Tokens ---
with st.sidebar:
    # Load default OpenAI API key from environment variables
    default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""

    # Input field for OpenAI API key with a help popover
    with st.popover("🔑 OpenAI"):
        openai_api_key = st.text_input(
            "Introduce your OpenAI API Key (https://platform.openai.com/)",
            value=default_openai_api_key,
            type="password",
            key="openai_api_key"
        )

    # Load default Anthropic API key from environment variables
    default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""

    # Input field for Anthropic API key with a help popover
    with st.popover("🔑 Anthropic"):
        anthropic_api_key = st.text_input(
            "Introduce your Anthropic API Key (https://console.anthropic.com/)",
            value=default_anthropic_api_key,
            type="password"

        )



# --- Main Content ---
# Check if the user has entered API keys for OpenAI or Anthropic
missing_openai = (
    not openai_api_key or
    "sk-" not in openai_api_key
)

missing_anthropic = (
    not anthropic_api_key
)

# If both API keys are missing, show a warning
if missing_openai and missing_anthropic:
    st.write("⚠️")
    st.warning("🔑 Please introduce an API Key to continue...")

else:
    # Sidebar for model selection
    with st.sidebar:
        st.divider()
        st.selectbox(
            "🤖 Select a Model",
            [
                model for model in MODELS
                if ("openai" in model and not missing_openai)
                or ("anthropic" in model and not missing_anthropic)
            ],
            key="model"
        )


        cols0 = st.columns(2)
        with cols0[0]:
            is_vector_db_loaded = (
        "vector_db" in st.session_state and st.session_state.vector_db is not None
    )

    # Toggle for enabling/disabling RAG mode
            st.toggle(
            "Use RAG",
            value=is_vector_db_loaded,       # ON if the vector DB is loaded
            key="use_rag",                   # Streamlit state key
            disabled=not is_vector_db_loaded # Disable toggle if DB not loaded
            )
        with cols0[1]:
            st.button("Clear Chat",on_click=lambda: st.session_state.messages.clear(), type="primary")
        
                # Section Header
        st.header("📚 RAG Sources")

        # --- File Upload for RAG with Documents ---
        st.file_uploader(
            label="📂 Upload a document",
            type=["pdf", "txt", "docx", "doc", "json", "md", "pptx", "ppt","pdf"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs"
        )

        # --- URL Input for RAG with Websites ---
        st.text_input(
            label="🌐 Introduce a URL",
            placeholder="https://example.com",
            on_change=load_url_to_db,
            key="rag_url"
        )

        # --- Show Uploaded Documents in an Expander ---
        with st.expander(f"📄 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write(
                [] if not is_vector_db_loaded 
                else [source for source in st.session_state.vector_db.get()["metadatas"]]
            )

#is simply grabbing the part after the slash (/) from whatever model string you stored in st.session_state.model.
#Main chat app
model_provider = st.session_state.model.split("/")[0] 
if model_provider =="openai":
    llm_stream = ChatOpenAI(
        api_key=openai_api_key,
        model_name = st.session_state.model.split("/")[-1],
        temperature=0.3,
        streaming=True,
    )

# if model_provider =="anthropic":
#     llm_stream = ChatAnthropic(
#         model_name = st.session_state.model.split("/")[-1],
#         temperature=0.3,
#         streaming=True,
#     )


# Display previous messages from the session
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Your message"):
    # Save user's message to session state
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display the user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare assistant's reply container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Convert stored messages into LangChain's message format
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        
        else: st.write_stream(stream_llm_rag_response(llm_stream, messages))


        # Stream the assistant's response
        #llm_stream is your model client (e.g., ChatOpenAI or ChatAnthropic) configured for streaming.

        # stream_llm_response(llm_stream, messages) is a generator function that:

        # Sends messages (your chat history) to the model.

        # Yields pieces of the reply (chunks/tokens) as they come in.

        # Updates message_placeholder each time so the user sees the reply building live.

        # st.write_stream(...) takes that generator and writes each chunk directly into the UI as soon as it’s yielded.
        # This gives you the typing effect instead of waiting for the whole reply.
       


# Step 6 – Convert conversation history for the model
# python
# Copy
# Edit
# messages = [
#     HumanMessage(content=m["content"]) if m["role"] == "user"
#     else AIMessage(content=m["content"])
#     for m in st.session_state.messages
# ]
# st.session_state.messages is our saved chat history — a list of dictionaries like:

# python
# Copy
# Edit
# [
#     {"role": "user", "content": "Hi there"},
#     {"role": "assistant", "content": "Hello! How can I help?"}
# ]
# The LLM API (via LangChain) doesn’t work with plain dicts — it expects special objects like HumanMessage and AIMessage.

# This list comprehension loops through each message in history and:

# Turns it into a HumanMessage if "role" is "user".

# Turns it into an AIMessage if "role" is "assistant".

# The result, messages, is now in the format the LLM client understands.