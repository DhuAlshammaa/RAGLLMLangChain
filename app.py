import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage

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
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there! How can I assist you today?"
        }
    ]
