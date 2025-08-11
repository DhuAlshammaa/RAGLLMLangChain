import os
import dotenv
from pathlib import Path

# LangChain and related imports
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import streamlit as st

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

DB_DOCS_LIMIT = 5

def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append(
        {"role": "assistant", "content": response_message}
    )


def load_url_to_db():
    """Load a single web page into the RAG vector database."""

    # Ensure a URL is stored in session state
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []

        # Only proceed if URL not already in sources and limit not exceeded
        if url not in st.session_state.rag_sources and len(st.session_state.rag_sources) < 10:
            try:
                # Load web content
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                st.session_state.rag_sources.append(url)

            except Exception as e:
                st.error(f"Error loading document from [{url}]: {e}")

            # If we successfully loaded docs, split and store in DB
            if docs:
                _split_and_load_docs(docs)
                st.toast(f"Document from URL **{url}** loaded successfully ✅", icon="📄")

        else:
            st.error("Maximum number of documents reached (10).")

def load_doc_to_db():
    """
    Loads uploaded documents into the RAG database after validating format,
    respecting document limits, and converting them into text chunks.
    """

    # Ensure required session variables exist
    if "rag_docs" not in st.session_state or "rag_sources" not in st.session_state:
        return

    docs = []  # Temporary list to hold all loaded documents

    for doc_file in st.session_state.rag_docs:
        # Skip if document already processed
        if doc_file.name in st.session_state.rag_sources:
            continue

        # Enforce document limit
        if len(st.session_state.rag_sources) >= DB_DOCS_LIMIT:
            st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
            return
        os.makedirs("source_files", exist_ok=True)
        # Save uploaded file to a temporary folder
        file_path = f"./source_files/{doc_file.name}"
        with open(file_path, "wb") as file:
            file.write(doc_file.read())

        try:
            # Select the appropriate loader based on file type
            if doc_file.type == "application/pdf":
                loader = PyPDFLoader(file_path)
            elif doc_file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif doc_file.type in ["text/plain", "text/markdown"]:
                loader = TextLoader(file_path)
            else:
                st.warning(f"Document type {doc_file.type} not supported.")
                continue

            # Load the document and add to our list
            docs.extend(loader.load())
            st.session_state.rag_sources.append(doc_file.name)

        except Exception as e:
            # Handle loading errors
            st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
            print(f"Error loading document {doc_file.name}: {e}")

        finally:
            # Remove the temp file regardless of success or error
            os.remove(file_path)

    # If we have loaded documents, split and add them to the DB
    if docs:
        split_and_load_docs(docs)
        st.toast(
            f"Document(s) {', '.join([doc.name for doc in st.session_state.rag_docs])} loaded successfully ✅"
        )


def initialize_vector_db(docs):
    """
    Initializes a Chroma vector database from given documents.
    Adds embeddings, assigns a unique collection name, and manages stored collections.
    """
    # Local 
    # # Create the vector database from provided documents
    # from time import time
    # vector_db = Chroma.from_documents(
    #     documents=docs,
    #     embedding=OpenAIEmbeddings(),
    #     collection_name=(
    #         f"{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}"
    #     )
    # )
    # For the cloud 
       # Create the vector database from provided documents
    from time import time
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(api_key=st.session_state.openai_api_key),
        collection_name=(
            f"{str(time()).replace('.', '')[:14]}_{st.session_state['session_id']}"
        )
    )

    # Manage Chroma collections to avoid memory overflow (keep only last 20)
    chroma_client = vector_db._client
    collection_names = sorted(
        [collection.name for collection in chroma_client.list_collections()]
    )

    print(f"Number of collections: {len(collection_names)}")

    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def split_and_load_docs(docs):
    """
    Splits documents into chunks and loads them into the vector database.
    Uses a RecursiveCharacterTextSplitter for optimal chunking.
    """

    # Step 1: Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    # Step 2: Split the documents into chunks
    document_chunks = text_splitter.split_documents(docs)

    # Step 3: Initialize or update the vector database in session state
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


def _get_context_retriever_chain(vector_db, llm):
    """
    Build a retriever that uses conversation history to generate a
    better search query, then fetches relevant docs from vector_db.
    """
    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up "
                 "information relevant to the conversation, focusing on the user's goal.")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(llm):
    """
    Create the full conversational RAG chain:
    1) history-aware retriever (gets context)
    2) stuff-docs chain (adds context to the prompt for the LLM)
    """
    # assumes your Chroma (or other) vector DB is in st.session_state.vector_db
    import streamlit as st
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. You will answer the user's queries. "
         "You will have some context to help with your answers, but it may not always be "
         "completely related or helpful. Use your knowledge plus the provided context "
         "to assist the user.\n\n<context>{context}</context>"),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    # final chain that: (history -> search query) -> retrieve docs -> stuff into prompt -> answer
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    """
    Streams a Retrieval-Augmented Generation (RAG) response from the LLM.

    Args:
        llm_stream: The language model instance with streaming enabled.
        messages: A list of conversation messages (history + user input).

    Yields:
        Chunks of the assistant's response as they are generated.
    """
    # Build the conversational RAG chain
    conversation_chain = get_conversational_rag_chain(llm_stream)

    # Initialize the assistant's response message
    response_message = "**RAG Response**\n"

    # Stream the answer from the chain
    for chunk in conversation_chain.pick("answer").stream({
        "messages": messages[:-1],   # conversation history
        "input": messages[-1].content  # latest user message
    }):
        response_message += chunk
        yield chunk  # send chunk to the caller as it's generated

    # Store the complete assistant message in session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_message
    })

 #Note:
# We don’t manually create or call a tokenizer here because OpenAIEmbeddings()
# automatically handles tokenization inside the OpenAI API. 
# When we pass text chunks to Chroma.from_documents with OpenAIEmbeddings, 
# LangChain sends the raw text to OpenAI, which:
#   1. Tokenizes the text internally (splits into tokens).
#   2. Converts tokens into embedding vectors.
# The result is a list of numbers (vector) for each chunk, ready for the vector DB.
# If we want to see or control tokenization, we can use the tiktoken library
# to count or inspect tokens before sending text to the embedding model.