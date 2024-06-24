import streamlit as st
import os
import PyPDF2
from Model1 import Model
from Display import Display
import fitz
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from semantic_router import Route

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "coffee":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid username or password")


def process_files(model, uploaded_files):
    selected_files = [None] * len(uploaded_files)
    st.write("----------\nSelect PDF's \n ----------")
    for i in range(len(uploaded_files)):
        button = st.checkbox(f"{uploaded_files[i].name}")
        selected_files[i] = button
    selected_pdf_files = [uploaded_files[i] for i, selected in enumerate(selected_files) if selected]

    data = []
    for i, file in enumerate(selected_pdf_files):
        if file.name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name  # Get the path of the temp file
            model.add_route(file.name)
            loader = PyPDFLoader(temp_file_path)
            data.extend(loader.load())
        else:
            st.write("Invalid file type")
            break
    return data


def run_models(model, display, data):
    display.container.empty()
    display.container.write("Processing...")

    # base llm
    llm_output = model.llm.invoke(model.question).content
    display.dic['LLM without retrieval'] = llm_output

    # Base retrieval
    model.get_retriever(data)
    chunks = model.retriever.invoke(model.question)
    retriever_data = []
    for i in range(len(chunks)):
        chunk = chunks[i].page_content.replace("\n", " ")
        retriever_data.append(f'Chunk {i+1}:')
        retriever_data.append(chunk)
    display.dic['Base retrieval'] = "\n\n".join(retriever_data)

    # RAG with hub prompt
    display.dic['RAG with hub prompt'] = model.create_rag_chain(custom_prompt=False)

    # RAG with custom prompt
    display.dic['RAG with custom prompt'] = model.create_rag_chain(custom_prompt=True)
    display.show()

    # Semantic router guard
    st.write("Semantic Router Guard:")
    for i, route in enumerate(model.routes):
        if utterance := st.text_input(f"Enter utterances for {route.name} seperated by a comma"):
            model.routes[i].utterances = utterance.split(",")
            model.routes[i].score_threshold = 0.75
    #display.dic['Semantic Router Guard'] = model.semantic_router()
    st.write(f"{'Route'}: {model.semantic_router().name}\n\n----------\n\n")
    #display.show()


def main():
    st.title("Retrieval Augmented Generation")

    uploaded_files = st.file_uploader("Upload PDF's", accept_multiple_files=True)
    if uploaded_files:
        model = Model()

        data = process_files(model, uploaded_files)
        model.question = st.text_input("Enter a question")

        display = Display()
        if model.question and data:
            run_models(model, display, data)
        else:
            display.container.empty()


st.set_page_config(
    page_title="Retrieval Augmented Generation",
    layout="wide",
)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main()
else:
    login()