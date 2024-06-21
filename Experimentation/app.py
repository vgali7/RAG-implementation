import streamlit as st
import os
import PyPDF2
from Model1 import Model
from Display import Display
import fitz
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    layout="wide",
)

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "coffee":
            st.session_state['logged_in'] = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def process_pdf_file(uploaded_file, model, display):
    try:
        if uploaded_file.name.endswith('.pdf'):
            model.question = st.text_input("Enter a question")
            display = Display()
            display.container.empty()

            if model.question:
                display.container.write("Processing...")

                # base llm
                llm_output = model.llm.invoke(model.question).content
                display.dic['LLM without retrieval'] = llm_output

                temp_file = "./temp.pdf"
                with open(temp_file, "wb") as file:
                    file.write(uploaded_file.getvalue())
                    file_name = "testing.pdf"

                loader = PyPDFLoader(temp_file)
                data = loader.load()
            
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

                display.dic['Semantic Router Guard'] = model.semantic_router()

                display.show()
            
            else:
                display.container.empty()
        
        else:
            st.write(f"Skipping {uploaded_file.name}. Invalid file type")

    except Exception as e:
        st.write(f"Error reading file {uploaded_file.name}:", e)

def main():
    st.title("Retrieval Augmented Generation")
    uploaded_files = st.file_uploader("Upload PDF", accept_multiple_files=True)

    if uploaded_files:
        model = Model()
        display = Display()

        for uploaded_file in uploaded_files:
            process_pdf_file(uploaded_file, model, display)

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main()
else:
    login()
