import streamlit as st
import os
import PyPDF2
from Model1 import Model
from Display import Display
import fitz
from langchain_community.document_loaders import PyPDFLoader

# run app using "streamlit run app.py"
st.set_page_config(
    page_title="Retrieval Augmented Generation",
    layout="wide",
)

st.title("Retrieval Augmented Generation")
uploaded_file = st.file_uploader("Upload PDF")

def extract_text_from_pdf(uploaded_file):
    uploaded_file.seek(0)  # Reset the file pointer to the start
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text.append(page.get_text("text"))
    return text
    
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.pdf'):
            model = Model()
            model.question = st.text_input("Enter a question")
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
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
                retriever = model.get_retriever(data)
                chunks = retriever.invoke(model.question)
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
            st.write("Invalid file type")
           
            

    except Exception as e:
        st.write("Error reading file:", e)
    