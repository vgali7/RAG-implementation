import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_json_chat_agent
from old_model import prompt
import json
from model_base import Model
from Display import Display


def implement_langchain(uploaded_files):
    model = Model()
    model2 = Model()
    data = process_files(model, uploaded_files)

    if data[0] or data[1]:
        question = st.form(key='my_form', clear_on_submit=True)
        model.question = question.text_input(label="Enter a question")
        model2.question = model.question
        submit_button = question.form_submit_button("Submit request")
        display = Display(model.question, "Json Split RAG")
        display2 = Display(model.question, "Text Split RAG")

        if model.question:
            run_rag(model, display, data)
            run_rag(model2, display2, data, split_json=True)
            run_agent(model, data)
            st.info('Process completed.')
        else:
            display.container.empty()

def process_files(model, uploaded_files):
    selected_files = [None] * len(uploaded_files)
    st.write("----------\nSelect Files \n ----------")
    for i in range(len(uploaded_files)):
        button = st.checkbox(f"{uploaded_files[i].name}")
        selected_files[i] = button
    selected_files = [uploaded_files[i] for i, selected in enumerate(selected_files) if selected]
    st.write("----------")

    json_data = []
    text_data = []
    json_split_data = []
    for i, file in enumerate(selected_files):
        if file.name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name  # Get the path of the temp file

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['file'] = file.name
            text_data.extend(docs)
            model.add_route(file.name, docs)

        elif file.name.endswith('.json'):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name  # Get the path of the temp file

            json_file = json.load(file)
            json_split_data.append(json_file)

            def metadata_func(record: dict, metadata: dict) -> dict:
                metadata["name"] = record.get("name")
                return metadata
        
            loader = JSONLoader(
                file_path=temp_file.name,
                jq_schema = '.',
                text_content=False,
                metadata_func=metadata_func
            )

            docs = loader.load()
            for doc in docs:
                doc.metadata['file'] = file.name
            
            json_data.extend(docs)
            model.add_route(file.name, docs)

        else:
            st.write("Invalid file type")
            break
    return text_data, json_data, json_split_data


def run_rag(model, display, data, split_json=False):
    display.container.write("Processing...")

    # Base retrieval
    if split_json:
        model.get_retriever(text_data=data[0], json_data=data[2], split_json=True)
    else:
        model.get_retriever(text_data=data[0], json_data=data[1])
    chunks = model.retriever.invoke(model.question)
    
    retriever_data = []
    for i in range(len(chunks)):
        chunk = chunks[i].page_content.replace("\n", " ")
        retriever_data.append(f'Chunk {i+1}:')
        retriever_data.append(chunk)
    
    if st.checkbox(f'chunks split = {split_json}:'):
        display.dic['Base retrieval'] = "\n\n".join(retriever_data)

    # RAG with hub prompt
    #display.dic['RAG with hub prompt'] = model.create_rag_chain(custom_prompt=False)

    # RAG with custom prompt
    display.dic['RAG with custom prompt'] = model.create_rag_chain()
    display.show()
    # Semantic router guard
    #if button := st.button('Semantic Router'):
    #    display.dic['Semantic Router Guard'] = model.semantic_router()

    #display.show()


def run_agent(model, data):
    st.header("Agent")

    model.get_retriever(data[0], data[1])
    tool = create_retriever_tool(
        retriever = model.retriever,
        name = "json_retriever",
        description = "Searches and returns relevant data from a variety of input jsons to answer questions",
    )

    tools = [tool]
    agent = create_json_chat_agent(model.llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    if model.question:
        result = agent_executor.invoke({"input": model.question})
        st.markdown(f'Question: {result["input"]}')
        st.markdown(f'Answer: {result["output"]}')