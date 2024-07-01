import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain.memory import ConversationBufferMemory
from old_model import prompt

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
    tools = []
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

            def metadata_func(record: dict, metadata: dict) -> dict:
                metadata["name"] = record.get("name")
                metadata["id"] = record.get("worker_id", record.get("manager_id"))
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
    return text_data, json_data


def run_rag(model, display, data):
    display.container.write("Processing...")

    # Base retrieval
    model.get_retriever(data[0], data[1])
    chunks = model.retriever.invoke(model.question)
    retriever_data = []
    for i in range(len(chunks)):
        chunk = chunks[i].page_content.replace("\n", " ")
        retriever_data.append(f'Chunk {i+1}:')
        retriever_data.append(chunk)
    display.dic['Base retrieval'] = "\n\n".join(retriever_data)

    # RAG with hub prompt
    #display.dic['RAG with hub prompt'] = model.create_rag_chain(custom_prompt=False)

    # RAG with custom prompt
    display.dic['RAG with custom prompt'] = model.create_rag_chain(custom_prompt=True)
    display.show()
    # Semantic router guard
    #if button := st.button('Semantic Router'):
    #    display.dic['Semantic Router Guard'] = model.semantic_router()

    #display.show()


def run_agent(model, data):
    st.subheader("Agent")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chat_history = memory.buffer_as_messages

    model.get_retriever(data[0], data[1])
    tool = create_retriever_tool(
    retriever = model.retriever,
        name = "json_retriever",
        description = "Searches and returns relevant data from a variety of input jsons to answer questions",
    )

    tools = [tool]
    agent = create_json_chat_agent(model.llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
    )

    if model.question:
        result = agent_executor.invoke({"input": model.question, "chat_history": chat_history})
        st.markdown(f'Question: {result["input"]}')
        st.markdown(f'Answer: {result["output"]}')