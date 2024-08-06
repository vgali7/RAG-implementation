import streamlit as st
from langchain_implementation import implement_langchain
import llama_index_implementation
import demo

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "coffee":
            st.session_state['logged_in'] = True
            st.session_state.langchain = False
            st.session_state.llama_index = False
            st.rerun()
        else:
            st.error("Invalid username or password")

def main():
    st.title("Retrieval Augmented Generation")

    option = st.radio("", ["Demo Account", "Model"])

    if option == "Demo Account":
        data, heirarchy, managers = demo.main()
        st.write("--------\n Select User")
        name = st.radio("", data.keys())
        json_data, team_data = demo.choose(name, data, heirarchy, managers)
        demo.implement(json_data, team_data)


    elif option == "Model":
        uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
        if uploaded_files:
            framework = st.radio("", ["Langchain", "LLamaIndex"])

            if framework == "Langchain":
                implement_langchain(uploaded_files)

            if framework == "LLamaIndex":
                tools = llama_index_implementation.process_files(uploaded_files)
                llama_index_implementation.query(tools)


st.set_page_config(
    page_title="Retrieval Augmented Generation",
    layout="wide",
)

if st.session_state.get('logged_in', False):
    main()
else:
    login()
