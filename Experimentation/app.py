import streamlit as st
from model_base import Model
import implementation
from Display import Display

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


def main():
    st.title("Retrieval Augmented Generation")

    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    if uploaded_files:
        model = Model()
        data = implementation.process_files(model, uploaded_files)

        if data[0] or data[1]:
            question = st.form(key='my_form', clear_on_submit=True)
            model.question = question.text_input(label="Enter a question")
            submit_button = question.form_submit_button("Submit request")
            display = Display(model.question)

            if model.question:
                implementation.run_rag(model, display, data)
                implementation.run_agent(model, data)
                st.info('Process completed.')
        else:
            display.container.empty()
                

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    layout="wide",
)

if st.session_state.get('logged_in', False):
    main()
else:
    login()
