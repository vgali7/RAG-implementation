import streamlit as st

class Display:
    def __init__(self):
        self.container = st.empty()
        self.dic = {
            'LLM without retrieval': None,
            'Base retrieval': None,
            'RAG with hub prompt': None,
            'RAG with custom prompt': None,
            'Semantic Router Guard': None
        }

    def show(self):
        self.container.empty()
        
        formatted_dict = "\n".join([f"{key}: \n\n{value}\n\n----------\n\n" for key, value in self.dic.items()])

        self.container.write(formatted_dict)