from langchain_community.document_loaders import JSONLoader
import csv
import os 
import tempfile
import streamlit as st
import json
from model_base import Model

def main():
    current_dir = os.getcwd()
    jsons = {}
    for file in os.listdir(os.getcwd()):
        if file.endswith('.csv'):
            csv_path = os.path.join(current_dir, file)

            with open(csv_path, mode='r', newline='', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    name = row["name"]
                    del row["name"]
                    jsons[name] = row
    return jsons          

def choose(name, data):
    st.write(data[name])
    json_data = []
    for key in data[name]:

        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp_file:
            json.dump(data[name][key], temp_file, indent=2)
            temp_file_path = temp_file.name 
            
        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["name"] = name
            metadata["source"] = key
            return metadata

        loader = JSONLoader(
            file_path=temp_file.name,
            jq_schema = '.',
            text_content=False,
            metadata_func=metadata_func
        )

        docs = loader.load()
    
        json_data.extend(docs)
    return json_data
        
def implement(json_data):
    model = Model()
    model.get_retriever(json_data=json_data)

    question = st.form(key='my_form', clear_on_submit=True)
    model.question = question.text_input(label="Enter a question")
    submit_button = question.form_submit_button("Submit")

    if model.question:
        st.write(model.question)
        st.write(model.create_rag_chain())