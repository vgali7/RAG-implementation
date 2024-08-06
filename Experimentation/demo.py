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
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']
    heirarchy, managers = {}, {}

    for file in os.listdir(os.getcwd()):
        if file.endswith('.csv'):
            csv_path = os.path.join(current_dir, file)

            for encoding in encodings:
                try:
                    with open(csv_path, mode='r', newline='', encoding=encoding) as csv_file:
                        csv_reader = csv.DictReader(csv_file)
                        if file.endswith('DETAILS.csv'):
                            for row in csv_reader:
                                name = row["name"]
                                del row["name"]
                                jsons[name] = row
                        else:
                            for row in csv_reader:
                                heirarchy[row['Manager']] = "Manager"
                                managers[row['Manager']] = [row['Employee 1'], row['Employee 2'], row['Employee 3']]
                    break
                except:
                    pass

    return jsons, heirarchy, managers         

def choose(name, data, heirarchy, managers):
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

    team_data = []
    if name in heirarchy:
        role = heirarchy[name]
        if role == "Manager":
            team = managers[name]
            for i, employee in enumerate(team):
                for key in data[employee]:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp_file:
                        json.dump(data[employee][key], temp_file, indent=2)
                        temp_file_path = temp_file.name 
                        
                    def metadata_func(record: dict, metadata: dict) -> dict:
                        metadata[f"employee {i}"] = employee
                        metadata["manager"] = name
                        metadata["source"] = key
                        return metadata

                    loader = JSONLoader(
                        file_path=temp_file.name,
                        jq_schema = '.',
                        text_content=False,
                        metadata_func=metadata_func
                    )

                    docs = loader.load()
                
                    team_data.extend(docs)
            st.write("--------------\nteam:")
            st.write(team_data)
        else:
            pass
    return json_data, team_data
        
def implement(json_data, team_data):
    model = Model()
    question = st.form(key='my_form', clear_on_submit=True)
    model.question = question.text_input(label="Enter a question")
    submit_button = question.form_submit_button("Submit")

    team_utterances = ["The user is not asking about his own personal data but about someone elses"]
    model.add_route("Team", team_utterances)
    
    self_utterances =  ["The user is asking about his own personal data"]
    model.add_route("Self", self_utterances)


    if model.question:
        st.write(model.question)
        if team_data == []:
            model.get_retriever(json_data)
        else:
            route = model.router()
            if route == "Team":
                model.get_retriever(team_data)
            elif route == "Self":
                model.get_retriever(json_data)
            else:
                model.get_retriever(json_data)


        output = model.create_rag_chain()
        st.write(output)
