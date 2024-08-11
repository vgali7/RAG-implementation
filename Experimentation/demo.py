from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
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
        if team_data == []:
            model.get_retriever(json_data)
        else:
            route = model.router()
            if route == "Team":
                st.write(f"Analyzing team data")
                model.get_retriever(team_data)
            else:
                st.write(f"Analyzing self data")
                model.get_retriever(json_data)


        output = model.create_rag_chain()
        st.write(f"Question: {model.question}")
        st.write(output)

        st.write("------\nAgent:")
        tool = create_retriever_tool(
            retriever = model.retriever,
            name = "json_retriever",
            description = "Searches and returns relevant data from a variety of input jsons to answer questions",
        )

        tools = [tool]
        human = """
        TOOLS
        ------
        Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
        {tools}
        RESPONSE FORMAT INSTRUCTIONS
        ----------------------------
        When responding to me, please output a response in the format:
        Use this if you want the human to use a tool. 
        Markdown code snippet formatted in the following schema:
        ```json
        {{
            "action": string, \ The action to take. Must be one of {tool_names}
            "action_input": string \ The input to the action
        }}
        ```
        **Option #2:**
        Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:
        ```json
        {{
            "action": "Final Answer",
            "action_input": string \ You should put what you want to return to use here
        }}
        ```
        USER'S INPUT
        --------------------
        Here is the user's input:
        {input}
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a powerful assistant who provides answers to questions based on retrieved data using context"),
                ("human", human),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        st.write(prompt)

        json_agent = create_json_chat_agent(model.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=json_agent, tools=tools)
        
        if model.question:
            result = agent_executor.invoke({"input": model.question})
            st.write(f'Question: {result["input"]}')
            st.write(result["output"])