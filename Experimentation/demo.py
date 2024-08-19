from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ChatMessageHistory
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
    question = st.form(key='model_question', clear_on_submit=True)
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


        #output = model.create_rag_chain() 
        #st.write(f"Question: {model.question}")
        #st.write(output)
 
        tool = create_retriever_tool(
            retriever = model.retriever,
            name = "ACCOUNT-PLAN DETAILS",
            description = "Searches and returns relevant data regarding sales for one or more salesman.",
        )

        tools = [tool]
        human = """
        TOOLS
        ------
        Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:
        {tools} as well as chat history
        RESPONSE FORMAT INSTRUCTIONS
        ----------------------------
        When responding to me, please output a response in one of the formats below:
        Use this if you want the human is asking a question that has no relevance to any of the tools. 
        **Option #1** :
            "action": string, \ This action is not allowed. The action must be one of {tool_names}
            "action_input": string \ The action must be one of {tool_names}.

        **Option #2** :
        Use this if you want to respond directly to the human using the available tools.
            "action": "Final Answer",
            "action_input": string \ You should put what you want to return to use here

        USER'S INPUT
        --------------------
        Here is the user's input:
        {input}
        """ 
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a powerful assistant who provides answers to questions based on retrieved data using context and chat history"),
                ("human", human),
                MessagesPlaceholder("agent_scratchpad"),
                MessagesPlaceholder("chat_history"),
            ]
        )
 
        json_agent = create_json_chat_agent(model.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=json_agent, tools=tools, handle_parsing_errors=True)
        agent_chain = RunnableWithMessageHistory(agent_executor, lambda session_id: model.memory, input_messages_key="input", history_messages_key="chat_history")
        
        response = agent_chain.invoke({"input": model.question} ,config={"configurable": {"session_id": "1"}})
        st.write("------\nAgent:")
        st.write(response["output"]) 

        if st.button("Follow up"):
            follow_up_question = input("Follow up:")
            if follow_up_question:
                response = agent_chain.invoke({"input": follow_up_question} ,config={"configurable": {"session_id": "1"}})
                st.write(response)
                print(response)
            