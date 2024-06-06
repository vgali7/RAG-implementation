import os
os.environ["OPENAI_API_KEY"] = ""

import pandas as pd
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import america_engine

current_dir = os.getcwd()
population_path = os.path.join(current_dir, "data", "population.csv")
population_df = pd.read_csv(population_path)

#RAG for CSV
population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
#population_query_engine.query("what is the population of canada")

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, metadata=ToolMetadata(
        name="population_data",
        description="this gives information about world population data"
    )),
    QueryEngineTool(query_engine=america_engine, metadata=ToolMetadata(
        name="america_data",
        description="this gives detailed information about america the country"
    ))
]

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)


while (prompt := input("Enter a prompt (q to Quit): ")) != "q":
    result = agent.query(prompt)
    print(result)