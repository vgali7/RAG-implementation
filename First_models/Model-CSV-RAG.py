import os
import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine import PandasQueryEngine


os.environ["OPENAI_API_KEY"] = ""

current_dir = os.getcwd()
population_path = os.path.join(current_dir, "data", "population.csv")
population_df = pd.read_csv(population_path)

instruction_str = """\
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression."""

new_prompt = PromptTemplate(
    """\
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression: """
)

#RAG for CSV
population_query_engine = PandasQueryEngine(df=population_df, verbose=True, instruction_str=instruction_str)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})
population_query_engine.query("what is the population of canada")


