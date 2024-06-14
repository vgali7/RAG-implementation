from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from semantic_router.utils.function_call import get_schema
from semantic_router.layer import RouteLayer
from llama_index.core.tools import FunctionTool
from pdf_reader import retriever
import os 

os.environ["OPENAI_API_KEY"] = ""
encoder = OpenAIEncoder()
question = "How do i replace my stolen card"

###load data
dataset_name = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
page_content_column = "response"  # or any other column you're interested in


def path(question) -> str:
    return retriever.get_relevant_documents(question)[0]
schema1 = get_schema(path)
print(schema1)


path_finder = Route(
    name="find_path",
    utterances=[
        "I want to get to the bank",
        "What is the path to the bank",
        "Show me how to get to the bank",
        "Where is the nearest bank"
    ],
    function_schema=schema1,
)


routes = [path_finder]
layer = RouteLayer(encoder=encoder, routes=routes)

def semantic_layer(question):
    route = layer(question)

    if route.name == "find_path":
        return f" \nANSWER: {path(question)}"
    else:
        pass
    return 

path_engine = FunctionTool.from_defaults(
    fn=semantic_layer,
    name='path_finder',
    description="this tool finds a path to the bank"
)

print(semantic_layer(question))




