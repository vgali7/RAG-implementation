from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from semantic_router.utils.function_call import get_schema
from semantic_router.layer import RouteLayer
import os 

os.environ["OPENAI_API_KEY"] = ""
encoder = OpenAIEncoder()
question = "How do i replace my stolen card"

###load data
dataset_name = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
page_content_column = "response"  # or any other column you're interested in


def replace_card() -> str:
    return "Here are the steps to follow to get a replacement card"
schema1 = get_schema(replace_card)
print(schema1)

def activate() -> str:
    return "Congratulations, your card has been activated!"
schema2 = get_schema(activate)
print(schema2)


lost_card = Route(
    name="lost_card",
    utterances=[
        "I lost my Visa, can you help me with blocking it",
        "What are the steps to block a stolen Visa card",
        "I need immediate help to secure my stolen Visa card    ",
        "Is there a way to instantly block my Visa card? It was stolen",
        "What do I need to do to stop my stolen Visa card from being used",
    ],
    function_schema=schema1,
)
activate_card = Route(
    name="activate_card",
    utterances=[
        "I want help to activate acard on mobile",
        "i have to activate an visa online where can i do it",
        "would it be possible to activate a Visa online?",
        "i want assistance to activate a visa",
        "I wouild like to activate a credit card",
    ],
    function_schema=schema2,
)

routes = [lost_card, activate_card]
layer = RouteLayer(encoder=encoder, routes=routes)

def semantic_layer(question):
    route = layer(question)

    if route.name == "lost_card":
        question += f" \nANSWER: {replace_card()}"
    elif route.name == "activate_card":
        question += f" \nANSWER: {activate()}"
    else:
        pass
    return question

print(semantic_layer(question))




