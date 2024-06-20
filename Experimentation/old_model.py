from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from langchain_elasticsearch import ElasticsearchStore
from semantic_router.layer import RouteLayer
from langchain_openai import OpenAIEmbeddings
import os 

#insert openapi key
os.environ["OPENAI_API_KEY"] = "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"
question = "What are examples of vector databases"
print(question)

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
print('---llm---:\n',llm.invoke(question).content)
print('---End LLM---')


data = []
current_dir = os.getcwd()
for file in os.listdir(current_dir):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(current_dir, file)
        loader = PyPDFLoader(pdf_path)
        data.extend(loader.load())


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(data)
vectorstore = ElasticsearchStore.from_documents(
            documents=docs,
            index_name="elastic_search_vectorstore",
            es_api_key="MGR5S0Y1QUJ0NWNBRFZPSWx6RGg6NmJrYVM1ZzZTaXlabjJCeUNfN1NHUQ==",
            embedding=OpenAIEmbeddings(),
            es_cloud_id="3438e9938373428281c9861abac4c00c:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGVjOTBhOWEzMmY5MTQwZjI4OTAwOGMzMjhiZTlkZmI2JGQ5ODE5NWU4MmJlMDQwNzNhOWYxZDAwZmYzM2YzMTZk"

        )
#vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
chunks = retriever.invoke(question)
print("---Retrival---:")
for i in range(len(chunks)):
    chunk = chunks[i].page_content.replace("\n", " ")
    print(f'{i+1}: \n {chunk}\n')
print('---End Retrieval---')


def create_rag_chain(prompt=None):
    if prompt == None: 
        prompt = hub.pull("rlm/rag-prompt")
    else:
        prompt = PromptTemplate.from_template(prompt)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm
        | StrOutputParser()
    )
    return rag_chain

print("---RAG with prompt from hub---:")
print(create_rag_chain().invoke(question))
print("---END RAG with prompt from hub---:")

print("---RAG with custom prompt---:")
#print(hub.pull("rlm/rag-prompt"))
custom_prompt = """
Using {context} as a retriever for data chucks relevant to {question}, polish the context to provide the answer to the question.
However if {question} is not strongly related to any of questions in {context}, say 
'I'm afraid I cannot assist with that. If you have any questions concering retrieval augmented generation, I would be happy to help'
"""
print(create_rag_chain(custom_prompt).invoke(question))
print("---END RAG with custom prompt---:")


print("---Semantic Router Guard---")
rag_route = Route(
    name="Retrieval_Augmented_Generation",
    utterances=[
        "Overview of retrieval augmented generation",
        "How does retrieval augmented generation work",
        "What are retrieval augmented generation frameworks",
        "What is a text splitter",
        "What are vector databases",
        "How to implement retrieval augmented generation",
        "What are pros and cons of retrieval augmented generation",
        "How do you enhance retrieval augmented generation",
        "What are some examples of cloud computing services"
    ],
)

encoder = OpenAIEncoder()
routes = [rag_route]
layer = RouteLayer(encoder=encoder, routes=routes)

def semantic_layer(question):
    route = layer(question)

    if route.name == "Retrieval_Augmented_Generation":
        print(question + f" \nANSWER:")
        print(create_rag_chain().invoke(question))
    else:
        output = "I'm afraid I cannot assist with that. If you have any questions concering retrieval augmented generation, I would be happy to help"
        print(output)
        
semantic_layer(question)
print("---End Semantic Router Guard---")


