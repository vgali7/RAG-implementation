from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from semantic_router.layer import RouteLayer
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch
import os 
os.environ["OPENAI_API_KEY"] = ""
question = "What is happening under the hood of a computer"
print(question)

class Model:
    def __init__(self):
        self.question = None
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.retriever = None
        #self.es = Elasticsearch(['http://localhost:9200'])
        self.es = Elasticsearch(
            cloud_id="3438e9938373428281c9861abac4c00c:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGVjOTBhOWEzMmY5MTQwZjI4OTAwOGMzMjhiZTlkZmI2JGQ5ODE5NWU4MmJlMDQwNzNhOWYxZDAwZmYzM2YzMTZk"
        ,
            api_key= "MGR5S0Y1QUJ0NWNBRFZPSWx6RGg6NmJrYVM1ZzZTaXlabjJCeUNfN1NHUQ==",
        )

        self.routes = []
    
    def add_route(self,file_name):
        print(file_name)
        route = Route(name = file_name, utterances = [])
        self.routes.append(route)

    def get_pdf(self):
        data = []
        current_dir = os.getcwd()
        for file in os.listdir(current_dir):
            if file.endswith('.pdf'):
                pdf_path = os.path.join(current_dir, file)
                loader = PyPDFLoader(pdf_path)
                data.extend(loader.load())

    def get_retriever(self,data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(data)
        if self.es.indices.exists(index="elastic_search_vectorstore"):
            self.es.indices.delete(index="elastic_search_vectorstore")
        vectorstore = ElasticsearchStore.from_documents(
            documents=docs,
            index_name="elastic_search_vectorstore",
            es_api_key="MGR5S0Y1QUJ0NWNBRFZPSWx6RGg6NmJrYVM1ZzZTaXlabjJCeUNfN1NHUQ==",
            embedding=OpenAIEmbeddings(),
            es_cloud_id="3438e9938373428281c9861abac4c00c:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGVjOTBhOWEzMmY5MTQwZjI4OTAwOGMzMjhiZTlkZmI2JGQ5ODE5NWU4MmJlMDQwNzNhOWYxZDAwZmYzM2YzMTZk"

        )
        #vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever()

    def create_rag_chain(self, custom_prompt=False):
        if custom_prompt == False: 
            prompt = hub.pull("rlm/rag-prompt")
        else:
            prompt = """
            Using {context} as a retriever, retrieve relevant information based on the question {question}. Evaluate each retrieved chunk to determine its strong relevance to the question. If a chunk is highly relevant, incorporate it as part of your response. Otherwise, disregard it.
            If any of the chunks are strongly related, respond with only the answer.
            If none of the chunks are strongly related, respond with
            'I'm afraid I cannot assist with that. If you have any questions concering concerning the selected files, I would be happy to help'
            """
            prompt = PromptTemplate.from_template(prompt)

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        )
        return rag_chain.invoke(self.question)

    def semantic_router(self):
        layer = RouteLayer(encoder= OpenAIEncoder(), routes=self.routes)
        route = layer(self.question)

        if route.name is not None:
            return route
        else:
            output = "I'm afraid I cannot assist with that. If you have any questions concerning the selected files, I would be happy to help"
            return output