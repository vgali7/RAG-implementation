from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from semantic_router.layer import RouteLayer
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch
import streamlit as st
import os 
os.environ["OPENAI_API_KEY"] = ""

class Model:
    def __init__(self):
        self.question = None
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.retriever = None
        #self.es = Elasticsearch(['http://localhost:9200'])
        self.es = Elasticsearch(
            cloud_id="e78e4d9167bc43e7842f1f756114af3a:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGQxZTJiZmFhZTg3NjRiNzBhNDNhMjQ3NWJkNzRiYWMwJGNhZmM1YjdiNzIyNzRhM2NhYzdiYjJiYWM3MThiNjgx"
        ,
            api_key= "",
        )

        self.routes = []
    
    def add_route(self,file_name, docs):
        utterances = self.llm.invoke(f'Create uttereances that describe this json {docs} use in a semantic router').content.split("\n")
        route = Route(name = file_name, utterances = utterances, score_threshold=0.82)
        self.routes.append(route)


    def get_retriever(self, text_data, json_data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        text_docs = text_splitter.split_documents(text_data)
        if self.es.indices.exists(index="elastic_search_vectorstore"):
            self.es.indices.delete(index="elastic_search_vectorstore")
        vectorstore = ElasticsearchStore.from_documents(
            documents=text_docs+json_data,
            index_name="elastic_search_vectorstore",
            es_api_key="",
            embedding=OpenAIEmbeddings(),
            es_cloud_id="e78e4d9167bc43e7842f1f756114af3a:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGQxZTJiZmFhZTg3NjRiNzBhNDNhMjQ3NWJkNzRiYWMwJGNhZmM1YjdiNzIyNzRhM2NhYzdiYjJiYWM3MThiNjgx"
        )
        #vectorstore = Chroma.from_documents(documents=text_docs+json_data, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def create_rag_chain(self, custom_prompt=False):
        if custom_prompt == False: 
            prompt = hub.pull("rlm/rag-prompt")
        else:
            prompt = """
            You are a powerful assistant who provides answers to questions based on retrieved data using context:
            <context>
            {context}
            </context>
            Question: {question}
            """
            prompt = PromptTemplate.from_template(prompt)
        
        rag_chain = ({"context": self.retriever, "question": RunnablePassthrough()} | prompt | self.llm | StrOutputParser())
        return rag_chain.invoke(self.question)

    def semantic_router(self):
        layer = RouteLayer(encoder= OpenAIEncoder(), routes=self.routes)
        route = layer(self.question)

        if route.name is not None:
            return route.name
        else:
            output = "No route detected"
            return output
            