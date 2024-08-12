from langchain.text_splitter import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from semantic_router.layer import RouteLayer
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch
from langchain.docstore.document import Document
import streamlit as st
import os 

api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key


class Model:
    def __init__(self):
        os.environ['OPENAI_API_KEY'] = api_key
        self.question = None
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.llm = ChatOpenAI(model_name="gpt-4o")
        self.retriever = None
        self.es = Elasticsearch('http://localhost:9200')
        self.routes = []
    
    def add_route(self, name, utterances):
        route = Route(name = name, utterances = utterances)
        self.routes.append(route)


    def get_retriever(self, json_data=[]):
        if self.es.indices.exists(index="elastic_search_vectorstore"):
            self.es.indices.delete(index="elastic_search_vectorstore")
        vectorstore = ElasticsearchStore.from_documents(
            documents=json_data,
            index_name="elastic_search_vectorstore",
            embedding=OpenAIEmbeddings(),
            es_url="http://localhost:9200",
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        

    def create_rag_chain(self):
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

    def router(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""Classify the following question into one of the following: Self, Team\n
            Use the following to define the classification\n
            'Self': The user is asking for information relevant to his own data, 
            'Team': The user is asking for information relevant to one or more of his employees or his entire team\n
            Question: {question}\n
            Answer in one word
            """
        )
        route_chain = {"question": RunnablePassthrough()} | prompt | self.llm
        route = route_chain.invoke(self.question).content
        return route
            
