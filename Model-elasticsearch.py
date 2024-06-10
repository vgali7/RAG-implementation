from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain import hub
from langchain.document_loaders import PyPDFLoader
import os 

#insert openapi key
os.environ["OPENAI_API_KEY"] = ""

question = "what is the path to the bank"

data = []
current_dir = os.getcwd()
for file in os.listdir(current_dir):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(current_dir, file)
        loader = PyPDFLoader(pdf_path)
        data.extend(loader.load())

### split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)
vectorstore = ElasticsearchStore.from_documents(
    documents=docs,
    index_name="elastic_search_vectorstore",
    es_api_key="",
    embedding=OpenAIEmbeddings(),
    es_cloud_id="3438e9938373428281c9861abac4c00c:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGVjOTBhOWEzMmY5MTQwZjI4OTAwOGMzMjhiZTlkZmI2JGQ5ODE5NWU4MmJlMDQwNzNhOWYxZDAwZmYzM2YzMTZk"

)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print(retriever.get_relevant_documents(query=question)[0])


