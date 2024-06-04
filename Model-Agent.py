from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor


import os 

#insert openapi key
os.environ["OPENAI_API_KEY"] = ""



question = "How do I set a new password"

###load data
dataset_name = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
page_content_column = "response"  # or any other column you're interested in

loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()

### split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


retriever_tool = create_retriever_tool(
    retriever,
    "banking_database_search",
    "Search for information about banking. For any questions about banking, you may use this tool!",
)
tools = [retriever_tool]



llm = ChatOpenAI(model_name="gpt-3.5-turbo")
prompt = hub.pull("hwchase17/openai-functions-agent")


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": input()})






