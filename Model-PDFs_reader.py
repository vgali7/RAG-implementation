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
from langchain.document_loaders import PyPDFLoader
import os 

#insert openapi key
os.environ["OPENAI_API_KEY"] = ""

question = "can you help me set a new password"

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
vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


llm = ChatOpenAI(model_name="gpt-3.5-turbo")
prompt = hub.pull("rlm/rag-prompt")
example_messages = prompt.invoke(
    {"context": retriever, "question": question}
).to_messages()
print(example_messages[0].content)
print('------')

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#custom_rag_prompt = PromptTemplate.from_template("Custom Prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt  #Custom prompt from custom_rag_prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)





