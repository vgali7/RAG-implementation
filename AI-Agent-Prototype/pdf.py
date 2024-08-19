import os
os.environ["OPENAI_API_KEY"] = ""
from llama_index.core import load_index_from_storage, StorageContext, VectorStoreIndex
from llama_index.readers.file import PDFReader

def get_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
        print("Building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


current_dir = os.getcwd()
america_path = os.path.join(current_dir, "data", "America.pdf")
america_pdf = PDFReader().load_data(file=america_path)

america_index = get_index(america_pdf, "america")

america_engine = america_index.as_query_engine()

