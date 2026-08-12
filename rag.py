from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def create_documents(scraped_documents):
    documents = []

    for item in scraped_documents:

        document = Document(
            page_content=item["content"],
            metadata={
                "source": item["url"]
            }
        )

        documents.append(document)

    return documents


def split_documents(documents):
    return text_splitter.split_documents(documents)


def create_vector_store(chunks):
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )


def retrieve_documents(retriever, query):
    return retriever.invoke(query)