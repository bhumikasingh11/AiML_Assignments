# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
# Bug 1 fix: replaced OpenAIEmbeddings (text-embedding-3-large, dim=3072) with
# HuggingFaceEmbeddings (all-MiniLM-L6-v2, dim=384) to match the Pinecone index.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# retrieval
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)
results = retriever.invoke("what is retrieval augmented generation?")

# show results
print("RESULTS:")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")