import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from RAG import PreprocessingPipline
from langchain_core.vectorstores import VectorStore


class Vec:
    def __init__(self):
        self.client = chromadb.PersistentClient(path='testing-var-nctx')
        self.chroma_client = Chroma(client=self.client, collection_name='testing-512-nctx-char-splitter',
                                    embedding_function=SentenceTransformerEmbeddings(
                                        model_name="sentence-transformers/all-MiniLM-L6-v2")
                                    )

    def _populating_database(self) -> None:
        pipeline = PreprocessingPipline("20240716890312078-401-402.pdf").get_each_page()
        self.chroma_client.add_documents(documents=pipeline)
        return None

    def get_vectorstore(self) -> VectorStore:
        return self.chroma_client


if __name__ == "__main__":  # Debugging
    clients = chromadb.PersistentClient(path='testing-var-nctx')
    collection = clients.get_collection(name='testing-512-nctx-char-splitter')
    print(collection.count())
