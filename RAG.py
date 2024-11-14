from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from llm_config import myLLM
import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
import re as regex
from typing import List
from langchain_core.output_parsers import StrOutputParser
from typing import Union

client = chromadb.PersistentClient(path="testing-var-nctx")


class PreprocessingPipline:
    def __init__(self, path: Union[str, None] = None):
        self.loader = PyMuPDFLoader(path) if path else None
        self.splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        self.ve_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                                      multi_process=True)
        self.vectorstore = Chroma(client=client, collection_name='testing-512-nctx-char-splitter',
                                  embedding_function=SentenceTransformerEmbeddings(
                                      model_name="sentence-transformers/all-MiniLM-L6-v2")
                                  )

    def get_each_page(self) -> List[Document]:
        _pattern = r'___________________________________.*'
        _pages = self.loader.load()
        docs = []
        for doc in _pages:
            # start_index = re.search(_pattern2, doc.page_content).span()[1]  --> Removes the header
            end_index = regex.search(_pattern, doc.page_content)  # Remove the footer
            if end_index is not None:
                doc.page_content = doc.page_content[:end_index.span()[0]]
            split = self.splitter.split_text(doc.page_content)
            docs += [Document(page_content=each, metadata={}) for each in split]
            # print(docs)  --> for debugging lmao
        return docs

    def get_related_documents(self, query: str) -> List[Document]:
        retriever = self.vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 2})
        docs = retriever.invoke(input=query)
        return docs


class Generation:
    def __init__(self):
        pass

    @staticmethod
    def generate_answer(user_query: str) -> str:
        llm = myLLM().get_configured_llm()
        context = PreprocessingPipline().get_related_documents(query=user_query)
        default_template = (f"You are a chatbot that helps the users with the problem they face. You are to give the "
                            f"answers based on the Indian constitution. If the answer cannot be given then just say "
                            f"you don't know. Give the answer in a well defined manner and tell the appropriate laws "
                            f"that are applicable for the given scenario. Just reply and don't ask the user for "
                            f"additional information.\nQuestion: {user_query} \nContext: {context[0]}\n{context[1]} "
                            f"\n")
        answer = llm.invoke(default_template)
        parser = StrOutputParser()
        return parser.parse(answer)


if __name__ == "__main__":
    my = PreprocessingPipline()
    m = input("Enter the Query: ")
    ducky = my.get_related_documents(query=m)
    instance = Generation()
    gen = instance.generate_answer(m)
    print(gen)
    exit()
    # Testing the retrieval system
    # for i in range(len(ducky)):
    #     print(f"Document No: {i+1}")
    #     print(ducky[i].page_content)
    #     print("---------------------------------------------------------")
