
from smolagents import Tool
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from torch.backends import mps
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

def get_emebdding_model(model_name="Alibaba-NLP/gte-large-en-v1.5"):
    device ="mps" if mps.is_available() else "cpu"
    encode_kwargs={"device":device,'normalize_embeddings':True}
    model_kwargs={"trust_remote_code":True}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)
    return embeddings

def embeddings_vector_store(model_name="Alibaba-NLP/gte-large-en-v1.5"):
    embeddings= get_emebdding_model(model_name)
   
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    return vector_store



class RetrieverTool(Tool):


    name = "retriever_tool"
    description = (
        "Checks semantic similarity of the input query against known queries in the knowledge base. "
        "If any retrieved query is too similar to the new query (score > gamma_max) or too different "
        "(score < gamma_min), it raises ValueError. Otherwise, it returns an acceptance message."
    )

    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to check against the existing knowledge base. This input must be a string"
            ),
        }
    }
    output_type = "string"
    def __init__(self, vectordb: VectorStore, gamma_max: float = 0.9, gamma_min: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.gamma_max= gamma_max
        self.gamma_min= gamma_min

    def forward(
        self,
        query: str,
    ) -> str:
        # Basic checks
        assert isinstance(query, str), "Your search query must be a string."

        if not len(self.vectordb.index_to_docstore_id):
            return f'Query: "{query}" is accepted (no existing records).'
        # Perform similarity search with filter
        docs = self.vectordb.similarity_search_with_score(
            query=query,
            k=5,
        )

        doc_min,doc_max=[],[]
        for doc in docs :
            print(doc[1])
            if doc[1] > self.gamma_max:
                doc_max.append(doc[0].page_content)
            elif doc[1] < self.gamma_min:
                doc_min.append(doc[0].page_content)

        
        if doc_max:
            raise ValueError(f"Wrong, Retrieved queries are too similar to recent query (Queries: {doc_max}) rewrite the question/query.")
    
        if doc_min:
            raise ValueError (f"Wrong, Retrieved queries are too different from recent query (Queries: {doc_min}) rewrite the question/query.")
        
        return f'Query: "{query}" is accepted (no existing records).'


