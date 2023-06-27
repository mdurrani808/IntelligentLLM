from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import convert_files_to_docs
from haystack.utils import print_documents
from typing import Dict, Any, List
from haystack import Pipeline

query = "residues H-bonds hydrogen bond bonding"
doc_dir = "data"
docs = convert_files_to_docs(dir_path=doc_dir, split_paragraphs=True)
document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(docs)
retriever = BM25Retriever(document_store,top_k=100, all_terms_must_match=True)
ranker = SentenceTransformersRanker(model_name_or_path="naver/splade_v2_max", top_k=10, devices=["cuda:0","cuda:1"])

p = Pipeline()
p.add_node(component=retriever, name="BM25Retriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["BM25Retriever"])

result = p.run(
    query=query
)

print_documents(result)
