import os

from haystack.document_stores import FAISSDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs
from haystack.nodes import EmbeddingRetriever, FARMReader
from haystack.pipelines import DocumentSearchPipeline, ExtractiveQAPipeline
from haystack.utils import print_documents, print_answers
from haystack.document_stores import PineconeDocumentStore

import torch

doc_dir = "data/tutorial6"
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

document_store = PineconeDocumentStore(
    api_key='',
    index='tester',
    similarity="cosine",
    environment= "",
    embedding_dim=768
)

document_store.write_documents(docs)

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="kamalkraj/BioSimCSE-BioLinkBERT-BASE", devices=[torch.device('cuda:0'), torch.device('cuda:1')])

document_store.update_embeddings(retriever)

reader = FARMReader(model_name_or_path="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", use_gpu=True, devices= [torch.device('cuda:0'), torch.device('cuda:1')])
#pipe = DocumentSearchPipeline(retriever)
pipe = ExtractiveQAPipeline(retriever, reader)

prediction = pipe.run(
    query = "Find me everything about hydrogen bonding", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)

print_answers(prediction, details="minimum")

# query = "Find all information related to hydrogen bonding"
# result = pipe.run(query, params={"Retriever": {"top_k": 6}})
# print_documents(result, max_text_len=200)