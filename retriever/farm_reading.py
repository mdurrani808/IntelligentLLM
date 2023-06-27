from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import BM25Retriever, DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import torch
from haystack.utils import convert_files_to_docs
from haystack.utils import print_answers
from haystack.utils import convert_files_to_docs
from haystack.nodes import FARMReader


# Let's first get some files that we want to use
doc_dir = "1BRS_data"


# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir,split_paragraphs=True)


document_store = InMemoryDocumentStore(use_bm25=True)# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)

dpr = DensePassageRetriever(document_store=document_store,)
#retriever = BM25Retriever(document_store,top_k=50, all_terms_must_match=True)
#reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True, top_k_per_candidate=50, context_window_size=1250)
#pipeline= ExtractiveQAPipeline(reader=reader, retriever=retriever)


query = "hydrogen bonds bonding"
result = pipeline.run(query=query, params={"Retriever": {"top_k": 50}, "Reader": {"top_k": 50}})
print_answers(result,details="minimum")