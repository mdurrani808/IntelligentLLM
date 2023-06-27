from haystack.document_stores import FAISSDocumentStore
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http
from haystack.nodes import EmbeddingRetriever, DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline
from haystack.utils import print_answers, print_documents
from haystack.document_stores import PineconeDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, SentenceTransformersRanker
from haystack import Pipeline

document_store = FAISSDocumentStore(
    faiss_index_factory_str="Flat",sql_url="sqlite://"
)



# Let's first get some files that we want to use
doc_dir = "1BRS_data"


# Convert files to dicts
docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the dicts containing documents to our DB.
document_store.write_documents(docs)
retriever = BM25Retriever(document_store)
#retriever = DensePassageRetriever(
 ##  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
  #  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
#)
document_store.update_embeddings(retriever)
document_store.save(index_path="my_index.faiss")
ranker = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", use_gpu=True)
p = Pipeline()
p.add_node(component=retriever, name="BM25Retriever", inputs=["Query"])
p.add_node(component=ranker, name="Ranker", inputs=["BM25Retriever"])
#reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2-distilled", use_gpu=True)


#pipe = ExtractiveQAPipeline(reader, retriever)

#query = "Find all information related to hydrogen bonding"
#result = pipe.run(query, params={"Retriever": {"top_k": 6}})
#print_documents(result, max_text_len=200)



query = "Find all information related to hydrogen bonding"
prediction = p.run(
    query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
)
print_answers(prediction, details="minimum")


#print_documents(result, max_text_len=200)
