import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import matplotlib.pyplot as plt
from Pre_Process import process
from generate_multi_query import generate_multi_query
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from generate_response import generate_response


load_dotenv()
openai_key = os.getenv("openai_api_key")
client = OpenAI(api_key=openai_key)
embedding_function = SentenceTransformerEmbeddingFunction()
file_path = "./data/microsoft_annual_report.pdf"
collection_name = "microsoft-collection"
chroma_collection = process(file_path, collection_name)


original_query = ("What were the most important factors that contributed to increases in revenue?")
aug_queries = generate_multi_query(client, original_query)
queries = [original_query] + aug_queries  

# for query in queries:
#     print(query,"\n")
# print("-" *100)    

results = chroma_collection.query(
    query_texts=queries, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

scores = cross_encoder.predict(pairs)

# print("Scores:")
# for score in scores:
#     print(score)

# print("New Ordering:")
# for o in np.argsort(scores)[::-1]:
#     print(o)

top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]
context = "\n\n".join(top_documents)




res = generate_response(client=client, query=original_query, context=context)
print("Final Answer:")
print(res)