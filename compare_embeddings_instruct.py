import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from embedding_comparer import Embedding_Comparer

Comparer = Embedding_Comparer(model=INSTRUCTOR('hkunlp/instructor-xl'), 
                                   degree_embeddings=pd.read_pickle("data/degree_embeddings.pkl"),
                                   job_embeddings=np.loadtxt("data/job_embeddings_instruct.txt"), 
                                   job_data=pd.read_parquet("data/joblist_full.parquet"))
# get top 5 degrees for every job
result_jobs_degrees_comparison = []
for i in range(len(Comparer.job_embeddings[:9])):
    result_jobs_degrees_comparison.append({Comparer.job_data.iloc[i,1]: Comparer.get_degrees_for_job(list(Comparer.job_embeddings[i].values())[0])})


# get degree for description
most_similar_degrees = Comparer.get_degrees_for_input("Ich kann gut in Java programmieren und kann gut mit Computern umgehen")
