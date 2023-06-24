import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from sentence_transformers import SentenceTransformer, util
import numpy as np
# load data and job embeddings
data = pd.read_parquet("joblist_full.parquet")

# filter for only unique job texts

job_embeddings = np.load("job_embeddings.npy")

# load a sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_best_job_fit(job_embeddings: np.array, question:str, n_jobs:int = 5) -> pd.DataFrame:
    question_embedding = model.encode(question)
    distance = util.dot_score(job_embeddings, question_embedding).tolist()

    distance = [element[0] for element in distance]
    distance_sorted = distance.copy()
    distance_sorted.sort(reverse=True)

    # get index of best fit
    ind_list = [distance.index(element) for element in distance_sorted[:n_jobs]]

    # return best job fit
    return data.loc[ind_list,["title_txt", "location", "company_name", "branche", "url"]]


res = get_best_job_fit(job_embeddings=job_embeddings, question="Ich suche einen Job als Data Engineer, wo ich mit der Google Cloud und mit Python arbeiten kann.")

print(res)