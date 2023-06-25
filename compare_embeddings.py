import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle

# read in embeddings
job_embeddings = np.load("job_embeddings.npy")

with open("module_embeddings.pkl", "rb") as fp:
    module_embeddings_dicts = pickle.load(fp)

# load job list data
data = pd.read_parquet("joblist_full.parquet")

# create full array of module embeddings
# module_embeddings = [list(element.values())[0] for element in module_embeddings_dicts]

# create mean vectors of every degree

# create list of degrees

# iterate through list of degrees and save mean vecotrs



def get_degrees_for_job(job_embedding: np.array, degree_embeddings: np.array, n_degrees:int = 5) -> pd.DataFrame:
    distance = util.dot_score(job_embedding, degree_embeddings).tolist()

    distance = [element[0] for element in distance]
    distance_sorted = distance.copy()
    distance_sorted.sort(reverse=True)

    # get index of best fit
    ind_list = [distance.index(element) for element in distance_sorted[:n_degrees]]

    # return best deg
    # return data.loc[ind_list,["title_txt", "location", "company_name", "branche", "url"]]

# print(util.dot_score(job_embeddings[0], module_embeddings))