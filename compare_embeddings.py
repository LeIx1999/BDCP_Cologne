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
# filter out duplicates
data = data.drop_duplicates(subset=["title_txt"])

#create set of unique degrees
unique_degrees = set([list(element.keys())[0].split('_')[0] for element in module_embeddings_dicts])

mean_degree_embeddings_list = []

for degree in unique_degrees:
    degree_embedding_list = []
    for module in module_embeddings_dicts:
        if degree in list(module.keys())[0]:
            degree_embedding_list.append(list(module.values())[0])
    mean_degree_embeddings_list.append({degree: np.mean(degree_embedding_list, axis=0)})


# create full array of module embeddings
# module_embeddings = [list(element.values())[0] for element in module_embeddings_dicts]

# create mean vectors of every degree

# create list of degrees

# iterate through list of degrees and save mean vecotrs



def get_degrees_for_job(job_embedding: np.array, degree_embeddings: list, n_degrees:int = 5) -> pd.DataFrame:
    #calculate similarity of two vectors
    degree_embeddings_values = [list(element.values())[0] for element in degree_embeddings]
    distance = util.dot_score(degree_embeddings_values, job_embedding).tolist()

    distance = [element[0] for element in distance]
    distance_sorted = distance.copy()
    distance_sorted.sort(reverse=True)

    # get index of best fit
    ind_list = [distance.index(element) for element in distance_sorted[:n_degrees]]
    most_similar_degrees = [list(degree_embeddings[index].keys())[0] for index in ind_list]
    degrees_with_distance = list(zip(most_similar_degrees, distance_sorted[:n_degrees]))

    return degrees_with_distance

    # return best deg
    # return data.loc[ind_list,["title_txt", "location", "company_name", "branche", "url"]]

result_jobs_degrees_comparison = []
for i in range(len(job_embeddings[:9])):
    result_jobs_degrees_comparison.append({data.iloc[i,1]: get_degrees_for_job(job_embeddings[i], mean_degree_embeddings_list)})

1+1
