import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pickle
import sys
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


class Embedding_Comparer():
    def __init__(self, model, degree_embeddings:list, job_embeddings:list, job_data:pd.DataFrame) -> None:
        self.model = model
        self.degree_embeddings = degree_embeddings
        self.job_data = job_data.drop_duplicates(subset=["title_txt"])
        self.job_embeddings = job_embeddings_list = []
        for i in range(len(job_embeddings)):
            job_embeddings_list.append({f"{self.job_data.iloc[i, 1]}_{self.job_data.iloc[i, 0]}": job_embeddings[i]})

    def _create_embeddings_from_input(self, input: str) -> str:
        return self.model.encode(input)

    def get_degrees_for_job(self, job_embedding: np.array,  n_degrees:int = 5) -> list:
        #calculate similarity of two vectors
        degree_embeddings_values = [list(element.values())[0]for element in self.degree_embeddings]
        distance_list = []
        for degree in degree_embeddings_values:
            distance_list.append(cosine_similarity(degree, job_embedding.reshape(1, -1)))

        distance = [float(element) for element in distance_list]
        distance_sorted = distance.copy()
        distance_sorted.sort(reverse=True)

        # get index of best fit
        ind_list = [distance.index(element) for element in distance_sorted[:n_degrees]]
        most_similar_degrees = [list(self.degree_embeddings[index].keys())[0] for index in ind_list]
        degrees_with_distance = list(zip(most_similar_degrees, distance_sorted[:n_degrees]))

        return degrees_with_distance

    def get_degrees_for_input(self, input: str, n_degrees:int = 5) -> list:
        # create embeddings from input
        input_embedding = self._create_embeddings_from_input(input)
        #calculate similarity of two vectors
        degree_embeddings_values = [list(element.values())[0] for element in self.degree_embeddings]
        distance_list = []
        for degree in degree_embeddings_values:
            distance_list.append(cosine_similarity(degree, input_embedding.reshape(1, -1)))

        distance = [float(element) for element in distance_list]
        distance_sorted = distance.copy()
        distance_sorted.sort(reverse=True)

        # get index of best fit
        ind_list = [distance.index(element) for element in distance_sorted[:n_degrees]]
        most_similar_degrees = [list(self.degree_embeddings[index].keys())[0] for index in ind_list]
        degrees_with_distance = list(zip(most_similar_degrees, distance_sorted[:n_degrees]))

        return degrees_with_distance
    
