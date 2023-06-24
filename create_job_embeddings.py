import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = INSTRUCTOR('hkunlp/instructor-xl')
instruction = "Represent the job offer:"

data = pd.read_parquet("joblist_full.parquet")

# filter out duplicates
data = data.drop_duplicates(subset=["title_txt"])

job_data = data["job_txt"].to_list()
instruction_job_list = [[instruction, element] for element in job_data]

# iterate through the jobs and create embeddings
embedding_list = []
with open("embeddings.txt", "ab") as f:
    # save after every job
    for element in instruction_job_list:
        job_data_embedding = model.encode([element])
        np.savetxt(f, job_data_embedding)
        print(instruction_job_list.index(element))



