import pandas as pd
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
import pickle

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model = INSTRUCTOR('hkunlp/instructor-xl')
# instruction = "Represent the module description

fom_degrees = pd.read_parquet("fom_material_t3.parquet")

# get the modules for each degree
degree_modules = fom_degrees.groupby("studiengang")["doc_id"].unique()

# create empty embedding list
embedding_list = []
# loop through degrees and read in modules
for degree in degree_modules.index.to_list():
    # loop through degrees
    for module in degree_modules[degree]:
        # create creader object
        reader = PdfReader(f"Moduluebersicht/{module}/{module}.pdf")
        module_text = ""
        for page in reader.pages:
            module_text += page.extract_text() + "\n"
        
        # create embedding from module_text
        embedding_list.append({f"{degree}_{module}": model.encode(module_text)})
    # print progress
    print(degree)


# save embeddings
with open("module_embeddings.pkl", "wb") as fp:
    pickle.dump(embedding_list, fp)


# create degree embeddings
#create set of unique degrees
unique_degrees = set([list(element.keys())[0].split('_')[0] for element in embedding_list])

mean_degree_embeddings_list = []

# iterate through degrees
for degree in unique_degrees:
    degree_embedding_list = []
    for module in embedding_list:
        if degree in list(module.keys())[0]:
            degree_embedding_list.append(list(module.values())[0])
    mean_degree_embeddings_list.append({degree: np.mean(degree_embedding_list, axis=0)})

# save embeddings
with open("degree_embeddings.pkl", "wb") as fp:
    pickle.dump(mean_degree_embeddings_list, fp)