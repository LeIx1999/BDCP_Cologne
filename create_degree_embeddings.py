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




