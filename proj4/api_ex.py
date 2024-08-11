# STEP 1
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Form

# STEP 2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

app = FastAPI()

@app.post("/ex/")
async def ex(sentences1: str = Form(), sentences2: str = Form()):

    # STEP 3
    # TEXT 1
    # TEXT 2

    # STEP 4
    embedding1 = model.encode(sentences1)
    embedding2 = model.encode(sentences2)
    similarities = model.similarity(embedding1, embedding2)

    # STEP 5
    return {"similarities": similarities}