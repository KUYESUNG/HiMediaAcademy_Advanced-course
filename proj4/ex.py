# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP 3
# The sentences to encode
sentences1 = "오늘의 날씨는 30도가 넘어가 매우 더운 날씨입니다"
sentences2 = "회사에 가는게 너무 싫어요 그냥 집에서 잠이나 자고싶다"

# STEP 4
# 2. Calculate embeddings by calling model.encode()
embedding1 = model.encode(sentences1)
embedding2 = model.encode(sentences2)
print(embedding1.shape)
print(embedding2.shape)
# [3, 384]

# STEP 5
# 3. Calculate the embedding similarities
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])