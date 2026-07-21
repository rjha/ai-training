import os 
import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 1. Generate coordinates
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


file_name = "./data/words01.txt"
print(f"Reading vocabulary from {file_name}...")
with open(file_name, "r", encoding="utf-8") as f:
    all_words = [line.strip() for line in f if line.strip()]

# Remove duplicates 
all_words = list(dict.fromkeys(all_words))
# shuffle to eliminate any hidden order
random.seed(42)
random.shuffle(all_words)

embeddings = model.encode(all_words)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# 2. Structure data beautifully for JavaScript
web_data = []
for i, word in enumerate(all_words):
    is_hindi = any(ord(char) > 127 for char in word)
    web_data.append({
        "word": word,
        "x": float(embeddings_2d[i, 0]),
        "y": float(embeddings_2d[i, 1]),
        "language": "hindi" if is_hindi else "english"
    })

# Write a JS file directly so you don't even have to deal with JSON fetch/CORS errors!
with open("embedding.js", "w", encoding="utf-8") as f:
    f.write(f"const embeddingData = {json.dumps(web_data, ensure_ascii=False)};")

print("Generated 'embedding.js' successfully!")
