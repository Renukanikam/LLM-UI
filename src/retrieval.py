from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load the Hugging Face dataset
dataset = load_dataset("mrtoy/mobile-ui-design")

# Pre-process the dataset
def embed_dataset(dataset):
    ui_texts = [data['description'] for data in dataset['train']]
    ui_embeddings = embedder.encode(ui_texts, convert_to_tensor=True)
    return ui_texts, ui_embeddings

ui_texts, ui_embeddings = embed_dataset(dataset)

# Retrieve relevant UIs based on the query
def retrieve_similar_ui(query, top_k=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, ui_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    retrieved_ui = []
    for score, idx in zip(top_results[0], top_results[1]):
        retrieved_ui.append({"description": ui_texts[idx], "score": score.item()})
    
    return retrieved_ui
