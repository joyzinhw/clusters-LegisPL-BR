import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import matplotlib.pyplot as plt

dataset = pd.read_excel('projetos_PL_2022_2024_limpo.xlsx', engine='openpyxl')
texts = dataset['ementa_limpa'].tolist()
print(f"Total de registros: {len(texts)}")

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

def get_bert_embeddings(texts, batch_size=16):
    """Gera embeddings usando BERT para português."""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            # Usa a média dos embeddings dos tokens como embedding da sentença
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)

embeddings = get_bert_embeddings(texts)
print(f"Embeddings shape: {embeddings.shape}")

pca = PCA(n_components=100, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings_pca)

score = silhouette_score(embeddings_pca, clusters)
print(f"Silhouette Score (K-Means): {score:.3f}")

reducer = umap.UMAP(random_state=42)
embeddings_umap = reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('Visualização 2D dos Clusters (UMAP)')
plt.xlabel('Componente UMAP 1')
plt.ylabel('Componente UMAP 2')
plt.grid(True)
plt.show()

dataset['cluster'] = clusters
output_file = 'projetos_PL_2022_2024_clusterizado_bert_pt.xlsx'
dataset.to_excel(output_file, index=False, engine='openpyxl')
print(f"Dataset clusterizado salvo em: {output_file}")
