# main.py
"""
Orquestra pipeline: processamento -> vetorizacao -> clustering -> UMAP -> salvar resultados e figuras.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from umap import UMAP

from processamento_dados import carregar_transformar, salvar
from similaridade_textual import computar_tfidf, computar_sbert_embeddings, salvar_objeto



OUTPUT_DIR = "outputs"

def rodar_pipeline(
    input_path=None,
    n_clusters=12,
    use_sbert=False,
    sbert_model="modelos/sbert_paraphrase_multilingual",
    random_state=42
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 1. Carregar / processar dados
    df = carregar_transformar(input_path) if input_path else carregar_transformar()
    salvar(df)  # salva CSV/PKL padronizado

    corpus = df["ementa"].astype(str).tolist()

    # 2. TF-IDF
    vect, X_tfidf = computar_tfidf(corpus, max_features=20000)
    salvar_objeto(vect, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
    from scipy import sparse
    sparse.save_npz(os.path.join(OUTPUT_DIR, "tfidf_matrix.npz"), X_tfidf)

    # 3. KMeans sobre TF-IDF (sparse -> usar dense reduced é melhor; aqui usamos KMeans direto sobre TF-IDF densificado por SVD opcional)
    # Recomendo usar TruncatedSVD para reduzir dimensionalidade antes do KMeans (por performance)
    from sklearn.decomposition import TruncatedSVD
    print("[i] Reduzindo TF-IDF com TruncatedSVD...")
    svd = TruncatedSVD(n_components=128, random_state=random_state)
    X_reduced = svd.fit_transform(X_tfidf)
    joblib.dump(svd, os.path.join(OUTPUT_DIR, "svd_tfidf.pkl"))

    print(f"[i] Treinando KMeans (k={n_clusters}) sobre representação TF-IDF reduzida...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_reduced)
    df["cluster_kmeans_tfidf"] = labels
    joblib.dump(kmeans, os.path.join(OUTPUT_DIR, "kmeans_tfidf.pkl"))

    # 4. UMAP para visualização (das reduções)
    umap = UMAP(n_components=2, random_state=random_state)
    embedding_2d = umap.fit_transform(X_reduced)
    df["x_umap_tfidf"] = embedding_2d[:,0]
    df["y_umap_tfidf"] = embedding_2d[:,1]
    joblib.dump(umap, os.path.join(OUTPUT_DIR, "umap_tfidf.pkl"))

    # 5. Se SBERT pedido: calcular embeddings e clusters (opcional)
    if use_sbert:
        try:
            embs = computar_sbert_embeddings(corpus, model_name=sbert_model)
            np.save(os.path.join(OUTPUT_DIR, "sbert_embeddings.npy"), embs)
            kmeans_sbert = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels_sbert = kmeans_sbert.fit_predict(embs)
            df["cluster_kmeans_sbert"] = labels_sbert
            joblib.dump(kmeans_sbert, os.path.join(OUTPUT_DIR, "kmeans_sbert.pkl"))

            umap_s = UMAP(n_components=2, random_state=random_state)
            emb_s_2d = umap_s.fit_transform(embs)
            df["x_umap_sbert"] = emb_s_2d[:,0]
            df["y_umap_sbert"] = emb_s_2d[:,1]
            joblib.dump(umap_s, os.path.join(OUTPUT_DIR, "umap_sbert.pkl"))
        except Exception as e:
            print("[warn] SBERT falhou:", e)

    # 6. Salvar resultados finais
    df.to_csv(os.path.join(OUTPUT_DIR, "projetos_PL_clusters_umap.csv"), index=False)
    print("[i] Resultados salvos em:", os.path.join(OUTPUT_DIR, "projetos_PL_clusters_umap.csv"))

    # 7. Gerar figuras simples
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df["x_umap_tfidf"], df["y_umap_tfidf"], c=df["cluster_kmeans_tfidf"], s=8)
    plt.title("UMAP (TF-IDF reduzido) + KMeans")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_tfidf_kmeans.png"), dpi=200)
    plt.close()
    print("[i] Figura salva:", os.path.join(OUTPUT_DIR, "umap_tfidf_kmeans.png"))

    return df

if __name__ == "__main__":
    df_out = rodar_pipeline(use_sbert=False, n_clusters=12)
