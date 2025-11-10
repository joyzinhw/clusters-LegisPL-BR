# avaliacao_clusters.py
"""
Avaliação de desempenho e interpretação dos clusters gerados pelo pipeline LegisPL-BR.
Gera métricas de coesão, separação, top termos e alinhamento político.
"""

import os
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm  # barra de progresso

OUTPUT_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "projetos_PL_clusters_umap.csv")

def avaliar_clusters():
    print("[i] Carregando dados e modelos...")
    df = pd.read_csv(CSV_PATH)

    # carregar representações reduzidas (TF-IDF -> SVD)
    svd = joblib.load(os.path.join(OUTPUT_DIR, "svd_tfidf.pkl"))
    X_tfidf = sps.load_npz(os.path.join(OUTPUT_DIR, "tfidf_matrix.npz"))
    X_reduced = svd.transform(X_tfidf)

    # === 1. Métricas quantitativas ===
    print("[i] Calculando métricas de qualidade...")
    sil_score = silhouette_score(X_reduced, df["cluster_kmeans_tfidf"])
    db_score = davies_bouldin_score(X_reduced, df["cluster_kmeans_tfidf"])
    print(f"🟢 Silhouette Score: {sil_score:.3f}")
    print(f"🟣 Davies-Bouldin Index: {db_score:.3f}")

    # === 2. Top termos TF-IDF por cluster ===
    print("[i] Extraindo top termos representativos...")
    vectorizer = joblib.load(os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl"))
    terms = np.array(vectorizer.get_feature_names_out())

    X_dense = X_tfidf.tocsr()
    top_terms_per_cluster = {}

    for k in tqdm(sorted(df["cluster_kmeans_tfidf"].unique()), desc="Clusters processados"):
        idx = (df["cluster_kmeans_tfidf"] == k).to_numpy()  # CORREÇÃO: converte para array
        cluster_mean = X_dense[idx].mean(axis=0)
        top_terms = terms[np.argsort(cluster_mean.A1)[::-1][:15]]
        top_terms_per_cluster[k] = top_terms.tolist()

    # salva em CSV
    pd.DataFrame([
        {"cluster": k, "top_termos": ", ".join(v)}
        for k, v in top_terms_per_cluster.items()
    ]).to_csv(os.path.join(OUTPUT_DIR, "top_termos_por_cluster.csv"), index=False)
    print("[i] ✅ Top termos salvos em: outputs/top_termos_por_cluster.csv")

    # === 3. Análise política ===
    print("[i] Calculando alinhamento político (blocos ideológicos)...")
    blocos_tab = pd.crosstab(df["cluster_kmeans_tfidf"], df["bloco_partidario"], normalize="index") * 100
    blocos_tab.to_csv(os.path.join(OUTPUT_DIR, "alinhamento_politico.csv"))
    print("[i] ✅ Alinhamento político salvo em: outputs/alinhamento_politico.csv")

    # === 4. Visualizações ===
    print("[i] Gerando gráficos...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df["x_umap_tfidf"], df["y_umap_tfidf"],
                          c=df["cluster_kmeans_tfidf"], s=8, cmap="tab20")
    plt.title("UMAP (TF-IDF + KMeans) - Clusters de Proposições")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "umap_clusters_tfidf.png"), dpi=200)
    plt.close()
    print("[i] ✅ Gráfico salvo: outputs/umap_clusters_tfidf.png")

    fig = px.scatter(
        df, x="x_umap_tfidf", y="y_umap_tfidf",
        color="bloco_partidario",
        hover_data=["ementa", "cluster_kmeans_tfidf", "partido_principal"],
        title="Projeção UMAP colorida por bloco ideológico"
    )
    fig.write_html(os.path.join(OUTPUT_DIR, "umap_blocos_interativo.html"))
    print("[i] ✅ Gráfico interativo salvo: outputs/umap_blocos_interativo.html")

    # === 5. Relatório resumo ===
    resumo = pd.DataFrame({
        "silhouette_score": [sil_score],
        "davies_bouldin_index": [db_score],
        "n_clusters": [df["cluster_kmeans_tfidf"].nunique()],
        "n_proposicoes": [len(df)]
    })
    resumo.to_csv(os.path.join(OUTPUT_DIR, "resumo_metricas.csv"), index=False)
    print("[✅] Avaliação concluída!")
    print("📄 Relatórios e gráficos salvos em 'outputs/'")

if __name__ == "__main__":
    avaliar_clusters()
