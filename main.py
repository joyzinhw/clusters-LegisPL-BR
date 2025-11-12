# main.py
# Joyce Moura - Clusterização sem visualizações (salva CSVs)

import pandas as pd
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from limpeza import carregar_dados
from similaridades import gerar_embeddings, calcular_similaridade_media

# ===============================
# 1️⃣ Carregar dados
# ===============================
df = carregar_dados("LegisPL-BR-main/dados/projetos_PL_2021_2024_completo.xlsx")

# Criar ID único para cada ementa (se ainda não existir)
df = df.reset_index().rename(columns={'index': 'ID'})

# ===============================
# 2️⃣ Remover linhas com erro no tema
# ===============================
df = df[~df['tema_principal'].str.startswith("Erro")].reset_index(drop=True)

# ===============================
# 3️⃣ Gerar embeddings e similaridades
# ===============================
model_name = 'neuralmind/bert-base-portuguese-cased'
all_sub_ementas = sum(df['sub_ementas'].tolist(), [])
model, embeddings_sub_ementas = gerar_embeddings(model_name, all_sub_ementas)
df = calcular_similaridade_media(df, model, embeddings_sub_ementas)

# ===============================
# 4️⃣ Clusterização automática com KMeans
# ===============================
sil_scores = {}
X = df['similaridade'].values.reshape(-1, 1)

for k in range(5, 51, 5):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sil_scores[k] = silhouette_score(X, labels)

best_k = max(sil_scores, key=sil_scores.get)

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans_final.fit_predict(X)

# ===============================
# 5️⃣ Métricas de desempenho
# ===============================
silhouette = silhouette_score(X, df['cluster'])
davies = davies_bouldin_score(X, df['cluster'])
calinski = calinski_harabasz_score(X, df['cluster'])

label_encoder = LabelEncoder()
y_true = label_encoder.fit_transform(df['tema_principal'])
y_pred = df['cluster']

ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)

# Salvar métricas
df_metricas = pd.DataFrame([{
    'Silhouette': silhouette,
    'Davies-Bouldin': davies,
    'Calinski-Harabasz': calinski,
    'ARI': ari,
    'NMI': nmi,
    'Melhor k': best_k
}])
df_metricas.to_csv("metricas_clusters_semanticos.csv", index=False)
print(df_metricas)
# ===============================
# 6️⃣ Resumo por cluster (tema predominante)
# ===============================
cluster_resumo = []
for cluster_id in sorted(df['cluster'].unique()):
    temas_lista = sum(df[df['cluster'] == cluster_id]['tokens_temas'].tolist(), [])
    if len(temas_lista) == 0:
        continue
    temas_contagem = Counter(temas_lista)
    tema_predominante, freq = temas_contagem.most_common(1)[0]
    total = sum(temas_contagem.values())
    percentual = (freq / total) * 100
    cluster_resumo.append({
        'Cluster': cluster_id,
        'Tema Predominante': tema_predominante,
        'Frequência': freq,
        'Percentual': round(percentual, 2)
    })

df_cluster_resumo = pd.DataFrame(cluster_resumo)
df_cluster_resumo.to_csv("resumo_clusters_por_tema_predominante.csv", index=False)
print(df_cluster_resumo)
# ===============================
# 7️⃣ Agrupar clusters pelo mesmo tema
# ===============================
tema_clusters = defaultdict(list)
for _, row in df_cluster_resumo.iterrows():
    tema_clusters[row['Tema Predominante']].append({
        'Cluster': row['Cluster'],
        'Frequência': row['Frequência'],
        'Percentual': row['Percentual']
    })

agrupado = []
for tema, clusters_info in tema_clusters.items():
    total_freq = sum(c['Frequência'] for c in clusters_info)
    percentual_total = round(total_freq / sum(df_cluster_resumo['Frequência']) * 100, 2)
    agrupado.append({
        'Tema': tema,
        'Clusters': [c['Cluster'] for c in clusters_info],
        'Frequência Total': total_freq,
        'Percentual Total': percentual_total
    })

df_agrupado = pd.DataFrame(agrupado).sort_values(by='Frequência Total', ascending=False)
df_agrupado.to_csv("resumo_agrupado_clusters_por_tema.csv", index=False)
print(df_agrupado)
# ===============================
# 8️⃣ Mapear IDs de ementas para cluster e tema predominante
# ===============================
# Criar mapeamento cluster -> tema predominante
cluster_to_tema = dict(zip(df_cluster_resumo['Cluster'], df_cluster_resumo['Tema Predominante']))
df['Tema_Cluster'] = df['cluster'].map(cluster_to_tema)

# Salvar CSV com IDs, cluster e tema
df_ids = df[['ID', 'cluster', 'Tema_Cluster', 'ementa detalhada']]
df_ids.to_csv("ementas_por_cluster_e_tema.csv", index=False)

# ===============================
# 9️⃣ Salvar dataset completo
# ===============================
df.to_csv("resultados_clusters_semanticos.csv", index=False)
