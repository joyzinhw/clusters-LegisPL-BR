import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_excel('projetos_PL_2022_2024_limpo.xlsx', engine='openpyxl')
print(f"Total de registros: {len(dataset)}")

def build_tfidf_pipeline(use_char_ngrams=True, word_ngram_range=(1, 2), char_ngram_range=(3, 5), min_df=0.005, max_df=0.85, max_features=50000, svd_components=200):
    word_tfidf = TfidfVectorizer(
        ngram_range=word_ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    if use_char_ngrams:
        char_tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=char_ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features // 2,
        )
        vectorizer = FeatureUnion([('word_tfidf', word_tfidf), ('char_tfidf', char_tfidf)])
    else:
        vectorizer = word_tfidf
    pipeline = Pipeline([('vectorizer', vectorizer), ('svd', TruncatedSVD(n_components=svd_components, random_state=42))])
    return pipeline

def cluster_texts(texts, n_clusters=10, **kwargs):
    pipeline = build_tfidf_pipeline(**kwargs)
    X = pipeline.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    print(f"Silhouette Score: {score:.3f}")
    return clusters, X

n_clusters = 10
texts = dataset['ementa_limpa'].tolist()
clusters, X = cluster_texts(
    texts,
    n_clusters=n_clusters,
    use_char_ngrams=True,
    word_ngram_range=(1, 2),
    char_ngram_range=(3, 5),
    min_df=0.005,
    max_df=0.85,
    max_features=50000,
    svd_components=200,
)

dataset['cluster'] = clusters

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')
plt.title('Visualização 2D dos Clusters (PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

output_file = 'projetos_PL_2022_2024_clusterizado.xlsx'
dataset.to_excel(output_file, index=False, engine='openpyxl')
print(f"Dataset clusterizado salvo em: {output_file}")
