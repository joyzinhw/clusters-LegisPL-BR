
# python -m spacy download pt_core_news_lg

"""
similaridade_textual.py
Versão aprimorada com integração completa do spaCy em português do Brasil.
Suporta múltiplas representações e cálculos de similaridade textual:
- TF-IDF
- Cosine of Word Embeddings (GloVe-PT)
- Word Mover’s Distance (WMD)
- Sentence Transformers (SBERT)
"""
import joblib
import scipy.sparse as sps

import os
import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Caminhos de saída
TFIDF_PATH = "outputs/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = "outputs/tfidf_matrix.npz"
SBERT_EMB_PATH = "outputs/sbert_embeddings.npy"
SIM_MATRIX_TFIDF = "outputs/similarity_tfidf.npy"
SIM_MATRIX_SBERT = "outputs/similarity_sbert.npy"
SIM_MATRIX_GLOVE = "outputs/similarity_glove.npy"
SIM_MATRIX_WMD = "outputs/similarity_wmd.npy"

# =============================================================================
# 🧩 Funções com spaCy em pt-BR
# =============================================================================

def carregar_spacy_pt():
    """Carrega o modelo spaCy pt_core_news_lg (com fallback automático)."""
    import spacy
    try:
        nlp = spacy.load("pt_core_news_lg")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
        nlp = spacy.load("pt_core_news_lg")
    return nlp

def get_stopwords_pt():
    """Obtém stopwords em português (Brasil) do spaCy."""
    nlp = carregar_spacy_pt()
    return list(nlp.Defaults.stop_words)

def tokenizar_texto(texto: str, nlp=None) -> List[str]:
    """Tokeniza e lematiza texto com spaCy pt-BR, removendo stopwords e pontuação."""
    if nlp is None:
        nlp = carregar_spacy_pt()
    doc = nlp(texto.lower())
    return [tok.lemma_ for tok in doc if tok.is_alpha and not tok.is_stop]

# =============================================================================
# 🔹 TF-IDF
# =============================================================================

def computar_tfidf(corpus, max_features: int = 20000, ngram_range=(1,2)):
    """Calcula matriz TF-IDF usando stopwords do spaCy."""
    print("[i] Treinando TF-IDF...")
    stop_words_pt = get_stopwords_pt()
    vect = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words_pt
    )
    X = vect.fit_transform(corpus)
    print(f"[i] TF-IDF shape: {X.shape}")
    return vect, X

# =============================================================================
# 🔹 SBERT
# =============================================================================

def carregar_sbert_model(model_name: str = "modelos/sbert_paraphrase_multilingual"):
    from sentence_transformers import SentenceTransformer
    print(f"[i] Carregando SBERT: {model_name}")
    return SentenceTransformer(model_name)

def computar_sbert_embeddings(corpus, model_name: str = "modelos/sbert_paraphrase_multilingual", batch_size: int = 64):
    model = carregar_sbert_model(model_name)
    print("[i] Computando embeddings SBERT...")
    embs = model.encode(corpus, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    print(f"[i] SBERT embeddings shape: {embs.shape}")
    return embs

# =============================================================================
# 🔹 COS (GloVe)
# =============================================================================

def carregar_glove_pt(path: str = "modelos/glove_s300.txt"):
    """Carrega embeddings GloVe pré-treinados em português (Hartmann et al., 2017)."""
    print(f"[i] Carregando GloVe-PT: {path}")
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings[word] = coefs
    print(f"[i] GloVe-PT carregado com {len(embeddings)} palavras.")
    return embeddings

def media_embeddings(texto, embeddings, nlp=None, dim=300):
    """Calcula a média dos vetores de palavras usando spaCy pt-BR."""
    tokens = tokenizar_texto(texto, nlp)
    vetores = [embeddings[w] for w in tokens if w in embeddings]
    return np.mean(vetores, axis=0) if vetores else np.zeros(dim)

def computar_glove_embeddings(corpus, embeddings):
    """Cria matriz de embeddings médios (Cosine of Word Embeddings)."""
    print("[i] Calculando embeddings médios (GloVe)...")
    nlp = carregar_spacy_pt()
    dim = len(next(iter(embeddings.values())))
    return np.vstack([media_embeddings(txt, embeddings, nlp, dim) for txt in corpus])

# =============================================================================
# 🔹 WMD (Word Mover’s Distance)
# =============================================================================

def computar_wmd_matriz(corpus, model):
    """Calcula matriz de distâncias WMD entre textos (tokenizados com spaCy pt-BR)."""
    from tqdm import tqdm
    nlp = carregar_spacy_pt()
    tokenizados = [tokenizar_texto(txt, nlp) for txt in corpus]
    n = len(tokenizados)
    matriz = np.zeros((n, n))
    for i in tqdm(range(n), desc="Calculando WMD"):
        for j in range(i + 1, n):
            dist = model.wmdistance(tokenizados[i], tokenizados[j])
            matriz[i, j] = matriz[j, i] = dist
    return matriz

# =============================================================================
# 🔹 Similaridade Genérica
# =============================================================================

def sim_cosine_matrix_dense(X):
    print("[i] Calculando similaridade (cosine) — denso ...")
    return cosine_similarity(X)

def sim_cosine_matrix_sparse(X):
    print("[i] Calculando similaridade TF-IDF (sparse -> denso parcial)...")
    return cosine_similarity(X, dense_output=True)

# =============================================================================
# 🚀 Execução direta
# =============================================================================

if __name__ == "__main__":
    import joblib
    import scipy.sparse as sps

    df = pd.read_csv("outputs/projetos_PL_2022_2024_processado.csv")
    corpus = df["ementa"].astype(str).tolist()

    # === TF-IDF ===
    vect, X_tfidf = computar_tfidf(corpus)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(vect, TFIDF_PATH)
    sps.save_npz(TFIDF_MATRIX_PATH, X_tfidf)
    np.save(SIM_MATRIX_TFIDF, sim_cosine_matrix_sparse(X_tfidf))

    # === SBERT ===
    embs_sbert = computar_sbert_embeddings(corpus)
    np.save(SBERT_EMB_PATH, embs_sbert)
    np.save(SIM_MATRIX_SBERT, sim_cosine_matrix_dense(embs_sbert))

    # === COS (GloVe-PT) ===
    try:
        glove = carregar_glove_pt("modelos/glove_s300.txt")
        embs_glove = computar_glove_embeddings(corpus, glove)
        np.save(SIM_MATRIX_GLOVE, sim_cosine_matrix_dense(embs_glove))
    except Exception as e:
        print("[warn] GloVe não gerado:", e)

    # === WMD (subamostragem para teste) ===
    try:
        from gensim.models import KeyedVectors
        model_w2v = KeyedVectors.load_word2vec_format("modelos/glove_s300.txt")
        matriz_wmd = computar_wmd_matriz(corpus[:200], model_w2v)  # amostra
        np.save(SIM_MATRIX_WMD, matriz_wmd)
    except Exception as e:
        print("[warn] WMD não gerado:", e)

    print("[✅] Similaridades salvas em 'outputs/'")



def salvar_objeto(obj, path):
    """Salva um objeto Python de forma genérica (joblib ou np.save)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, (np.ndarray, np.matrix)):
        np.save(path, obj)
    elif hasattr(obj, "todense") or hasattr(obj, "toarray"):
        sps.save_npz(path, obj)
    else:
        joblib.dump(obj, path)