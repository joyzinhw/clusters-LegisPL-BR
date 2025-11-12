import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def gerar_embeddings(model_name, sub_ementas):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sub_ementas, convert_to_tensor=True, show_progress_bar=True)
    return model, embeddings

def calcular_similaridade_media(df, model, embeddings_sub_ementas):
    print("\n⚙️ Calculando similaridades médias ponderadas por tamanho das sub-ementas...")
    similarities_avg = []
    start_idx = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sub_ementas = row['sub_ementas']
        tema_tokens = row['tokens_temas']

        if len(sub_ementas) == 0 or len(tema_tokens) == 0:
            similarities_avg.append(0.0)
            continue

        end_idx = start_idx + len(sub_ementas)
        ementa_emb = embeddings_sub_ementas[start_idx:end_idx]
        start_idx = end_idx

        tema_emb = model.encode(tema_tokens, convert_to_tensor=True)
        sim_matrix = util.cos_sim(tema_emb, ementa_emb)

        if sim_matrix.numel() == 0:
            avg_sim = 0.0
        else:
            max_sim_per_sub_ementa = sim_matrix.max(dim=0).values
            lengths = np.array([len(s) for s in sub_ementas])
            weights = lengths / lengths.sum()
            avg_sim = float(np.sum(max_sim_per_sub_ementa.cpu().numpy() * weights))

        similarities_avg.append(avg_sim)

    df['similaridade'] = similarities_avg
    return df
