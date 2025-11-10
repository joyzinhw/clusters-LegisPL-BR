from sentence_transformers import SentenceTransformer

# modelo multilíngue compatível com português
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"[i] Baixando modelo {model_name}...")
model = SentenceTransformer(model_name)

# salvar localmente
save_path = "modelos/sbert_paraphrase_multilingual"
model.save(save_path)
print(f"[✅] Modelo salvo em: {save_path}")
