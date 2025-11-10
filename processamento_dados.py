import re
import os
import pandas as pd
from typing import Tuple

DEFAULT_INPUT = "legis/LegisPL-BR-main/dados/projetos_PL_2021_2024_transformado.xlsx"
DEFAULT_OUTPUT_CSV = "outputs/projetos_PL_2022_2024_processado.csv"
DEFAULT_OUTPUT_PKL = "outputs/projetos_PL_2022_2024_processado.pkl"

# mapeamento exemplo de partido -> bloco (adicione ou corrija conforme seu dicionário)
PARTIDO_TO_BLOCO = {
    # Esquerda
    "PT": "Esquerda", "PSOL": "Esquerda", "PCdoB": "Esquerda",
    # Direita
    "PL": "Direita", "PSC": "Direita", "PRTB": "Direita",
    # Centrao
    "MDB": "Centrao", "PP": "Centrao", "PSD": "Centrao", "PSB": "Centrao",
    # Outros (fallback)
}

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    # normalizações simples
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9áàâãéèêíïóôõöúçñ \-]", " ", s)  # mantém acentos latinos
    s = s.strip()
    return s

def extract_partido_main(autores_str: str) -> str:
    """
    Espera formatos como: "Fulano (PL-SP), Beltrano (PT-RJ)".
    Extrai a primeira sigla de partido (entre parênteses) ou retorna ''.
    """
    if not isinstance(autores_str, str):
        return ""
    m = re.search(r"\(([^)]+)\)", autores_str)
    if not m:
        return ""
    sig = m.group(1).split("-")[0].strip().upper()
    return sig

def map_partido_para_bloco(sigla: str) -> str:
    return PARTIDO_TO_BLOCO.get(sigla, "Outros" if sigla else "")

def carregar_transformar(path: str = DEFAULT_INPUT) -> pd.DataFrame:
    print(f"[i] Lendo arquivo: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    # garanta colunas chaves existam
    expected_cols = ["id", "ano", "ementa", "autores", "data da apresentacao", "data da ultima tramitacao"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[warn] Colunas esperadas faltando: {missing} (verifique o arquivo)")
    # limpeza e padronização
    df["ementa_raw"] = df.get("ementa", "").astype(str)
    df["ementa"] = df["ementa_raw"].apply(clean_text)
    df["autores_raw"] = df.get("autores", "").astype(str)
    df["partido_principal"] = df["autores_raw"].apply(extract_partido_main)
    df["bloco_partidario"] = df["partido_principal"].apply(map_partido_para_bloco)
    # flags
    df["autoria_coletiva"] = df["autores_raw"].str.contains(",")
    # datas
    for dt_col in ["data da apresentacao", "data da ultima tramitacao"]:
        if dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    # dias de tramitacao
    if "data da apresentacao" in df.columns and "data da ultima tramitacao" in df.columns:
        df["dias_tramitacao"] = (df["data da ultima tramitacao"] - df["data da apresentacao"]).dt.days.fillna(0).astype(int)
    # remover ementas vazias (opcional)
    df = df[df["ementa"].str.strip() != ""].reset_index(drop=True)
    return df

def salvar(df: pd.DataFrame, csv_path: str = DEFAULT_OUTPUT_CSV, pkl_path: str = DEFAULT_OUTPUT_PKL):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    print(f"[i] Salvo CSV em: {csv_path}")
    print(f"[i] Salvo PKL em: {pkl_path}")

if __name__ == "__main__":
    df = carregar_transformar()
    salvar(df)