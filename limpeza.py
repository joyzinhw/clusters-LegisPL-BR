import pandas as pd
import re

def tokenizar_virgula(texto):
    partes = re.split(r',|;', texto)
    return [p.strip() for p in partes if len(p.strip()) > 0]

def carregar_dados(caminho_excel):
    print("📘 Lendo o arquivo Excel...")
    df = pd.read_excel(caminho_excel, sheet_name="Sheet1")
    df = df[['ementa detalhada', 'temas']].dropna(subset=['ementa detalhada', 'temas'])
    
    df['sub_ementas'] = df['ementa detalhada'].apply(tokenizar_virgula)
    df['tokens_temas'] = df['temas'].apply(tokenizar_virgula)
    df['tema_principal'] = df['tokens_temas'].apply(lambda x: x[0] if len(x) > 0 else "Desconhecido")
    
    print("\n🧩 Exemplo de tokenização por vírgula:")
    print(df[['ementa detalhada', 'sub_ementas']].head())
    
    return df
