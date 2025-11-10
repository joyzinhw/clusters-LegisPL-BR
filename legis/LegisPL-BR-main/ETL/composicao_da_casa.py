import pandas as pd
from datetime import datetime, date
import pickle
import ast


def criar_datas_de_referencia(anos):
    """
    Cria uma lista de datas de referência (dia 15 de cada mês) para os anos fornecidos.
    """
    datas = []
    hoje = date.today()
    for ano in anos:
        for mes in range(1, 13):
            data_referencia = date(ano, mes, 15)
            if data_referencia <= hoje:
                datas.append(data_referencia)
    return datas


def buscar_deputados_ativos_por_data(df_entradas, data_alvo):
    df_entradas['data_entrada'] = pd.to_datetime(df_entradas['data_entrada']).dt.date
    df_entradas['data_saida'] = pd.to_datetime(df_entradas['data_saida']).dt.date
    #data_alvo_date = datetime.strptime(data_alvo, "%Y-%m-%d").date()
    try:
        # Converte a coluna de string para objeto de data


        # Converte a data alvo de string para objeto de data


        # Filtra os deputados cujo mandato está ativo na data alvo
        deputados_ativos = df_entradas[
            (df_entradas['data_entrada'] <= data_alvo) &
            (df_entradas['data_saida'] >= data_alvo)
            ]

        return deputados_ativos['id']
    except ValueError:
        print("Erro: O formato da data_alvo deve ser 'YYYY-MM-DD'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return pd.DataFrame()

def find_party_on_date(df, deputy_ids, specific_date):
    try:
        df['data_entrada_partido'] = pd.to_datetime(df['data_entrada_partido'])
        df['data_saida_partido'] = pd.to_datetime(df['data_saida_partido'])
        specific_date = pd.to_datetime(specific_date)
    except FileNotFoundError:
        return "Arquivo 'filiacoes_deputados.xlsx - Sheet1.csv' não encontrado."
    except Exception as e:
        return f"Ocorreu um erro: {e}"

    results = []
    for dep_id in deputy_ids:
        partido_encontrado = "Não encontrado"
        # Filtra as filiações do deputado
        deputy_affiliations = df[df['id'] == dep_id]

        # Itera sobre as filiações para encontrar a que corresponde à data
        for index, row in deputy_affiliations.iterrows():
            if row['data_entrada_partido'] <= specific_date <= row['data_saida_partido']:
                partido_encontrado = row['sigla_partido']
                break

        results.append({'id': dep_id, 'partido': partido_encontrado})

    return results

def analisar_e_contar_blocos(caminho_arquivo):
    # Dicionário para mapear cada partido ao seu bloco ideológico
    partido_para_bloco = {
        "PT": "Esquerda", "PSOL": "Esquerda", "PCdoB": "Esquerda", "REDE": "Esquerda",
        "PSB": "Esquerda", "PDT": "Esquerda", "PV": "Esquerda",
        "MDB": "Centrão", "PSD": "Centrão", "PP": "Centrão", "PL": "Centrão",
        "UNIÃO": "Centrão", "AVANTE": "Centrão", "CIDADANIA": "Centrão",
        "PODE": "Centrão", "SOLIDARIEDADE": "Centrão", "PATRIOTA": "Centrão",
        "PROS": "Centrão", "PRD": "Centrão",
        "REPUBLICANOS": "Direita", "NOVO": "Direita", "PSC": "Direita", "DEM": "Direita",
        "PSL": "Direita", "PRTB": "Direita", "PTB": "Direita", "PSDB": "Direita",
    }

    # Tenta ler o arquivo e retorna None em caso de erro
    try:
        df = pd.read_excel(caminho_arquivo)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{caminho_arquivo}'")
        return None

    # Listas para armazenar as contagens de cada linha
    contagem_esquerda = []
    contagem_direita = []
    contagem_centrao = []

    # Itera sobre cada linha do DataFrame
    for index, row in df.iterrows():
        # Inicializa os contadores para a linha atual
        qtd_esquerda = 0
        qtd_direita = 0
        qtd_centrao = 0
        # A coluna 'partidos' é uma string; ast.literal_eval a converte para uma lista de dicionários
        try:
            lista_partidos = ast.literal_eval(row['partidos'])
        except (ValueError, SyntaxError):
            # Se a célula for inválida ou vazia, trata como uma lista vazia
            lista_partidos = []
        # Itera sobre a lista de partidos da linha
        for deputado in lista_partidos:
            partido = deputado.get('partido')
            # Remove caracteres especiais como '*' para garantir a correspondência
            if isinstance(partido, str):
                partido = partido.strip().replace('*', '')

            # Busca o bloco do partido e incrementa o contador correspondente
            bloco = partido_para_bloco.get(partido)
            if bloco == "Esquerda":
                qtd_esquerda += 1
            elif bloco == "Direita":
                qtd_direita += 1
            elif bloco == "Centrão":
                qtd_centrao += 1

        # Adiciona a contagem final às listas
        contagem_esquerda.append(qtd_esquerda)
        contagem_direita.append(qtd_direita)
        contagem_centrao.append(qtd_centrao)

    # Adiciona as novas colunas ao DataFrame
    df['qtd_esquerda'] = contagem_esquerda
    df['qtd_direita'] = contagem_direita
    df['qtd_centrao'] = contagem_centrao

    return df

if __name__ == "__main__":
    '''
    df_entradas = pd.read_excel('entradas_saidas_deputados.xlsx')
    df_partidos = pd.read_excel('filiacoes_deputados.xlsx')

    df_datas = pd.DataFrame(columns=['data', 'partidos'])
    for data in criar_datas_de_referencia([2021,2022,2023,2024,2025]):
        deputados_ids = buscar_deputados_ativos_por_data(df_entradas, data)
        partidos_por_data = find_party_on_date(df_partidos, deputados_ids, data)
        df_datas = pd.concat([df_datas, pd.DataFrame([{'data': data, 'partidos': partidos_por_data}])], ignore_index=True)

    df_datas.to_excel('datas.xlsx', index=False)
    '''
    caminho_do_arquivo = 'datas.xlsx'
    df_analisado = analisar_e_contar_blocos(caminho_do_arquivo)

    # Exibe o DataFrame resultante com as novas colunas
    if df_analisado is not None:
        print(df_analisado.head())

    df_analisado.to_excel('analisados.xlsx', index=False)
