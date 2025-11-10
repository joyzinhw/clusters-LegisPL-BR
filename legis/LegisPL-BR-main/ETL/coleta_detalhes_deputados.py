import requests
import pandas as pd
import time
import json
from datetime import datetime, date
import pickle


BASE_URL = "https://dadosabertos.camara.leg.br/api/v2"


def criar_datas_de_referencia(anos):
    """
    Cria uma lista de datas de referência (dia 15 de cada mês) para os anos fornecidos.

    Args:
        anos (list): Uma lista de anos para gerar as datas.

    Returns:
        list: Uma lista de objetos datetime.date.
    """
    datas = []
    hoje = date.today()
    for ano in anos:
        for mes in range(1, 13):
            data_referencia = date(ano, mes, 15)
            if data_referencia <= hoje:
                datas.append(data_referencia)
    return datas


def coletar_dados_historicos(nome_arquivo="historico_deputados_bruto.pkl"):
    """
    Busca o histórico de todos os deputados e salva em um arquivo pickle.

    Args:
        nome_arquivo (str): O nome do arquivo pickle para salvar os dados.
    """
    historico_deputados_lista = []

    try:
        print("Buscando lista completa de deputados...")
        deputados_ids_url = f"{BASE_URL}/deputados?itens=1000&ordem=ASC&ordenarPor=nome"
        resp_ids = requests.get(deputados_ids_url, headers={"accept": "application/json"})
        resp_ids.raise_for_status()
        deputados_ids = resp_ids.json()["dados"]

        print("Buscando histórico para cada deputado...")
        for i, deputado in enumerate(deputados_ids):
            deputado_id = deputado["id"]
            historico_url = f"{BASE_URL}/deputados/{deputado_id}/historico"

            time.sleep(0.1)

            try:
                resp = requests.get(historico_url, headers={"accept": "application/json"})
                resp.raise_for_status()

                historico = resp.json()

                historico_deputados_lista.append({
                    "id": deputado_id,
                    "historico": historico
                })
                print(f"[{i + 1}/{len(deputados_ids)}] Histórico do deputado {deputado_id} obtido.")

            except requests.exceptions.RequestException as e:
                print(f"[{i + 1}/{len(deputados_ids)}] Erro ao buscar histórico do deputado {deputado_id}: {e}")
                historico_deputados_lista.append({
                    "id": deputado_id,
                    "historico": []
                })
            except Exception as e:
                print(f"[{i + 1}/{len(deputados_ids)}] Erro inesperado ao processar deputado {deputado_id}: {e}")
                historico_deputados_lista.append({
                    "id": deputado_id,
                    "historico": []
                })

        if historico_deputados_lista:
            with open(nome_arquivo, 'wb') as f:
                pickle.dump(historico_deputados_lista, f)
            print(f"\nDados de todos os deputados salvos com sucesso em '{nome_arquivo}'")
        else:
            print("\nNenhum dado de deputado foi coletado. O arquivo pickle não foi criado.")

    except requests.exceptions.RequestException as e:
        print(f"Erro de requisição inicial: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


def processar_historico_para_entradas_saidas(nome_arquivo_bruto, nome_arquivo_saida="entradas_saidas_deputados.xlsx"):
    """
    Lê o histórico de deputados, extrai todas as datas de entrada e saída
    e salva em uma nova planilha Excel.
    """
    try:
        with open(nome_arquivo_bruto, 'rb') as f:
            historico_deputados_lista = pickle.load(f)

        print(f"Dados brutos lidos com sucesso do arquivo '{nome_arquivo_bruto}'")

        df_bruto = pd.DataFrame(historico_deputados_lista)
        df_final_lista = []

        print("Processando histórico para extrair entradas e saídas...")

        for _, row in df_bruto.iterrows():
            deputado_id = row['id']
            historico = row['historico'].get('dados', [])
            print(historico)
            if not isinstance(historico, list) or not historico:
                continue
            print("passei")
            # Ordenar o histórico por data para processar cronologicamente
            historico_ordenado = sorted(historico, key=lambda x: datetime.strptime(x['dataHora'], "%Y-%m-%dT%H:%M"))
            print(historico_ordenado)
            periodos_mandato = []
            entrada_atual = None

            for evento in historico_ordenado:
                if not isinstance(evento, dict) or 'dataHora' not in evento or 'situacao' not in evento:
                    continue

                data_evento_str = evento['dataHora']
                situacao = evento['situacao']

                # Identifica um evento de entrada (Posse)
                if 'Posse' in evento.get('descricaoStatus', '') and entrada_atual is None:
                    entrada_atual = datetime.strptime(data_evento_str, "%Y-%m-%dT%H:%M").date()

                # Identifica um evento de saída (Fim de Mandato ou similar)
                elif situacao == 'Fim de Mandato' and entrada_atual is not None:
                    data_saida = datetime.strptime(data_evento_str, "%Y-%m-%dT%H:%M").date()
                    periodos_mandato.append((entrada_atual, data_saida))
                    entrada_atual = None

            # Se o mandato ainda estiver ativo, a data de saída é hoje
            if entrada_atual is not None:
                periodos_mandato.append((entrada_atual, datetime.now().date()))

            # Adiciona os dados ao DataFrame final
            for entrada, saida in periodos_mandato:
                # O nome do deputado é o mesmo para todos os eventos no histórico
                nome_deputado = historico_ordenado[0].get('nome')
                df_final_lista.append({
                    'id': deputado_id,
                    'nome_deputado': nome_deputado,
                    'data_entrada': entrada.strftime("%Y-%m-%d"),
                    'data_saida': saida.strftime("%Y-%m-%d")
                })

        if df_final_lista:
            df_final = pd.DataFrame(df_final_lista)
            df_final.to_excel(nome_arquivo_saida, index=False)
            print(f"\nAnálise concluída e salva em '{nome_arquivo_saida}'.")
        else:
            print("\nNenhum dado de entrada/saída foi extraído. O arquivo Excel não foi criado.")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo_bruto}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


def processar_historico_para_filiacoes(nome_arquivo_bruto, nome_arquivo_saida="filiacoes_deputados.xlsx"):
    """
    Lê o histórico de deputados, extrai as datas de filiação e desfiliação de cada partido
    e salva em uma nova planilha Excel.
    """
    try:
        with open(nome_arquivo_bruto, 'rb') as f:
            historico_deputados_lista = pickle.load(f)

        print(f"Dados brutos lidos com sucesso do arquivo '{nome_arquivo_bruto}'")

        df_bruto = pd.DataFrame(historico_deputados_lista)
        filiacoes_lista = []

        print("Processando histórico para extrair filiações e desfiliações...")

        for _, row in df_bruto.iterrows():
            deputado_id = row['id']
            historico = row['historico'].get('dados', [])

            if not isinstance(historico, list) or not historico:
                continue

            historico_ordenado = sorted(historico, key=lambda x: datetime.strptime(x['dataHora'], "%Y-%m-%dT%H:%M"))

            filiacao_atual = None
            data_entrada_partido = None
            nome_deputado = historico_ordenado[0].get('nome')

            for evento in historico_ordenado:
                if not isinstance(evento, dict) or 'dataHora' not in evento or 'siglaPartido' not in evento:
                    continue

                sigla_partido_evento = evento.get('siglaPartido')
                data_evento_str = evento['dataHora']

                if sigla_partido_evento != filiacao_atual:
                    if filiacao_atual is not None:
                        data_saida_partido = datetime.strptime(data_evento_str, "%Y-%m-%dT%H:%M").date()
                        filiacoes_lista.append({
                            'id': deputado_id,
                            'nome_deputado': nome_deputado,
                            'sigla_partido': filiacao_atual,
                            'data_entrada_partido': data_entrada_partido.strftime("%Y-%m-%d"),
                            'data_saida_partido': data_saida_partido.strftime("%Y-%m-%d")
                        })

                    filiacao_atual = sigla_partido_evento
                    data_entrada_partido = datetime.strptime(data_evento_str, "%Y-%m-%dT%H:%M").date()

            if filiacao_atual is not None:
                filiacoes_lista.append({
                    'id': deputado_id,
                    'nome_deputado': nome_deputado,
                    'sigla_partido': filiacao_atual,
                    'data_entrada_partido': data_entrada_partido.strftime("%Y-%m-%d"),
                    'data_saida_partido': datetime.now().date().strftime("%Y-%m-%d")
                })

        if filiacoes_lista:
            df_filiacoes = pd.DataFrame(filiacoes_lista)
            df_filiacoes.to_excel(nome_arquivo_saida, index=False)
            print(f"\nAnálise de filiações concluída e salva em '{nome_arquivo_saida}'.")
        else:
            print("\nNenhum dado de filiação foi extraído. O arquivo Excel não foi criado.")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{nome_arquivo_bruto}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


# --- Execução do Script ---
if __name__ == "__main__":
    # O código abaixo é apenas para fins de demonstração.
    # Você já executou a Etapa 1. Agora, basta executar a Etapa 2.
    #coletar_dados_historicos()
    # --- Etapa 2: Ler do pickle e processar os dados ---
    #processar_historico_para_entradas_saidas(nome_arquivo_bruto="historico_deputados_bruto.pkl")
    processar_historico_para_filiacoes(nome_arquivo_bruto="historico_deputados_bruto.pkl")
