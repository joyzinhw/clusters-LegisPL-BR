import pandas as pd
import os

# Caminho da pasta com os arquivos
path = '/home/joyzinhw/propor/legis/LegisPL-BR-main/dados'
saida = '/home/joyzinhw/propor/colunas_arquivos.txt'

# Abre o arquivo de saída
with open(saida, 'w', encoding='utf-8') as out:
    out.write("📊 RELATÓRIO DE COLUNAS DOS ARQUIVOS EXCEL\n")
    out.write("=====================================================\n")

    # Percorre todos os arquivos .xlsx
    for file in os.listdir(path):
        if file.endswith('.xlsx'):
            file_path = os.path.join(path, file)
            out.write(f"\n📘 Arquivo: {file}\n")
            out.write("-----------------------------------------------------\n")
            try:
                # Lê as abas (sheets)
                xls = pd.ExcelFile(file_path)
                for sheet in xls.sheet_names:
                    out.write(f"  🧩 Aba: {sheet}\n")
                    df = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
                    for col in df.columns:
                        out.write(f"     - {col}\n")
                out.write("\n")
            except Exception as e:
                out.write(f"⚠️ Erro ao ler {file}: {e}\n")

print(f"✅ Relatório gerado com sucesso em:\n{saida}")
