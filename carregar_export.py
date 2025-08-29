import os
import pandas as pd
import numpy as np

colunas_iniciais = ["Caminho de ponto", "DTS", "Unidade", "Detecção", "Canal", 
                    "Amostras", "Tempo máx.", "Velocidade (Hz)", "Valor do processo", 
                    "Unidade", "Dado"]

def salvar_npz(caminho_ponto, dados_df, registro_path="Banco de dados/SPYAI_arvore_vibration.csv"):
    caminho_ponto = caminho_ponto.replace("\\", '|').strip().replace("Hierarchy", "P100").replace(" ", "")

    if caminho_ponto.startswith('|'):
        caminho_ponto = caminho_ponto[1:]

    partes = [parte.strip() for parte in caminho_ponto.split('|')]

    ultima_parte = partes[-1]

    # Só processa se última parte começar com AV e terminar com V
    if not (ultima_parte.startswith("AV") and ultima_parte.endswith("A")):
        print(f"Ignorado: {caminho_ponto} (não é AV terminado com V)")
        return

    # Detecta o número após AV
    numero_AV = ''.join(filter(str.isdigit, ultima_parte))
    if not numero_AV:
        print(f"Ignorado: {caminho_ponto} (sem número após AV)")
        return

    # Nome da pasta de nível (ex.: AV01, AV02, AV03...)
    pasta_nivel = f"AV{numero_AV}"

    # Pasta base até antes da última parte
    pasta_base = os.path.join("Banco de dados", *partes[:-1])

    # Caminho até a pasta nível (ex.: ...\AV01)
    pasta_AVx = os.path.join(pasta_base, pasta_nivel)
    os.makedirs(pasta_AVx, exist_ok=True)

    # Subpasta final (ex.: AV 01HV, AV 02VV, AV 03AV)
    pasta_destino = os.path.join(pasta_AVx, ultima_parte)
    os.makedirs(pasta_destino, exist_ok=True)

    # Nome do arquivo .npz
    nome_arquivo = f"Export_Time_{partes[-2]}.npz"
    caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)

    dados_df.columns = colunas_iniciais + list(dados_df.columns[len(colunas_iniciais):])

    np.savez(caminho_arquivo, data=dados_df.values, columns=dados_df.columns.values)
    print(f'Salvo: {caminho_arquivo}')

    # Caminho a ser salvo no CSV (até AVxx)
    caminho_csv = os.path.join("P100", *partes[1:-1], pasta_nivel)

    # Atualiza o CSV (evitando duplicatas)
    if os.path.exists(registro_path):
        registros_existentes = pd.read_csv(registro_path)
        if (registros_existentes['Caminho ponto'] == caminho_csv).any():
            return

    df_registro = pd.DataFrame({'Caminho ponto': [caminho_csv]})
    if not os.path.exists(registro_path):
        df_registro.to_csv(registro_path, mode='w', index=False, header=True)
    else:
        df_registro.to_csv(registro_path, mode='a', index=False, header=False)

def processar_csv(csv_path1,csv_path2):
    df1 = pd.read_csv(csv_path1, encoding="ISO-8859-1", decimal=",", delimiter=";", names=range(16394), skiprows=1)
    df2 = pd.read_csv(csv_path2, encoding="ISO-8859-1", decimal=",", delimiter=";", names=range(16394), skiprows=1)

    datas_df1 = set(df1.iloc[:,1].dropna().astype(str).unique())
    datas_df2 = set(df2.iloc[:,1].dropna().astype(str).unique())

    novas_datas = datas_df2 - datas_df1
    if novas_datas:
        print(f"Datas novas em {csv_path2}: {novas_datas}")
        df_novos = df2[df2.iloc[:,1].astype(str).isin(novas_datas)]
        df1 = pd.concat([df1, df_novos], ignore_index=True)

    grupos = df1.groupby(df1.columns[0])
    for caminho_ponto, grupo in grupos:
        salvar_npz(str(caminho_ponto), grupo)

csv_path1 = "/home/mvduarte/SPYAI_16_08_25/Export_Time_EGF.csv"

csv_path2 = "/home/mvduarte/SPYAI_16_08_25/Export_Time_EGF_6.csv"
# csv_path2 = "C:\\Users\\letic\\OneDrive - Universidade Federal de Uberlândia\\Documentos\\SPYAI\\Arcelormittal\\EGF-6 meses\\Export_Time.csv"

processar_csv(csv_path1,csv_path2)
