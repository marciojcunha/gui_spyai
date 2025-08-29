import os
import numpy as np
import datetime
import pandas as pd
from tkinter import filedialog

# Função para salvar as informações de cada linha no formato npz
def salvar_dados(pasta, path):

    # Criação das subpastas, caso não existam
    subpastas = ['Polia/AV01/AV01VV', 'Polia/AV01/AV01HV', 'Disco/AV03/AV03VV', 'Disco/AV03/AV03HV']
    for subpasta in subpastas:
        caminho_subpasta = os.path.join(path, subpasta)
        if not os.path.exists(caminho_subpasta):
            os.makedirs(caminho_subpasta)
    data_inicio = datetime.datetime(2022, 1, 22, 10, 0, 0)
    mes = 1 
    ano = 2022
    for p in os.listdir(path):
        caminho_pasta = os.path.join(path,p)
        # Data inicial da medição
        
        if 'Test ' in caminho_pasta:
            files = os.listdir(caminho_pasta)
            for j in range(3): 
                posic = np.random.randint(0,len(files)) 
                arquivo = files[posic] 
                if arquivo.endswith(".npy"):
                    caminho_arquivo = os.path.join(caminho_pasta, arquivo)
                    try:
                        dados = np.load(caminho_arquivo, allow_pickle=True)
                        data_medicao = datetime.datetime(int(ano),int(mes), 22, 10, 0, 0) 
                        mes +=1 
                        if mes > 12 : 
                            mes = 1 
                            ano +=1                        
                        for i, linha in enumerate(dados):
                            # Gerar informações específicas da linha
 
                            unidade = "g"
                            detecao = -1
                            canal = i + 1  # Canal será o número da linha
                            amostras = len(linha)
                            tempo_max = 25000 / amostras
                            velocidade = 1238 / 60
                            valor_processo = unidade
                            dado = linha

                            # Preencher os dados
                            dados_salvar = {
                                "Caminho de ponto": f'GERINGONÇA/{pasta}',  # Lista com 1 elemento
                                "DTS": [data_medicao.strftime("%d/%m/%Y %H:%M:%S")],  # Lista com 1 elemento
                                "Unidade": [unidade],  # Lista com 1 elemento
                                "Detecção": [detecao],  # Lista com 1 elemento
                                "Canal": [canal],  # Lista com 1 elemento
                                "Amostras": [amostras],  # Lista com 1 elemento
                                "Tempo máx.": [tempo_max],  # Lista com 1 elemento
                                "Velocidade (Hz)": [velocidade],  # Lista com 1 elemento
                                "Valor do processo": [valor_processo],  # Lista com 1 elemento
                                "Unidade": [unidade],  # Lista com 1 elemento
                                "Dado": [dado[0]]  # Primeira amostra do dado como lista
                            }

                            # Adiciona as demais amostras do dado com índices dinâmicos
                            for index in range(1, len(dado)):
                                dados_salvar[f'Dado_{index}'] = dado[index]

                            # Salvar os dados nas pastas adequadas
                            if i == 0:
                                subpasta_destino = os.path.join(path, 'Polia/AV01/AV01VV')
                                nome_arquivo = 'Export_Time_Polia.npz'
                            elif i == 1:
                                subpasta_destino = os.path.join(path, 'Polia/AV01/AV01HV')
                                nome_arquivo = 'Export_Time_Polia.npz'
                            elif i == 2:
                                subpasta_destino = os.path.join(path, 'Disco/AV03/AV03VV')
                                nome_arquivo = 'Export_Time_Disco.npz'
                            elif i == 3:
                                subpasta_destino = os.path.join(path, 'Disco/AV03/AV03HV')
                                nome_arquivo = 'Export_Time_Disco.npz'
                            else:
                                continue

                            # Criar o caminho para salvar o arquivo npz
                            caminho_npz = os.path.join(subpasta_destino, nome_arquivo)
                            # Salvar os dados no arquivo npz
                            if os.path.exists(caminho_npz):
                                # Carregar os dados existentes
                                dados_existentes = np.load(caminho_npz, allow_pickle=True)
                                dados_existentes_df = pd.DataFrame(dados_existentes['data'])
                                dados_existentes_df.columns = dados_existentes['columns']
                                dados_salvar_df = pd.DataFrame(dados_salvar)
                                dados_existentes_df = pd.concat( [dados_existentes_df, dados_salvar_df] )                                 
                                # Salvar os dados combinados
                                np.savez(caminho_npz, data=np.array(dados_existentes_df.values,dtype=object), columns=dados_existentes_df.columns)
                            else:
                                dados_salvar_df = pd.DataFrame(dados_salvar)

                                # Caso não exista, criar o arquivo npz com os dados
                                np.savez(caminho_npz, data=np.array(dados_salvar_df.values,dtype=object), columns=dados_salvar_df.columns)

                    except:
                        print('Erro ao carregar arquivo.')
                print(j)

    print(f"Salvo arquivos da pasta: {path}")


dir = '/home/lav/Leticia/SPYAI/SPYAI_23_07_25/Banco de dados/Testes'
pastas = ['Desalinhamento', 'Desbalanceamento', 'Folga', 'Normal']
from shutil import rmtree
try : 
    for pasta in pastas:
        rmtree(f'/home/lav/Leticia/SPYAI/SPYAI_23_07_25/Banco de dados/Testes/{pasta}/Disco')
        rmtree(f'/home/lav/Leticia/SPYAI/SPYAI_23_07_25/Banco de dados/Testes/{pasta}/Polia')
except: 
    pass        
for pasta in pastas:
    path = os.path.join(dir,pasta)

    # Chama a função para salvar os dados
    salvar_dados(pasta, path)

