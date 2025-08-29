import numpy as np
import pandas as pd
import os
import tkinter.messagebox as mb
from dynambase import Quality_Signal
from dynamanalisador import Analisador
from dynamclassification import Classification
from dynamrul import rul
import dynamrul as dr
import dynamsignal as ds
from platform import system


class Ponto():
    def __init__(self, path):
        self.path = path
        subpastas = [p for p in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, p))]
        self.path_h = None
        self.path_v = None
        self.path_a = None
        for pasta in subpastas:
            if pasta[-2:-1].upper() == "H":
                self.path_h = os.path.join(self.path, pasta)
            if pasta[-2:-1].upper() == "V":
                self.path_v = os.path.join(self.path, pasta)
            if pasta[-2:-1].upper() == "A":
                self.path_a = os.path.join(self.path, pasta)

        self.dados_sinal_h = None
        self.dados_sinal_a = None
        self.dados_sinal_v = None

        if self.path_h != None:
            files = os.listdir(self.path_h)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found is None:
                raise FileNotFoundError("Arquivo Export_Time*.npz não encontrado")
            self.dados_sinal_h = np.load(os.path.join(self.path_h, found), allow_pickle=True)
        if self.path_v != None:
            files = os.listdir(self.path_v)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found is None:
                raise FileNotFoundError("Arquivo Export_Time*.npz não encontrado")
            self.dados_sinal_v = np.load(os.path.join(self.path_v, found), allow_pickle=True)
        if self.path_a != None:
            files = os.listdir(self.path_a)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found is None:
                raise FileNotFoundError("Arquivo Export_Time*.npz não encontrado")
            self.dados_sinal_a = np.load(os.path.join(self.path_a, found), allow_pickle=True)
        
        self.path_directions = [self.path_h, self.path_v, self.path_a]
        self.directions = [self.dados_sinal_h, self.dados_sinal_v, self.dados_sinal_a]

    def analise_ponto(self):
        for i, direction in enumerate(self.directions):
            if direction is not None:
                dados_numpy = direction
                path_direction = self.path_directions[i]

                # Verifica a unidade do sinal
                dados_numpy = Quality_Signal.unit_verification(dados_numpy)
                analise_file = os.path.join(path_direction, "SPYAI_analise.npz")
                if system() == 'Windows' : 
                    analise_file.replace('/','\\')
                else : 
                    analise_file.replace('\\','/')     
                # Calcula os sintomas pra geral
                if os.path.exists(analise_file):
                    dados_analise = np.load(analise_file, allow_pickle=True)
                    dados_analise_data = [item[0] for item in dados_analise['data'][:, np.where(dados_analise['columns'] == 'DATA')[0]].tolist()]

                    data_numpy = dados_numpy['data'][:, np.where(dados_numpy['columns'] == 'DTS')[0]].flatten()
                    datas_novas = [d for d in data_numpy if d not in dados_analise_data]

                    if datas_novas:
                        print(f"Chamando analisador para {len(datas_novas)} novos dados")
                        a = Analisador(path_direction, dados_numpy=dados_numpy)
                        a.analisar(data=datas_novas)
                    else:
                        print("Nenhum dado novo para analisar.")
                else:
                    print("Nenhum arquivo de análise encontrado. Processando tudo.")
                    a = Analisador(path_direction, dados_numpy=dados_numpy)
                    a.analisar(data='all')

    def classification_ponto(self):
        cadastro_file = os.path.join(os.path.dirname(self.path), "SPYAI_cadastro.csv")
        if system() == 'Windows' : 
            cadastro_file.replace('/','\\')
        else : 
            cadastro_file.replace('\\','/')     
        if not os.path.exists(cadastro_file):
            mb.showwarning("FALTA DE CADASTRO", "O equipamento não foi cadastrado, por favor faça o cadastro do EQUIPAMENTO.")
        else:
            dados_cadastro = pd.read_csv(cadastro_file, sep=r'[;,]', engine = 'python')

        for i, direction in enumerate(self.directions):
            if direction is not None:
                dados_numpy = direction
                path_direction = self.path_directions[i]
                if system() == 'Windows' : 
                    path_direction.replace('/','\\')
                else : 
                    path_direction.replace('\\','/')     
                classif_file = os.path.join(path_direction, "SPYAI_pred.npz")

                datas_np = dados_numpy['data'][:, np.where(dados_numpy['columns'] == 'DTS')[0][0]]

                # Verifica se já foi feito a classificação anteriormente
                if os.path.exists(classif_file):
                    dados_class = np.load(classif_file, allow_pickle=True)
                    aux = dados_class['data']
                    datas = [] 
                    for i in range(len(aux)) : 
                        datas.append(aux[i,2])

                    # Verifica se tem datas novas para ser classificadas e previstas
                    datas_np = [] 
                    aux =   dados_numpy['data']
                    for i in range(len(aux)) : 
                        datas_np.append(aux[i,1])
                    datas_novas = [d for d in datas_np if d not in datas]

                    if len(datas_novas) != 0:
                        for data in datas_novas:
                            print(f'Classificação sendo feita para {data}.')
                            sinal =  dados_numpy['data'][dados_numpy['data'][:, np.where(dados_numpy['columns'] == 'DTS')[0][0]] == data]

                            treinamento = False
                            idx_dados = int(np.where(dados_numpy['columns'] == 'Dado')[0])
                            path_sinal_base = path_direction + os.path.sep + 'features_sinal_base.npz'
                            Classification(path_sinal_base, path_direction, self.path_v, sinal, dados_cadastro, treinamento, data, idx_dados)

                    else:
                        print("Nenhuma data para classificar.")            
                else:
                #Faz a classificação de todos as datas se for a primeira vez que o ponto tá sendo carregado
                    print("Nenhum arquivo de classificação encontrado. Processando tudo.")
                    data_numpy = dados_numpy['data'][:, np.where(dados_numpy['columns'] == 'DTS')[0]].flatten()
                    data_numpy = np.array(sorted(data_numpy, key=lambda x: pd.to_datetime(x,dayfirst=True)))

                    # Classifica com as IA
                    # mb.showwarning("IA EM TREINAMENTO", "Por favor, aguarde enquanto a IA é treinada.")
                    init = 0 
                    end = len(data_numpy) 
                    step = 1
                    if  dr.utilities.date_to_numeric(data_numpy[0], type = 'horas') >  dr.utilities.date_to_numeric(data_numpy[1], type = 'horas'):
                        init = end - 1 
                        end = -1 
                        step = -1
                    if '06/01/2016 15:05:02' == data_numpy[0] : 
                        init +=1 
                    for i in range(init, end, step):
                        data = data_numpy[i]                                     
                        print(data)
                        sinal =  dados_numpy['data'][dados_numpy['data'][:, np.where(dados_numpy['columns'] == 'DTS')[0][0]] == data]

                        if data == data_numpy[init]:
                            treinamento = True
                        else:
                            treinamento = False
                        idx_dados = int(np.where(dados_numpy['columns'] == 'Dado')[0])
                        path_sinal_base = path_direction + os.path.sep + 'features_sinal_base.npz'
                        Classification(path_sinal_base, path_direction, self.path_v, sinal, dados_cadastro, treinamento, data, idx_dados)
        
    def rul_ponto(self):
        if self.path_h != None:
            self.dados_analise_h = np.load(os.path.join(self.path_h, "SPYAI_analise.npz"), allow_pickle=True)
        else:
            self.dados_analise_h = None
        if self.path_v != None:
            self.dados_analise_v = np.load(os.path.join(self.path_v, "SPYAI_analise.npz"), allow_pickle=True)
        else: 
            self.dados_analise_v = None
        if self.path_a != None:
            self.dados_analise_a = np.load(os.path.join(self.path_a, "SPYAI_analise.npz"), allow_pickle=True)
        else: 
            self.dados_analise_a = None
        
        self.dados_analise = [self.dados_analise_h, self.dados_analise_v, self.dados_analise_a]                 
        pred_file = os.path.join(self.path, "SPYAI_rul.npz")

        # Se o arquivo rul já existir
        if os.path.exists(pred_file):
            dados_pred = np.load(pred_file, allow_pickle=True)
            pred = dados_pred['data']
            datas_pred = [] 
            for i in range(1, len(pred)) : 
                datas_pred.append(pred[i,0])

            # Verifica se tem datas novas para serem previstas
            todas_datas_novas = []
            datas_novas_directions = [] 
            for i in range(len(self.directions)):
                if self.directions[i] != None:
                    aux = self.directions[i]['data'][:,1]
                    datas_novas_direction = []
                    timestamps = pd.to_datetime(aux, dayfirst=True)
                    grouped_times = timestamps.floor('h').strftime('%d/%m/%Y %H:00:00')
                    datas_novas_direction = sorted(list(set(grouped_times)), key=lambda x: pd.to_datetime(x, dayfirst=True), reverse=True)
                    datas_novas_hour = [d for d in datas_novas_direction if d not in datas_pred]
                    datas_novas_directions.append(datas_novas_hour)
                    todas_datas_novas.append(datas_novas_hour)
            
            if len(todas_datas_novas) == 0:
                for data_nova in todas_datas_novas:
                    fs = []
                    sinais = []
                    features = []
                    time_unit = 'meses'
                    time_lag = 6
                    rpm = 1200
                    for i in datas_novas_directions[i]:
                        if data_nova in  datas_novas_directions[i]:
                            try:
                                dados_analise = self.dados_analise[i]
                                dados_sinal = self.directions[i]
                                idx_data_analise = np.where(dados_analise['columns'] == 'DATA')[0][0]
                                idx_data_sinal = np.where(dados_sinal['columns'] == 'DTS')[0][0]
                                idx_features = np.where(dados_analise['columns'] == 'harm1')[0][0]
                                idx_amostras = np.where(dados_sinal['columns'] == 'Amostras')[0][0]
                                idx_dado = np.where(dados_sinal['columns'] == 'Dado')[0][0]
                                idx_fs = np.where(dados_analise['columns'] == 'Fs')[0][0]
                                idx_RPM = np.where(dados_analise['columns'] == 'RPM')[0][0]


                                datas_analise = pd.to_datetime(dados_analise['data'][:, idx_data_analise], dayfirst=True)
                                datas_sinal = pd.to_datetime(dados_sinal['data'][:, idx_data_sinal], dayfirst=True)

                                linhas_analise = np.where(datas_analise.floor('h').strftime('%d/%m/%Y %H:00:00') == data_hour)[0]
                                linhas_sinal = np.where(datas_sinal.floor('h').strftime('%d/%m/%Y %H:00:00') == data_hour)[0]

                                row_analise = dados_analise['data'][linhas_analise[-1]]
                                row_sinal = dados_sinal['data'][linhas_sinal[-1]]
                                fs_direction = int(row_analise[idx_fs])
                                amostras = int(row_sinal[idx_amostras])
                                if i == 0 :
                                    rpm = int(row_analise[idx_RPM])
                                sinal_direction = row_sinal[idx_dado: idx_dado + amostras].astype(float)
                                sinal_direction = sinal_direction[~np.isnan(sinal_direction)]
                                features_direction = row_analise[idx_features: idx_features + 7].astype(float)
                                fs.append(fs_direction)
                                sinais.append(sinal_direction)
                                features.append(features_direction)
                            except :
                                continue

                    # Calcula rul para as 3 direções
                    RUL = rul(self.path, time_unit, time_lag)
                    RUL.calcula_rul(data_hour,rpm,fs,sinais, features)
            else:
                print('Nenhuma data para prever.')

        # Se for a primeira vez que o rul tá rodando
        else:
            all_datas = []
            for dados in self.dados_analise:
                if dados is not None:
                    idx_data = np.where(dados['columns'] == 'DATA')[0][0]
                    all_datas.extend(list(dados['data'][:, idx_data]))

                timestamps = pd.to_datetime(all_datas, dayfirst=True)
                grouped_times = timestamps.floor('h').strftime('%d/%m/%Y %H:00:00')
                self.grouped_hours = sorted(list(set(grouped_times)), key=lambda x: pd.to_datetime(x, dayfirst=True), reverse=True)
                init = 0 
                end = len(self.grouped_hours) 
                step = 1
                if  dr.utilities.date_to_numeric(self.grouped_hours[0], type = 'horas') >  dr.utilities.date_to_numeric(self.grouped_hours[1], type = 'horas'):
                    init = end - 1 
                    end = -1 
                    step = -1
                if '06/01/2016 15:05:02' == self.grouped_hours[0] : 
                    init +=1 
            for k_ext in range(init, end, step):
            # Roda a previsão de todos os pontos
                data_hour = self.grouped_hours[k_ext]
                if data_hour == '28/05/2025 14:00:00' : 
                    para_aqui = 1
                fs = []
                sinais = []
                features = []
                time_unit = 'meses'
                time_lag = 6
                for i in range(len(self.dados_analise)):
                    if self.dados_analise[i] != None:
                        try:
                            dados_analise = self.dados_analise[i]
                            dados_sinal = self.directions[i]
                            idx_data_analise = np.where(dados_analise['columns'] == 'DATA')[0][0]
                            idx_data_sinal = np.where(dados_sinal['columns'] == 'DTS')[0][0]
                            idx_features = np.where(dados_analise['columns'] == 'harm1')[0][0]
                            idx_amostras = np.where(dados_sinal['columns'] == 'Amostras')[0][0]
                            idx_dado = np.where(dados_sinal['columns'] == 'Dado')[0][0]
                            idx_fs = np.where(dados_analise['columns'] == 'Fs')[0][0]
                            idx_RPM = np.where(dados_analise['columns'] == 'RPM')[0][0]


                            datas_analise = pd.to_datetime(dados_analise['data'][:, idx_data_analise], dayfirst=True)
                            datas_sinal = pd.to_datetime(dados_sinal['data'][:, idx_data_sinal], dayfirst=True)

                            linhas_analise = np.where(datas_analise.floor('h').strftime('%d/%m/%Y %H:00:00') == data_hour)[0]
                            linhas_sinal = np.where(datas_sinal.floor('h').strftime('%d/%m/%Y %H:00:00') == data_hour)[0]

                            row_analise = dados_analise['data'][linhas_analise[-1]]
                            row_sinal = dados_sinal['data'][linhas_sinal[-1]]
                            fs_direction = int(row_analise[idx_fs])
                            if i > 0  and fs_direction < fs[0] : 
                                continue
                            amostras = int(row_sinal[idx_amostras])
                            if i == 0 :
                                rpm = int(row_analise[idx_RPM])
                            sinal_direction = row_sinal[idx_dado: idx_dado + amostras].astype(float)
                            sinal_direction = sinal_direction[~np.isnan(sinal_direction)]
                            features_direction = row_analise[idx_features: idx_features + 7].astype(float)
                            fs.append(fs_direction)
                            sinais.append(sinal_direction)
                            features.append(features_direction)
                        except :
                            continue

                # Calcula rul para as 3 direções
                RUL = rul(self.path, time_unit, time_lag)
                RUL.calcula_rul(data_hour,rpm,fs,sinais, features)







        

        




                    
