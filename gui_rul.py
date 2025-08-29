import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import dynamplot 
import dynamplot as dp
import dynamsignal as ds
class GUI_rul:
    def __init__(self, root, path):
        self.root = root
        for widget in self.root.winfo_children():
            widget.destroy()

        self.path_h, self.path_v, self.path_a = None, None, None

        # === Carregamento dos arquivos ===
        self.path = path
        for pasta in os.listdir(self.path):
            full_path = os.path.join(self.path, pasta)
            if os.path.isdir(full_path):
                if pasta[-2:-1].upper() == "H":
                    self.path_h = full_path
                if pasta[-2:-1].upper() == "V":
                    self.path_v = full_path
                if pasta[-2:-1].upper() == "A":
                    self.path_a = full_path
        
        if self.path_h != None:
            self.dados_analise_h, self.dados_sinal_h, self.dados_class_h = self.load_npz(self.path_h)
        else:
            self.dados_analise_h, self.dados_sinal_h, self.dados_class_h = None, None, None
        if self.path_v != None:
            self.dados_analise_v, self.dados_sinal_v, self.dados_class_v = self.load_npz(self.path_v)
        else:
            self.dados_analise_v, self.dados_sinal_v, self.dados_class_v = None, None, None
        if self.path_a != None:
            self.dados_analise_a, self.dados_sinal_a, self.dados_class_a = self.load_npz(self.path_a)
        else:
            self.dados_analise_a, self.dados_sinal_a, self.dados_class_a = None, None, None
        self.npz_rul = self._carregar_npz(lambda f: f == "SPYAI_rul.npz")

        self.analises = [self.dados_analise_h, self.dados_analise_v, self.dados_analise_a]
        self.sinais =  [self.dados_sinal_h, self.dados_sinal_v, self.dados_sinal_a]
        self.classif = [self.dados_class_h, self.dados_class_v, self.dados_class_a]

        # === Leitura dos dados principais ===
        all_datas = []
        for dados in self.analises:
            if dados is not None:
                idx_data = np.where(dados['columns'] == 'DATA')[0][0]
                all_datas.extend(list(dados['data'][:, idx_data]))

        timestamps = pd.to_datetime(all_datas, dayfirst=True)
        grouped_times = timestamps.floor('h').strftime('%d/%m/%Y %H:00:00')
        self.datas = sorted(list(set(grouped_times)), key=lambda x: pd.to_datetime(x, dayfirst=True), reverse=False)
        self.current_date_index = len(self.datas)-1

        self.columns_class = self.dados_class_h['columns']
        self.data_class = self.dados_class_h['data']
        idx_tag = np.where(self.columns_class == "TAG")[0][0]
        idx_ponto = np.where(self.columns_class == "PONTO")[0][0]
        self.TAG = self.data_class[0, idx_tag]
        self.ponto = self.data_class[0, idx_ponto]

        self.intervalo_confianca = 95

        self._setup_gui()
    
    def load_npz(self, path_dir):
        if path_dir is None:
            return None, None
        npz_path_analise = os.path.join(path_dir, "SPYAI_analise.npz")
        npz_path_class = os.path.join(path_dir, "SPYAI_pred.npz")
        if os.path.exists(npz_path_analise):
            files = os.listdir(path_dir)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found:
                return np.load(npz_path_analise, allow_pickle=True), np.load(os.path.join(path_dir, found), allow_pickle=True), np.load(npz_path_class, allow_pickle=True)
        return None, None, None

    def _carregar_npz(self, filtro):
        nome_arquivo = next((f for f in os.listdir(self.path) if filtro(f)), None)
        if nome_arquivo:
            return np.load(os.path.join(self.path, nome_arquivo), allow_pickle=True)
        else:
            raise FileNotFoundError(f"Arquivo correspondente não encontrado em {self.path}")
    
    def acao_anterior(self):
        self.current_date_index -= 1
        if self.current_date_index < 0 :
            self.current_date_index = 0
        self.combo_datas.current(self.current_date_index)
        self.update_all()

    def acao_proximo(self):
        self.current_date_index += 1
        if self.current_date_index > len(self.datas) - 1:
            self.current_date_index = len(self.datas) - 1
        self.combo_datas.current(self.current_date_index)
        self.update_all()

    def _setup_gui(self):
        # Topo
        frame_topo = tk.Frame(self.root, background='white')
        frame_topo.place(relx=0.02, rely=0, relwidth=0.98, relheight=0.05)

        frame_datas_info = tk.Frame(frame_topo, background='white')
        frame_datas_info.pack(side='left', fill='x')

        self.combo_datas = ttk.Combobox(frame_datas_info, values=self.datas, state='normal', width=15)
        self.combo_datas.current(self.current_date_index)
        self.combo_datas.pack(side='left')
        self.combo_datas.bind('<<ComboboxSelected>>', self.on_data_change)

        tk.Button(frame_datas_info, text='Anterior', command=self.acao_anterior).pack(side='left', padx=5)
        tk.Button(frame_datas_info, text='Próximo', command=self.acao_proximo).pack(side='left', padx=5)

        self.label_info = tk.Label(frame_datas_info, text="", font=('Arial', 11), anchor='w', bg='white')
        self.label_info.pack(side='left', padx=10)

        frame_confianza = tk.Frame(frame_topo)
        frame_confianza.pack(side='left', padx=20)

        label_ic = tk.Label(frame_confianza, text="Intervalo de Confiança:", font=('Arial', 10), bg='white')
        label_ic.pack(side='left', pady=(0,5))

        self.combo_ic = ttk.Combobox(frame_confianza, values=[80, 90, 95], state='normal', width=5)
        self.combo_ic.current(2)
        self.combo_ic.pack(side='top')
        self.combo_ic.bind('<<ComboboxSelected>>', self.on_ic_change)

        # Gráfico POF
        self.frame_pof = tk.Frame(self.root, background='white')
        self.frame_pof.place(relx=0.02, rely=0.07, relwidth=0.98, relheight=0.4)

        self.fig_pof, self.ax_pof = plt.subplots(figsize=(24, 10))
        self.canvas_pof = FigureCanvasTkAgg(self.fig_pof, master=self.frame_pof)
        self.canvas_pof.get_tk_widget().pack()

        self.frame_pred = tk.Frame(self.root, background='white')
        self.frame_pred.place(relx=0.02, rely=0.45, relwidth=0.98, relheight=0.15)

        self.label_classificacao = tk.Label(self.frame_pred, text="", font=('Arial', 12), fg='black', anchor='w',bg='white')
        self.label_classificacao.pack(pady=5, fill='x')

        self.label_previsao = tk.Label(self.frame_pred, text="", font=('Arial', 11), fg='black', wraplength=650, justify='left', anchor='w',bg='white')
        self.label_previsao.pack(pady=5, fill='x')

        # Abas
        self.frame_bottom = tk.Frame(self.root)
        self.frame_bottom.place(relx=0.02, rely=0.55, relwidth=0.98, relheight=0.35)

        self.notebook = ttk.Notebook(self.frame_bottom)
        self.notebook.pack(fill='both', expand=True)

        self.tab_variacao = tk.Frame(self.notebook)
        self.notebook.add(self.tab_variacao, text="VARIAÇÃO")

        self.tab_parametros = tk.Frame(self.notebook)
        self.notebook.add(self.tab_parametros, text="PARÂMETROS")

        self.tab_sinal = tk.Frame(self.notebook)
        self.notebook.add(self.tab_sinal, text="SINAL")

        self.update_all()

    def update_label_info(self):
        idx = self.current_date_index
        info = f"TAG: {self.TAG}  |  Ponto: {self.ponto[:-2]}  |  Data: {self.datas[idx]}"
        self.label_info.config(text=info)

    def update_all(self):
        idx = self.current_date_index
        # idx_analise = self.current_date_index_analise

        self.update_label_info()
       
        # === POF ===
        self.ax_pof.clear()
        pof = f'pof{self.intervalo_confianca}%'
        idx_pof = np.where(self.npz_rul['columns'] == pof)[0][0]
        idx_msg = np.where(self.npz_rul['columns'] == 'msg')[0][0]
        len_pred = int(len(self.npz_rul['data'][0,idx_pof])/3)
        mensagens = self.npz_rul['data'][:,idx_msg]
        pof_datas_anteriores = np.zeros( idx+1 )
        mes = np.zeros( idx+1 )
        for i in range(1,idx+1):
            if self.npz_rul['data'][i,idx_pof][0] < .1 : 
                pof_datas_anteriores[i-1] = float(self.npz_rul['data'][1,2][-6:-2])
            else : 
                pof_datas_anteriores[i-1] = self.npz_rul['data'][i,idx_pof][len_pred+len_pred-1]
            mes[i-1] =  self.npz_rul['data'][i,3]   

        plt.close()
        plt.figure(figsize=(50, 20))
        self.ax_pof.plot(mes[0:idx], pof_datas_anteriores[0:idx], marker='o', label='Probability of Failure (POF)')
        IC_inf = self.npz_rul['data'][idx+1,idx_pof][0:len_pred]
        IC_mean = self.npz_rul['data'][idx+1,idx_pof][len_pred:2*len_pred]
        IC_sup = self.npz_rul['data'][idx+1,idx_pof][2*len_pred:3*len_pred]
        mes_pred = mes[idx-1]+np.arange(0,len_pred)
        mes_pred = np.concatenate(([mes[idx-1]],mes_pred))
        IC_inf = np.concatenate(([IC_mean[0]],IC_inf))
        IC_sup = np.concatenate(([IC_mean[0]],IC_sup))
        IC_mean = np.concatenate(([IC_mean[0]],IC_mean))
        
        self.ax_pof.plot(mes_pred,  IC_inf, '--g', label = 'Limite Inferior')
        self.ax_pof.plot(mes_pred, IC_mean, '--b')
        self.ax_pof.plot(mes_pred, IC_sup, '--r',label = 'Limite Superior')
        self.ax_pof.fill_between(mes_pred,IC_mean,IC_inf,color='blue' ,alpha=0.2)
        self.ax_pof.fill_between(mes_pred,IC_sup,IC_mean, color='red' ,alpha=0.2)
    
        self.ax_pof.set_ylim(0, 100)
        self.ax_pof.set_ylabel('POF')
        self.ax_pof.set_title('Probabilidade de Falha ao Longo do Tempo')
        self.ax_pof.grid(True)
        self.ax_pof.legend(loc='upper left')
        self.fig_pof.autofmt_xdate(rotation=45)
        self.canvas_pof.draw()

        idx_class1 = np.where(self.dados_class_h['columns'] == 'TIPO FALHA - PROB. 1')[0][0]
        idx_class2 = np.where(self.dados_class_h['columns'] == 'TIPO FALHA - PROB. 2')[0][0]
        idx_class3 = np.where(self.dados_class_h['columns'] == 'TIPO FALHA - PROB. 3')[0][0]
        self.label_classificacao.config(text=f"CLASSIFICAÇÃO: {self.dados_class_h['data'][idx, idx_class1]} | {self.dados_class_h['data'][idx, idx_class2]} | {self.dados_class_h['data'][idx, idx_class3]}")
        self.label_previsao.config(text=f"PREVISÃO: {self.npz_rul['data'][idx+1,idx_msg]}")

        # === Variação ===
        for w in self.tab_variacao.winfo_children():
            w.destroy()

        fig_v = plt.subplots(1, 1, figsize=(8, 6))

        # Dados
        idx_anomaly = np.where(self.npz_rul['columns'] == 'anomalia-anterior')[0][0]
        y = self.npz_rul['data'][0:idx, idx_anomaly]
        x = mes[0:idx]
        fig_v = self.evolution_fig(x,y)

        # Integrar ao Tkinter
        canvas_v = FigureCanvasTkAgg(fig_v, master=self.tab_variacao)
        canvas_v.get_tk_widget().pack(fill='both', expand=True)


        # === Parâmetros ===
        for w in self.tab_parametros.winfo_children(): w.destroy()

        def criar_figura(direction, feat, idx, idx_analise, intervalo=95):
            directions = ["Horizontal", "Vertical", "Axial"]
            parametros = ['harm1', 'v_10_1000' , 'acel_high' , 'JB_harm6' , 'low_order_rms' , 'rms_no_harm_1000' , 'high_order_rms']
            #mes = np.zeros( idx+1 )
            idx_direction = directions.index(direction)
            idx_param = parametros.index(feat)
            idx_80 = [np.arange(0,6), np.arange(6,12), np.arange(12,18)]
            idx_90 = [np.arange(18,24), np.arange(24,30), np.arange(30,36)]
            idx_95 = [np.arange(36,42), np.arange(42,48), np.arange(48,54)]
            if intervalo == 95:
                idx_pof = idx_95
            if intervalo == 90:
                idx_pof = idx_90
            if intervalo == 80:
                idx_pof = idx_80
            try :
                idx_feature_analise = np.where(self.analises[idx_direction]['columns'] == feat)[0][0]
                idx_data_analise = np.where(self.analises[idx_direction]['columns'] == 'DATA')[0][0]
                datas_analise = self.analises[idx_direction]['data'][:,idx_data_analise]
                aux = self.analises[idx_direction]['data'][:,idx_data_analise]
                datas_analise_sort = np.zeros(len(datas_analise)) 
                for i in range(len(datas_analise)) :
                    datas_analise_sort[i] = ds.utilities.date_to_numeric(datas_analise[i],type='horas') 
               
                index = np.argsort(datas_analise_sort) 
                data_hour_sort = ds.utilities.date_to_numeric(datas_analise[self.current_date_index],type='horas')
                datas_analise = datas_analise[index] 
                datas_analise_sort = datas_analise_sort[index]
                data_hour_sort = ds.utilities.date_to_numeric(datas_analise[self.current_date_index],type='horas')                   
                feature_passadas = [] 
                mes = [] 
                mes_base = ds.utilities.date_to_numeric(datas_analise[0],type='meses') 
                for m in range(len(datas_analise)):
                    if datas_analise_sort[m] < data_hour_sort:
                        idx_linha = np.where(aux == datas_analise[m])[0]
                        feature_passadas.append(self.analises[idx_direction]['data'][idx_linha,idx_feature_analise])
                        mes.append( ds.utilities.date_to_numeric(datas_analise[m],type='meses') - mes_base   )

                idx_features_pred = np.where(self.npz_rul['columns'] == 'features_pred')[0][0]
                features_pred = np.array(self.npz_rul['data'][idx,idx_features_pred])
                n_features = int(len(features_pred)/(3*54))
                features_pred = features_pred.reshape(3,n_features,54)
                if features_pred[idx_direction,idx_param,0] < 1e-8 : 
                    fig_param, ax_param = plt.subplots(figsize=(8, 8))
                    ax_param.plot(mes, feature_passadas, marker='o', label='Previsao')
                else : 
                    fig_param, ax_param = plt.subplots(figsize=(8, 8))
                    ax_param.plot(mes, feature_passadas, marker='o', label='Previsao')
                    IC_inf = np.array(features_pred[idx_direction,idx_param,idx_pof[0]], dtype=float)
                    IC_mean = np.array(features_pred[idx_direction,idx_param,idx_pof[1]], dtype=float)
                    IC_sup = np.array(features_pred[idx_direction,idx_param,idx_pof[2]], dtype = float)
                    mes_pred = mes[-1]+np.arange(0,len_pred)
                    ganho = feature_passadas[-1][0]/IC_mean[0]
                    mes_pred = np.concatenate(([mes[-1]],mes_pred))
                    IC_inf = np.concatenate(([IC_mean[0]],IC_inf))*ganho
                    IC_sup = np.concatenate(([IC_mean[0]],IC_sup))*ganho
                    IC_mean = np.concatenate(([IC_mean[0]],IC_mean))*ganho       
                    ax_param.plot(mes_pred,  IC_inf, '--g', label = 'Limite Inferior')
                    ax_param.plot(mes_pred, IC_mean, '--b')
                    ax_param.plot(mes_pred, IC_sup, '--r',label = 'Limite Superior')
                    ax_param.fill_between(mes_pred,IC_mean,IC_inf,color='blue' ,alpha=0.2)
                    ax_param.fill_between(mes_pred,IC_sup,IC_mean, color='red' ,alpha=0.2)
            
                ax_param.set_ylabel('Amplitude')
                ax_param.set_title('Mês')
                ax_param.grid(True)
                ax_param.legend(loc='upper left')
                fig_param.tight_layout()
                return fig_param
            except : 
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.text(0.5, 0.5, 'Sem aquisição de dados', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
                ax.set_ylim(0, 1)
                ax.set_xlim(-1, 2)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                return fig


        self.nb_param = ttk.Notebook(self.tab_parametros)
        self.nb_param.pack(fill='both', expand=True)

        # Definição das abas principais e figuras
        directions = ["Horizontal", "Vertical", "Axial"]
        nomes_graficos = ['harm1', 'v_10_1000' , 'acel_high' , 'JB_harm6' , 'low_order_rms' , 'rms_no_harm_1000' , 'high_order_rms']
        #nomes_graficos = ["Harm 1", "RMS", "Envelope", "Harm 6", "Low Order", "No harm", "High Order"]

        # Loop para criar cada aba de direção
        for direction in directions:
            # Cria a aba de direção
            frame_direcao = ttk.Frame(self.nb_param)
            self.nb_param.add(frame_direcao, text=direction)

            # Notebook interno para os gráficos dentro da direção
            nb_graficos = ttk.Notebook(frame_direcao)
            nb_graficos.pack(fill='both', expand=True)

            # Adiciona os gráficos como abas dentro da direção
            for nome in nomes_graficos:
                frame_fig = ttk.Frame(nb_graficos)
                nb_graficos.add(frame_fig, text=nome)

                # Canvas dos gráficos
                data_hour = self.datas[self.current_date_index]
                idx_analise = self.directions_data(data_hour, idx_analise=True, direction=directions.index(direction))
                fig = criar_figura(direction,nome,idx,idx_analise,intervalo=self.intervalo_confianca)
                canvas = FigureCanvasTkAgg(fig, master=frame_fig)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)

        # === Sinais ===
        for w in self.tab_sinal.winfo_children(): w.destroy()

        data_hour = self.datas[self.current_date_index]
        self.directions_data(data_hour)

        # Extração dos sinais para as 3 direções
        x_h, dt_h = self.get_signal_and_dt_for_hour(self.dados_analise_h, self.dados_sinal_h, data_hour) if self.direction_h else (None, None)
        x_v, dt_v = self.get_signal_and_dt_for_hour(self.dados_analise_v, self.dados_sinal_v, data_hour) if self.direction_v else (None, None)
        x_a, dt_a = self.get_signal_and_dt_for_hour(self.dados_analise_a, self.dados_sinal_a, data_hour) if self.direction_a else (None, None)
        
        # Gerar os gráficos para aceleração, velocidade e envelope
        fig_acel = dp.plotar_acel(x_h, dt_h, x_v, dt_v, x_a, dt_a, title="Aceleração")
        fig_vel = dp.plotar_vel(x_h, dt_h, x_v, dt_v, x_a, dt_a, title="Velocidade")
        fig_env = dp.plotar_env(x_h, dt_h, x_v, dt_v, x_a, dt_a, title="Envelope")

        # 1) Cria um sub-notebook dentro da aba SINAL
        self.nb_sinal = ttk.Notebook(self.tab_sinal)
        self.nb_sinal.pack(fill='both', expand=True)

        #Cria a toolbar interativa uma vez, fora do loop de abas
        frame_toolbar = ttk.Frame(self.tab_sinal)
        frame_toolbar.pack(fill='x', side='bottom', padx=5, pady=5)

        # 2) Cria as abas Aceleração, Velocidade e Envelope DENTRO do sub-notebook
        for nome, fig in zip(["Aceleração", "Velocidade", "Envelope"], [fig_acel, fig_vel, fig_env]):
            frame = ttk.Frame(self.nb_sinal)
            self.nb_sinal.add(frame, text=nome)

            canvas = FigureCanvasTkAgg(fig, master=frame)  # canvas dentro do frame da aba
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

            # Cria a toolbar interativa apenas uma vez
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            toolbar.pack(fill='x', side='bottom', padx=5, pady=5)

    
    def directions_data(self, data_hour, idx_analise = False, direction=0):
        def exists(dados_analise, dados_sinal):
            if dados_analise is None or dados_sinal is None:
                return False
            idx_data_analise = np.where(dados_analise['columns'] == 'DATA')[0][0]
            idx_data_sinal = np.where(dados_sinal['columns'] == 'DTS')[0][0]
            datas_analise = pd.to_datetime(dados_analise['data'][:, idx_data_analise], dayfirst=True).floor('h').strftime('%d/%m/%Y %H:00:00')
            datas_sinal = pd.to_datetime(dados_sinal['data'][:, idx_data_sinal], dayfirst=True).floor('h').strftime('%d/%m/%Y %H:00:00')
            if idx_analise:
                return idx_data_analise
            return data_hour in datas_analise and data_hour in datas_sinal

        if idx_analise == False:
            self.direction_h = exists(self.dados_analise_h, self.dados_sinal_h)
            self.direction_v = exists(self.dados_analise_v, self.dados_sinal_v)
            self.direction_a = exists(self.dados_analise_a, self.dados_sinal_a)
        else:
            if direction == 0:
                idx_analise = exists(self.dados_analise_h, self.dados_sinal_h)
            if direction == 1:
                idx_analise = exists(self.dados_analise_v, self.dados_sinal_v)
            if direction == 2:
                idx_analise = exists(self.dados_analise_a, self.dados_sinal_a)
            return idx_analise
    
    def get_signal_and_dt_for_hour(self, dados_analise, dados_sinal, target_hour):
        if dados_analise is None or dados_sinal is None:
            return None, None
        idx_data_analise = np.where(dados_analise['columns'] == 'DATA')[0][0]
        idx_data_sinal = np.where(dados_sinal['columns'] == 'DTS')[0][0]
        idx_amostras = np.where(dados_sinal['columns'] == 'Amostras')[0][0]
        idx_dado = np.where(dados_sinal['columns'] == 'Dado')[0][0]
        idx_dt = np.where(dados_analise['columns'] == 'dt')[0][0]

        datas_analise = pd.to_datetime(dados_analise['data'][:, idx_data_analise], dayfirst=True)
        datas_sinal = pd.to_datetime(dados_sinal['data'][:, idx_data_sinal], dayfirst=True)

        linhas_analise = np.where(datas_analise.floor('h').strftime('%d/%m/%Y %H:00:00') == target_hour)[0]
        linhas_sinal = np.where(datas_sinal.floor('h').strftime('%d/%m/%Y %H:00:00') == target_hour)[0]

        if len(linhas_analise) == 0 or len(linhas_sinal) == 0:
            return None, None

        row_analise = dados_analise['data'][linhas_analise[-1]]
        row_sinal = dados_sinal['data'][linhas_sinal[-1]]
        amostras = int(row_sinal[idx_amostras])
        sinal = row_sinal[idx_dado: idx_dado + amostras].astype(float)
        sinal = sinal[~np.isnan(sinal)]
        dt = row_analise[idx_dt]
        return sinal, dt

    def on_data_change(self, event):
        self.current_date_index = self.combo_datas.current()
        self.update_all()

    def on_ic_change(self, event):
        self.intervalo_confianca = int(self.combo_ic.get())
        self.update_all()
    
    def evolution_fig(self,mes, annomaly) : 
        npto = len(mes)
        x_g =[]  
        x_b = []
        x_y = []   
        x_r = []     
        y_g =[] 
        y_b = [] 
        y_y = [] 
        y_r = [] 
        for i in range(npto) : 
            if annomaly[i] > 2 : 
                x_r.append(mes[i]) 
                y_r.append(annomaly[i]) 
            else : 
                if annomaly[i] > 1 : 
                    x_y.append(mes[i]) 
                    y_y.append(annomaly[i])
                elif annomaly[i] > 0 : 
                    x_b.append(mes[i]) 
                    y_b.append(annomaly[i])  
                else : 
                    x_g.append(mes[i]) 
                    y_g.append(annomaly[i])      
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))   
        if len(x_g) > 0 :
                axs.plot(x_g, y_g, '*g',markersize=12 )
        if len(x_b) > 0 :
                axs.plot(x_b, y_b, '*b',markersize=12 )
        if len(x_y) > 0 :
                axs.plot(x_y, y_y, '*y',markersize=12 )
        if len(x_r) > 0 :
                axs.plot(x_r, y_r, '*r',markersize=12 )
        axs.set_xlabel('Mês')
        axs.set_ylim([-.5, 3.5] )
        axs.grid()
        fig.tight_layout()
        return fig

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x700")
    path = '/home/mvduarte/SPYAI_16_08_25/Banco de dados/P100/SI1/EXA/EX0401/105.401/105.401ME/AV01'
    app = GUI_rul(root, path)
    root.mainloop()