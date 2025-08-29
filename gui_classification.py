import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from datetime import datetime
from scipy.signal.windows import hann
import dynamsignal as ds
import dynamplot as dp
import datetime

class GUI_classification:
    def __init__(self, master, path):
        self.master = master
        self.path = path
        self.path_h = self.path_v = self.path_a = None

        for pasta in os.listdir(self.path):
            full_path = os.path.join(self.path, pasta)
            if os.path.isdir(full_path):
                if pasta[-2:].upper() == "HA":
                    self.path_h = full_path
                elif pasta[-2:].upper() == "VV":
                    self.path_v = full_path
                elif pasta[-2:].upper() == "AV":
                    self.path_a = full_path

        self.dados_class_h, self.dados_analise_h, self.dados_sinal_h = self.load_npz(self.path_h)
        self.dados_class_v, self.dados_analise_v, self.dados_sinal_v = self.load_npz(self.path_v)
        self.dados_class_a, self.dados_analise_a, self.dados_sinal_a = self.load_npz(self.path_a)

        all_datas = []
        for dados in [self.dados_class_h, self.dados_class_v, self.dados_class_a]:
            if dados is not None:
                idx_data = np.where(dados['columns'] == 'DATA')[0][0]
                all_datas.extend(list(dados['data'][:, idx_data]))

        timestamps = pd.to_datetime(all_datas, dayfirst=True)
        grouped_times = timestamps.floor('h').strftime('%d/%m/%Y %H:00:00')
        self.grouped_hours = sorted(list(set(grouped_times)), key=lambda x: pd.to_datetime(x, dayfirst=True), reverse=True)
        self.current_date_index = 0

        for widget in self.master.winfo_children():
            widget.destroy()

        top_frame = tk.Frame(self.master, background='white')
        top_frame.place(relx=0.02, rely=0, relwidth=1, relheight=0.05)

        self.combo_datas = ttk.Combobox(top_frame, values=self.grouped_hours, state='normal', width=25)
        self.combo_datas.current(len(self.grouped_hours) - 1)
        self.combo_datas.pack(side=tk.LEFT, padx=5)
        self.combo_datas.bind('<<ComboboxSelected>>', self.on_date_change)

        tk.Button(top_frame, text='Anterior', command=self.acao_anterior).pack(side='left', padx=5)
        tk.Button(top_frame, text='Próximo', command=self.acao_proximo).pack(side='left', padx=5)

        self.label_info = tk.Label(top_frame, text='', font=('Arial', 12, 'bold'), background='white')
        self.label_info.pack(side=tk.LEFT, padx=20)

        # Criando o main_frame
        main_frame = tk.Frame(master)
        main_frame.place(relx=0.02, rely=0.05, relwidth=0.96, relheight=0.95)

        # Left Frame onde o pizza_notebook será colocado
        left_frame = tk.Frame(main_frame)
        left_frame.place(relx=0, rely=0, relwidth=0.3, relheight=0.95)

        # Criando o pizza_notebook
        self.pizza_notebook = ttk.Notebook(left_frame, height=250)
        self.pizza_notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Adicionando a scrollbar abaixo do pizza_notebook
        # self.scrollbar = ttk.Notebook(left_frame, height=300)
        # self.scrollbar.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Right frame e outras partes do layout
        right_frame = tk.Frame(main_frame)
        right_frame.place(relx=0.31, rely=0, relwidth=0.69, relheight=1)

        # Notebook para os três bubble charts
        self.bubble_notebook = ttk.Notebook(right_frame, height=250)
        self.bubble_notebook.place(relx=0, rely=0, relwidth=1, relheight=0.3)

        self.tabela_frame = ttk.LabelFrame(right_frame, height=250)
        self.tabela_frame.place(relx=0, rely=0.3, relwidth=1, relheight=0.3)

        self.signals_notebook = ttk.Notebook(right_frame, height=250)
        self.signals_notebook.place(relx=0, rely=0.6, relwidth=1, relheight=0.3)

        self.update_all()

    def load_npz(self, path_dir):
        if path_dir is None:
            return None, None, None
        npz_class_path = os.path.join(path_dir, "SPYAI_pred.npz")
        npz_analise_path = os.path.join(path_dir, "SPYAI_analise.npz")
        if os.path.exists(npz_class_path):
            files = os.listdir(path_dir)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found:
                return np.load(npz_class_path, allow_pickle=True), np.load(npz_analise_path, allow_pickle=True), np.load(os.path.join(path_dir, found), allow_pickle=True)
        return None, None, None
    
    def on_date_change(self, event):
        self.current_date_index = self.combo_datas.current()
        self.update_all()
    
    def acao_anterior(self):
        if self.current_date_index < len(self.grouped_hours) - 1:
            self.current_date_index += 1
            self.combo_datas.current(self.current_date_index)
            self.update_all()

    def acao_proximo(self):
        if self.current_date_index > 0:
            self.current_date_index -= 1
            self.combo_datas.current(self.current_date_index)
            self.update_all()
    
    def directions_data(self, data_hour):
        def exists(dados_analise, dados_sinal):
            if dados_analise is None or dados_sinal is None:
                return False
            idx_data_analise = np.where(dados_analise['columns'] == 'DATA')[0][0]
            idx_data_sinal = np.where(dados_sinal['columns'] == 'DTS')[0][0]
            datas_analise = pd.to_datetime(dados_analise['data'][:, idx_data_analise], dayfirst=True).floor('h').strftime('%d/%m/%Y %H:00:00')
            datas_sinal = pd.to_datetime(dados_sinal['data'][:, idx_data_sinal], dayfirst=True).floor('h').strftime('%d/%m/%Y %H:00:00')
            return data_hour in datas_analise and data_hour in datas_sinal

        self.direction_h = exists(self.dados_class_h, self.dados_sinal_h)
        self.direction_v = exists(self.dados_class_v, self.dados_sinal_v)
        self.direction_a = exists(self.dados_class_a, self.dados_sinal_a)

    def update_all(self):
        self.update_info_label()
        self.build_bubble_chart()
        self.build_tabela()
        self.build_signal_tabs()
        self.build_pizza_chart()
        #self.build_scroll()
    
    def update_info_label(self):
        data_hour = self.grouped_hours[self.current_date_index]
        self.directions_data(data_hour)

        idx_tag = np.where(self.dados_class_h['columns'] == 'TAG')[0][0]
        idx_ponto = np.where(self.dados_class_h['columns'] == 'PONTO')[0][0]
        row = self.dados_class_h['data'][self.current_date_index]
        tag = row[idx_tag]
        ponto = row[idx_ponto]
        info = f"TAG: {tag} | Ponto: {ponto[:-2]} | Data: {data_hour}"
        self.label_info.config(text=info)
    
    def build_bubble_chart(self):
        # Limpa as abas existentes
        for tab in self.bubble_notebook.tabs():
            self.bubble_notebook.forget(tab)

        # Para cada direção, gera um bubble chart
        for direcao, path_dir, label in [
            ('H', self.path_h, 'Horizontal'),
            ('V', self.path_v, 'Vertical'),
            ('A', self.path_a, 'Axial')
        ]:
            # Verifica se o diretório está presente e não está vazio
            if path_dir is None or not os.path.isdir(path_dir) or not os.listdir(path_dir):
                # Se não houver dados, mostra a mensagem "Sem aquisição de dados"

                fig, ax = plt.subplots(figsize=(6, 2))
                ax.text(0.5, 0.5, 'Sem aquisição de dados', horizontalalignment='center', verticalalignment='center', fontsize=12, color='red')
                ax.set_ylim(0, 1)
                ax.set_xlim(-1, 2)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()

                # Coloca no notebook
                frame = tk.Frame(self.bubble_notebook)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
                self.bubble_notebook.add(frame, text=label)
                plt.close(fig)
                continue

            # Caso o diretório tenha dados, carrega os arquivos
            npz = np.load(os.path.join(path_dir, "SPYAI_pred.npz"), allow_pickle=True)
            cols = npz['columns']
            data = npz['data']

            # Extrai datas e alertas
            if direcao == 'H':
                idx_data = np.where(cols == 'DATA')[0][0]
                idx_alerta_h = np.where(cols == 'Alerta')[0][0]
                datas_raw = pd.to_datetime(data[:, idx_data], dayfirst=True, errors='coerce').to_pydatetime()
                alertas_raw = data[:, idx_alerta_h]
                # Filtra só valores válidos
                valid = [(d, float(a)) for d, a in zip(datas_raw, alertas_raw) if a != '--']
                if not valid:
                    continue
                valid.sort(key=lambda x: x[0])
                datas, alertas = zip(*valid)
                datas_h = datas 
                alertas_h = alertas
            else:
                idx_data = np.where(cols == 'DATA')[0][0]
                idx_alerta_h = np.where(cols == 'Alerta')[0][0]
                datas_raw = pd.to_datetime(data[:, idx_data], dayfirst=True, errors='coerce').to_pydatetime()
                alertas_raw = data[:, idx_alerta_h]
                # Filtra só valores válidos
                valid = [(d, float(a)) for d, a in zip(datas_raw, alertas_raw) if a != '--']
                if not valid:
                    continue
                valid.sort(key=lambda x: x[0])
                datas_d, alertas_d = zip(*valid) 
                datas = [] 
                alertas = []  
                datas_hfloat = self.dates_to_numeric(datas_h, type = 'horas')
                datas_dfloat = self.dates_to_numeric(datas_d, type = 'horas') 
                for i in range( len(datas_d) ) : 
                    index = np.argmin( np.abs(datas_dfloat[i]-datas_hfloat) )
                    if np.abs(datas_dfloat[i]-datas_hfloat[index]) < 1 : 
                        datas.append(datas_d[i]) 
                        alertas.append( alertas_h[index] ) 

            # Monta coordenadas
            x_vals = np.arange(len(datas))
            colors = {0: 'green', 1: 'blue', 2: 'yellow', 3: 'red'}
            fig, ax = plt.subplots(figsize=(max(6, len(datas) * 0.4), 2))
            ax.scatter(
                x_vals, [0.5] * len(x_vals),
                s=250,
                c=[colors.get(int(a), 'gray') for a in alertas],
                edgecolors='black'
            )
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks(x_vals)
            ax.set_xticklabels([d.strftime('%d/%m/%Y') for d in datas],
                            rotation=45, ha='right')
            fig.tight_layout()

            # Adiciona no notebook com o layout correto
            frame = tk.Frame(self.bubble_notebook)
            scroll_canvas = tk.Canvas(frame, height=300)
            scroll_x = tk.Scrollbar(frame, orient='horizontal', command=scroll_canvas.xview)
            scroll_canvas.configure(xscrollcommand=scroll_x.set)

            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            scroll_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            inner_frame = tk.Frame(scroll_canvas)
            canvas_plot = FigureCanvasTkAgg(fig, master=inner_frame)
            canvas_plot.draw()
            canvas_plot.get_tk_widget().pack()
            scroll_canvas.create_window((0, 0), window=inner_frame, anchor='nw')
            inner_frame.update_idletasks()
            scroll_canvas.config(scrollregion=scroll_canvas.bbox("all"))
            self.bubble_notebook.add(frame, text=label)
            plt.close(fig)

    def build_tabela(self):
        # Limpa os widgets existentes na tabela
        for w in self.tabela_frame.winfo_children():
            w.destroy()
        data_hour = self.grouped_hours[self.current_date_index]
        for label, dados, flag in zip(['H', 'V', 'A'],
                                      [self.dados_class_h, self.dados_class_v, self.dados_class_a],
                                      [self.direction_h, self.direction_v, self.direction_a]):
            frame_label = tk.LabelFrame(self.tabela_frame, text=f"Direção {label}")
            frame_label.pack(side='left', fill='both', expand=True, padx=5, pady=2)
            tree = ttk.Treeview(frame_label, columns=('Campo', 'Valor'), show='headings', height=5)
            tree.heading('Campo', text='Campo')
            tree.heading('Valor', text='Valor')
            tree.column('Campo', width=100)
            tree.column('Valor', width=120)
            tree.pack(padx=5, pady=5, fill='both', expand=True)
            if dados is not None and flag:
                try:
                    cols = dados['columns']
                    idx_data = np.where(cols == 'DATA')[0][0]
                    datas_all = pd.to_datetime(dados['data'][:, idx_data], dayfirst=True)
                    linhas = np.where(datas_all.floor('h').strftime('%d/%m/%Y %H:00:00') == data_hour)[0]
                    if len(linhas) == 0:
                        continue
                    row = dados['data'][linhas[-1]]
                    idx_alerta = np.where(cols == 'Alerta')[0][0]
                    idx_falha1 = np.where(cols == 'TIPO FALHA - PROB. 1')[0][0]
                    idx_falha2 = np.where(cols == 'TIPO FALHA - PROB. 2')[0][0]
                    idx_falha3 = np.where(cols == 'TIPO FALHA - PROB. 3')[0][0]
                    idx_desb = np.where(cols == 'DESB.')[0][0]
                    idx_raz = np.where(cols == 'Razão 1X ')[0][0]

                    # Lógica para cor do alerta
                    if label == 'H':
                        alerta_value = int(float(row[idx_alerta]))
                        alerta_color = {
                            0: ('VERDE', 'black'),
                            1: ('AZUL', 'black'),
                            2: ('AMARELO', 'black'),
                            3: ('VERMELHO', 'black')
                        }

                        # Verifica se o alerta está em nosso dicionário
                        alerta_text, alerta_color_code = alerta_color.get(alerta_value, ('N/A', 'black'))

                        # Inserindo as informações na treeview
                        alerta_h = ("ALERTA", alerta_text)
                        alerta = alerta_h
                    else:
                        alerta = alerta_h
                    tree.insert('', 'end', values=alerta, tags=('alerta',))  # Usando a tag 'alerta' para a cor
                    tree.insert('', 'end', values=("FALHA 1", row[idx_falha1]))
                    tree.insert('', 'end', values=("FALHA 2", row[idx_falha2]))
                    tree.insert('', 'end', values=("FALHA 3", row[idx_falha3]))
                    tree.insert('', 'end', values=("DESBALANCEAMENTO", f"{row[idx_desb]}"))
                    tree.insert('', 'end', values=("RAZÃO 1X", f"{row[idx_raz]}"))
                    tree.insert('', 'end', values=("DATA", row[idx_data]))

                    # Configura as cores da árvore de acordo com a tag
                    tree.tag_configure('alerta', foreground=alerta_color_code)

                except Exception as e:
                    print(f"Erro criando tabela {label}: {e}")
            else:
                tree.insert('', 'end', values=("Status", "Sem aquisição."))
    
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

    def build_signal_tabs(self):
        # Limpa as abas existentes
        for t in self.signals_notebook.tabs():
            self.signals_notebook.forget(t)
        
        data_hour = self.grouped_hours[self.current_date_index]
        self.directions_data(data_hour)

        # Extração dos sinais para as 3 direções
        x_h, dt_h = self.get_signal_and_dt_for_hour(self.dados_analise_h, self.dados_sinal_h, data_hour) if self.direction_h else (None, None)
        x_v, dt_v = self.get_signal_and_dt_for_hour(self.dados_analise_v, self.dados_sinal_v, data_hour) if self.direction_v else (None, None)
        x_a, dt_a = self.get_signal_and_dt_for_hour(self.dados_analise_a, self.dados_sinal_a, data_hour) if self.direction_a else (None, None)
        
        # Gerar os gráficos para aceleração, velocidade e envelope
        fig_acel = dp.plotar_acel(x_h, dt_h, x_v, dt_v, x_a, dt_a, title="Aceleração")
        fig_vel = dp.plotar_vel(x_h, dt_h, x_v, dt_v, x_a, dt_a, title="Velocidade")
        fig_env = dp.plotar_env(x_h, dt_h, x_v, dt_v, x_a, dt_a, title="Envelope")

        # Cria a toolbar interativa uma vez, fora do loop de abas
        frame_toolbar = ttk.Frame(self.signals_notebook)
        frame_toolbar.pack(fill='x', side='bottom', padx=5, pady=5)

        # Adiciona os gráficos nas abas do notebook
        for nome, fig in zip(["Aceleração", "Velocidade", "Envelope"], [fig_acel, fig_vel, fig_env]):
            frame = ttk.Frame(self.signals_notebook)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            self.signals_notebook.add(frame, text=nome)
            
            # Cria a toolbar interativa apenas uma vez
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            toolbar.pack(fill='x', side='bottom', padx=5, pady=5) 


    def build_pizza_chart(self):
        # Limpa as abas existentes
        for tab in self.pizza_notebook.tabs():
            self.pizza_notebook.forget(tab)

        # Para cada direção, gera um gráfico de pizza
        for dados_class, flag, label_dir in [
            (self.dados_class_h, self.direction_h, 'Horizontal'),
            (self.dados_class_v, self.direction_v, 'Vertical'),
            (self.dados_class_a, self.direction_a, 'Axial'),
        ]:
            frame = tk.Frame(self.pizza_notebook)

            # Se não houver dados para essa direção ou horário
            if dados_class is None or not flag:
                msg = tk.Label(frame, text='Sem dados.', fg='red')
                msg.pack(fill='both', expand=True)
                self.pizza_notebook.add(frame, text=label_dir)
                continue

            cols = dados_class['columns']
            data = dados_class['data']

            # Filtra pela hora selecionada
            idx_data = np.where(cols == 'DATA')[0][0]
            datas_all = pd.to_datetime(data[:, idx_data], dayfirst=True)
            horas = datas_all.floor('h').strftime('%d/%m/%Y %H:00:00')
            target = self.grouped_hours[self.current_date_index]
            idxs = np.where(horas == target)[0]
            if len(idxs) == 0:
                msg = tk.Label(frame, text='Sem dados para este horário', fg='red')
                msg.pack(fill='both', expand=True)
                self.pizza_notebook.add(frame, text=label_dir)
                continue

            row = data[idxs[-1]]

            # Extrai os 3 tipos de falha e suas probabilidades
            labels = []
            sizes = []
            for i in [1, 2, 3]:
                col_name = f'TIPO FALHA - PROB. {i}'
                if col_name in cols:
                    entry = row[np.where(cols == col_name)[0][0]]
                    if isinstance(entry, str) and '(' in entry:
                        tipo, prob_str = entry.split('(', 1)
                        labels.append(tipo.strip())
                        # converte '50%' -> 50.0 e verifica se a probabilidade é válida
                        try:
                            prob = float(prob_str.replace('%)', '').strip())
                            if not np.isnan(prob) and 0 <= prob <= 100:
                                sizes.append(prob)
                            else:
                                sizes.append(0)  # Caso a probabilidade seja inválida, coloca 0
                        except ValueError:
                            sizes.append(0)  # Se não conseguir converter, coloca 0
                    else:
                        # fallback caso o formato seja diferente
                        labels.append(str(entry))
                        sizes.append(0)

            # Verifica se há dados válidos para o gráfico de pizza
            if not sizes or all(np.isnan(size) or size == 0 for size in sizes):
                msg = tk.Label(frame, text='Sem dados válidos para gráfico de pizza', fg='red')
                msg.pack(fill='both', expand=True)
                self.pizza_notebook.add(frame, text=label_dir)
                continue

            # Desenha o gráfico de pizza
            fig, ax = plt.subplots(figsize=(6, 6))  # Ajuste o tamanho conforme necessário

            # Gráfico de pizza com as porcentagens dentro
            wedges, texts, autotexts = ax.pie(
                sizes,
                #labels=labels,
                autopct='%1.1f%%',  # Exibe a porcentagem dentro do gráfico
                startangle=90,
                colors=['red', 'orange', 'yellow', 'green'],  # Altere as cores conforme necessário
                wedgeprops={'edgecolor': 'black'}  # Adiciona bordas aos pedaços do gráfico
            )

            # Estilo das porcentagens dentro do gráfico
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color('black')  # Cor do texto da porcentagem

            # Adiciona a legenda à esquerda
            ax.legend(wedges, labels, title="Detectado", loc="center right", bbox_to_anchor=(0, 0.5, 1, 1.5))

            # Ajusta o layout para que tudo fique bem visível
            fig.tight_layout()

            # Insere no notebook
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            self.pizza_notebook.add(frame, text=label_dir)
            plt.close(fig)

    def build_scroll(self):
        # Limpa os widgets existentes na barra de rolagem
        for widget in self.scrollbar.winfo_children():
            widget.destroy()

        # Obtém a data selecionada (data_hour)
        data_hour = self.grouped_hours[self.current_date_index]

        # Cria um Canvas para a área rolável
        scroll_canvas = tk.Canvas(self.scrollbar, height=400)  # Ajuste a altura conforme necessário
        scroll_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Cria a barra de rolagem associada ao canvas
        scrollbar_widget = ttk.Scrollbar(self.scrollbar, orient="vertical", command=scroll_canvas.yview)
        scrollbar_widget.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.configure(yscrollcommand=scrollbar_widget.set)

        # Cria uma estrutura para exibir as direções (H, V, A)
        directions = [
            ('Horizontal', self.dados_class_h, self.direction_h),
            ('Vertical', self.dados_class_v, self.direction_v),
            ('Axial', self.dados_class_a, self.direction_a)
        ]

        # Cria um frame dentro do canvas para organizar o conteúdo
        content_frame = tk.Frame(scroll_canvas)
        scroll_canvas.create_window((0, 0), window=content_frame, anchor="nw")

        # Itera sobre as direções (Horizontal, Vertical, Axial)
        for label, dados_class, flag in directions:
            # Se não houver dados para a direção ou horário, pula
            if dados_class is None or not flag:
                continue

            # Cria um frame para cada direção
            direction_frame = ttk.Frame(content_frame, padding="10")
            direction_frame.pack(fill=tk.X, pady=10, padx=10)

            # Extrai os dados da direção
            cols = dados_class['columns']
            data = dados_class['data']

            # Define os índices para as colunas de dados
            idx_data = np.where(cols == 'DATA')[0][0]
            idx_prob1 = np.where(cols == 'TIPO FALHA - PROB. 1')[0][0]
            idx_prob2 = np.where(cols == 'TIPO FALHA - PROB. 2')[0][0]
            idx_prob3 = np.where(cols == 'TIPO FALHA - PROB. 3')[0][0]

            # Filtra os dados pela data_hour
            for linha in data:
                data_str = linha[idx_data]
                
                try:
                    data = datetime.strptime(data_str, "%d/%m/%Y %H:%M:%S")
                except:
                    data = datetime.strptime(data_str, "%Y-%m-%d %H:%M:%S")
                
                # Filtra pela data_hour (para garantir que estamos usando a data correta)
                if data.strftime('%d/%m/%Y %H:%M:%S') == data_hour:
                    # Formatação da label com a data
                    label_text = f"{data.strftime('%d/%m/%Y %H:%M:%S')}"

                    # Cria uma label para a data
                    tk.Label(direction_frame, text=label_text, font=("Arial", 12, 'bold')).pack(side=tk.TOP, anchor="w", padx=15)

                    # Extrai as probabilidades de falha
                    prob1 = linha[idx_prob1]
                    prob2 = linha[idx_prob2]
                    prob3 = linha[idx_prob3]

                    # Formatação para a falha e suas probabilidades
                    failure_text = f"{label.upper()}:\n Falhas: {prob1} | {prob2} | {prob3}"

                    # Cria uma label para a falha e probabilidades
                    tk.Label(direction_frame, text=failure_text, font=("Arial", 10), anchor="w").pack(side=tk.TOP, anchor="w", padx=15)

        # Atualiza a área rolável do canvas
        content_frame.scroll_idletasks()
        scroll_canvas.config(scrollregion=scroll_canvas.bbox("all"))

        # Move o scroll para o topo após adicionar os itens
        scroll_canvas.yview_moveto(0)

    def date_to_numeric(self, date, type = 'meses') : 
        aux = '%s'%(date) 
        j = aux.find('-')
        if j < 0 : 
            j = aux.find('/')   
        if j < 3 : # data dia - mês - ano ... 
            dia = int(aux[0:2])
            mes = int(aux[3:5])
            ano = int(aux[6:10])
        else : 
            ano = int(aux[0:4])
            mes = int(aux[5:7])
            dia = int(aux[8:10])

        hora = 0
        minuto = 0 
        segundo = 0 
        if len(aux) > 10 : # tem horas minutos e segundos
            hora = int(aux[11:13])
            minuto = int(aux[14:16])
            segundo = int(aux[17:19])
        t = datetime.datetime(ano, mes, dia, hora, minuto, segundo) 
        seconds = t.timestamp()
        if type == 'meses' : 
            return seconds/2628288 # transforma para mês
        if type == 'dias' :
            return seconds/86400 # transforma para dia
        if type == 'horas' : 
            return seconds/3600 # transforma para hora


    def dates_to_numeric(self, dates, type = 'meses') : 
        dates_float = np.zeros( len(dates) )
        for i in range( len(dates) ) : 
            dates_float[i] = self.date_to_numeric(dates[i], type = type) 
        return dates_float
    

# if __name__ == "__main__":
#     root = tk.Tk()
#     path = r"C:/Users/letic/OneDrive - Universidade Federal de Uberlândia/Documentos/SPYAI/SPYAI_13_08_25/Banco de dados/P100/SI1/EXA/EX0401/105.401/105.401ME/AV01"
#     GUI_classification(root, path)
#     root.mainloop()
