import os
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import dynamplot 
class GUI_analise:
    def __init__(self, container, path):
        self.path = path
        self.path_h = self.path_v = self.path_a = None

        for pasta in os.listdir(self.path):
            full_path = os.path.join(self.path, pasta)
            if os.path.isdir(full_path):
                if pasta[-2:-1].upper() == "H":
                    self.path_h = full_path
                elif pasta[-2:-1].upper() == "V":
                    self.path_v = full_path
                elif pasta[-2:-1].upper() == "A":
                    self.path_a = full_path

        self.dados_analise_h, self.dados_sinal_h = self.load_npz(self.path_h)
        self.dados_analise_v, self.dados_sinal_v = self.load_npz(self.path_v)
        self.dados_analise_a, self.dados_sinal_a = self.load_npz(self.path_a)

        all_datas = []
        for dados in [self.dados_analise_h, self.dados_analise_v, self.dados_analise_a]:
            if dados is not None:
                idx_data = np.where(dados['columns'] == 'DATA')[0][0]
                all_datas.extend(list(dados['data'][:, idx_data]))

        timestamps = pd.to_datetime(all_datas, dayfirst=True)
        grouped_times = timestamps.floor('h').strftime('%d/%m/%Y %H:00:00')
        self.grouped_hours = sorted(list(set(grouped_times)), key=lambda x: pd.to_datetime(x, dayfirst=True), reverse=True)
        self.current_date_index = 0

        for widget in container.winfo_children():
            widget.destroy()

        self.top_frame = tk.Frame(container, background='white')
        self.top_frame.place(relx=0.02, rely=0, relwidth=1, relheight=0.05)

        self.top_tables_frame = tk.Frame(container, bg='white')
        self.top_tables_frame.place(relx=0.02, rely=0.05, relwidth=0.96, relheight=0.20)

        self.middle_frame = tk.Frame(container, bg='white')
        self.middle_frame.place(relx=0.02, rely=0.25, relwidth=1, relheight=0.40)

        self.bottom_frame = tk.Frame(container, bg='white')
        self.bottom_frame.place(relx=0.02, rely=0.65, relwidth=0.98, relheight=0.25)

        self.combo_data = ttk.Combobox(self.top_frame, values=self.grouped_hours, state='normal')
        self.combo_data.current(self.current_date_index)
        self.combo_data.pack(side='left', padx=5)
        self.combo_data.bind('<<ComboboxSelected>>', self.on_data_selected)

        tk.Button(self.top_frame, text='Anterior', command=self.acao_anterior).pack(side='left', padx=5)
        tk.Button(self.top_frame, text='Próximo', command=self.acao_proximo).pack(side='left', padx=5)

        self.label_info = tk.Label(self.top_frame, text='', font=('Arial', 12, 'bold'), background='white')
        self.label_info.pack(side='left', padx=20)

        self.reload_all()


    def load_npz(self, path_dir):
        if path_dir is None:
            return None, None
        npz_path = os.path.join(path_dir, "SPYAI_analise.npz")
        if os.path.exists(npz_path):
            files = os.listdir(path_dir)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found:
                return np.load(npz_path, allow_pickle=True), np.load(os.path.join(path_dir, found), allow_pickle=True)
        return None, None

    def directions_data(self, data_hour):
        def exists(dados_analise, dados_sinal):
            if dados_analise is None or dados_sinal is None:
                return False
            idx_data_analise = np.where(dados_analise['columns'] == 'DATA')[0][0]
            idx_data_sinal = np.where(dados_sinal['columns'] == 'DTS')[0][0]
            datas_analise = pd.to_datetime(dados_analise['data'][:, idx_data_analise], dayfirst=True).floor('h').strftime('%d/%m/%Y %H:00:00')
            datas_sinal = pd.to_datetime(dados_sinal['data'][:, idx_data_sinal], dayfirst=True).floor('h').strftime('%d/%m/%Y %H:00:00')
            return data_hour in datas_analise and data_hour in datas_sinal

        self.direction_h = exists(self.dados_analise_h, self.dados_sinal_h)
        self.direction_v = exists(self.dados_analise_v, self.dados_sinal_v)
        self.direction_a = exists(self.dados_analise_a, self.dados_sinal_a)

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

    def reload_all(self):
        self.build_middle_panel()
        self.build_bottom_panel()
        self.build_top_tables()

    def build_middle_panel(self):
        for w in self.middle_frame.winfo_children():
            w.destroy()

        data_hour = self.grouped_hours[self.current_date_index]
        self.directions_data(data_hour)

        x_h, dt_h = self.get_signal_and_dt_for_hour(self.dados_analise_h, self.dados_sinal_h, data_hour) if self.direction_h else (None, None)
        x_v, dt_v = self.get_signal_and_dt_for_hour(self.dados_analise_v, self.dados_sinal_v, data_hour) if self.direction_v else (None, None)
        x_a, dt_a = self.get_signal_and_dt_for_hour(self.dados_analise_a, self.dados_sinal_a, data_hour) if self.direction_a else (None, None)

        idx_tag = np.where(self.dados_analise_h['columns'] == 'TAG')[0][0]
        idx_ponto = np.where(self.dados_analise_h['columns'] == 'Ponto')[0][0]
        row = self.dados_analise_h['data'][self.current_date_index]
        info = f"TAG: {row[idx_tag]} | Ponto: {row[idx_ponto][:-2]} | Data: {data_hour}"
        self.label_info.config(text=info)

        # fig = dynamplot.plotar_4graficos(x_h, dt_h, x_v, dt_v, x_a, dt_a, title=info)
        # canvas = FigureCanvasTkAgg(fig, master=self.middle_frame)
        # canvas.draw()
        # canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

        # # Adiciona a barra de ferramentas interativa
        # toolbar = NavigationToolbar2Tk(canvas, self.middle_frame)
        # toolbar.update()
        # canvas.get_tk_widget().pack()

        fig = dynamplot.plotar_4graficos(x_h, dt_h, x_v, dt_v, x_a, dt_a, title=info)

        # Cria o canvas do gráfico
        canvas = FigureCanvasTkAgg(fig, master=self.middle_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.place(relx=0, rely=0.1, relwidth=0.98, relheight=0.88)  # Deixa 5% para a toolbar no topo

        # Cria a toolbar interativa
        toolbar = NavigationToolbar2Tk(canvas, self.middle_frame)
        toolbar.update()
        toolbar.place(relx=0, rely=0, relwidth=1, relheight=0.1)  # Ocupa o topo do frame

    def build_bottom_panel(self):
        for w in self.bottom_frame.winfo_children():
            w.destroy()

        abas = ttk.Notebook(self.bottom_frame)
        abas.pack(fill='both', expand=True)

        features_cols = {
            "RMS": 'rms_v',
            "PICO": 'env_peak',
            "CURTOSE": 'kurtosis',
            "FATOR DE CRISTA": 'crest_factor',
            "FATOR K": 'k_factor'
        }

        for nome_aba, col_name in features_cols.items():
            fig, ax = plt.subplots(figsize=(8, 3))
            for label, dados in zip(['H', 'V', 'A'], [self.dados_analise_h, self.dados_analise_v, self.dados_analise_a]):
                if dados is None:
                    continue
                try:
                    idx_data = np.where(dados['columns'] == 'DATA')[0][0]
                    idx_feat = np.where(dados['columns'] == col_name)[0][0]
                    datas_all = pd.to_datetime(dados['data'][:, idx_data], dayfirst=True)

                    # Agrupar apenas por DATA (sem hora)
                    data_group = datas_all.floor('D').strftime('%d/%m/%Y')

                    df_plot = pd.DataFrame({
                        'DATA': data_group,
                        'DATA_REAL': datas_all,
                        'VALOR': dados['data'][:, idx_feat]
                    })

                    df_plot = df_plot.sort_values('DATA')  # Ordena pelo campo 'DATA'
                    df_result = df_plot.groupby('DATA').apply(lambda x: x.loc[x['DATA_REAL'].idxmax()])

                    # Ordenar pelo índice de DATA_REAL para garantir que as datas mais antigas fiquem à esquerda
                    df_result = df_result.sort_values('DATA_REAL')

                    ax.plot(df_result['DATA_REAL'], df_result['VALOR'], label=f"{label}")

                    # Inclinar os rótulos do eixo X em 60 graus
                    plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
                    fig.tight_layout()

                    # Ajuste de tamanho do gráfico
                    fig.subplots_adjust(bottom=0.15)  # Ajuste a margem inferior para evitar corte de rótulos

                except Exception as e:
                    print(f"Erro plotando {nome_aba} - direção {label}: {e}")

            # ax.set_title(nome_aba)
            ax.set_xlabel("DATA")
            ax.legend()
            ax.grid(True)

            # Inclinar os rótulos do eixo X em 60 graus
            plt.setp(ax.get_xticklabels(), rotation=60, ha='right')
            fig.tight_layout()

            frame = tk.Frame(abas)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            abas.add(frame, text=nome_aba)

    def build_top_tables(self):
        for w in self.top_tables_frame.winfo_children():
            w.destroy()
        data_hour = self.grouped_hours[self.current_date_index]
        for label, dados, flag in zip(['H', 'V', 'A'],
                                      [self.dados_analise_h, self.dados_analise_v, self.dados_analise_a],
                                      [self.direction_h, self.direction_v, self.direction_a]):
            frame_label = tk.LabelFrame(self.top_tables_frame, text=f"Direção {label}")
            frame_label.pack(side='left', fill='both', expand=True, padx=5, pady=2)
            tree = ttk.Treeview(frame_label, columns=('Campo', 'Valor'), show='headings', height=5)
            tree.heading('Campo', text='Campo')
            tree.heading('Valor', text='Valor')
            tree.column('Campo', width=100)
            tree.column('Valor', width=100)
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
                    idx_fs = np.where(cols == 'Fs')[0][0]
                    idx_unidade = np.where(cols == 'Unidade')[0][0]
                    idx_rpm = np.where(cols == 'RPM')[0][0]
                    tree.insert('', 'end', values=("Fs", f"{row[idx_fs]:.1f} Hz"))
                    tree.insert('', 'end', values=("Unidade", row[idx_unidade]))
                    if label == 'H':
                        rpm_all = ("RPM", f"{row[idx_rpm]:.2f} rpm")
                        tree.insert('', 'end', values=rpm_all)
                    else :  
                        tree.insert('', 'end', values=rpm_all)   
                    tree.insert('', 'end', values=("DATA", row[idx_data]))
                except Exception as e:
                    print(f"Erro criando tabela {label}: {e}")
            else:
                tree.insert('', 'end', values=("Status", "Sem aquisição."))

    def on_data_selected(self, event):
        nova_data = self.combo_data.get()
        try:
            indice = self.grouped_hours.index(nova_data)
            self.current_date_index = indice
            self.reload_all()
        except ValueError:
            pass

    def acao_anterior(self):
        if self.current_date_index < len(self.grouped_hours) - 1:
            self.current_date_index += 1
            self.combo_data.current(self.current_date_index)
            self.reload_all()

    def acao_proximo(self):
        if self.current_date_index > 0:
            self.current_date_index -= 1
            self.combo_data.current(self.current_date_index)
            self.reload_all()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    path = r"/home/lav/Leticia/SPYAI/SPYAI_23_07_25/Banco de dados/P100/SI1/EXA/EX0401/105.401/105.401ME/AV01"
    GUI_analise(root, path)
    root.mainloop()