import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
import pandas as pd
import os

BASE_DIR = os.getcwd()
# BASE_DIR = os.path.join(BASE_DIR, 'SPYAI_12_08_25')
BASE_DIR     =  os.path.join(BASE_DIR, 'Banco de dados')
CSV_PATH     = os.path.join(BASE_DIR, "SPYAI_arvore_vibration.csv")

class GUI_cadastro:
    def __init__(self, parent, on_save=None):
        """
        on_save: callback que será chamado após o CSV ser atualizado
        """
        self.parent = parent
        self.on_save = on_save
        self.font_label = ("Times New Roman", 10)
        self.font_entry = ("Times New Roman", 10)
        # carrega árvore de caminhos
        if os.path.isfile(CSV_PATH):
            self.df_arvore = pd.read_csv(CSV_PATH, sep=";")
            paths = self.df_arvore['Caminho ponto'].dropna().unique().tolist()
            # extrai ativos (penúltimo elemento)
            ativos = []
            for p in paths:
                parts = p.split('/')
                # if len(parts) == 1 : 
                #     parts = p.split('/')
                if len(parts) >= 2:
                    ativos.append(parts[-2])
            self.ativos_unicos = sorted(set(ativos))
        else:
            self.df_arvore = pd.DataFrame(columns=['Caminho ponto'])
            self.ativos_unicos = []
        
        style = ttk.Style()
        style.theme_use("default")

        # Estilo do botão
        style.configure(
            "My.TButton",
            background="#34495e",     # cor de fundo
            foreground="white",       # cor do texto
            font=("Arial", 10, "bold"),
            padding=6)
        
        style.map(
            "My.TButton",
            background=[("active", "#2c3e50")],  # cor ao passar o mouse / clicar
            foreground=[("active", "white")])
        
        style = ttk.Style()
        style.theme_use("default")

        # Estilo para Labels com fundo branco
        style.configure("White.TLabel", background="white")

        # Estilo para Checkbuttons com fundo branco
        style.configure("White.TCheckbutton", background="white")

        # Estilo para Radiobutton com fundo branco
        style.configure("White.TRadiobutton", background="white")
                    
        self.create_widgets()

    def create_widgets(self):

        # Estilo para frames brancos
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#2c3e50")
        style.configure("TNotebook.Tab", background="white", foreground="black")
        style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", "black")])
        style.configure("White.TFrame", background="white")  # Fundo branco para as abas

        # Notebook 1
        self.notebook_top = ttk.Notebook(self.parent)
        self.notebook_top.place(relx=0.02, rely=0.01, relwidth=0.96, relheight=0.3)

        self.tab_local = ttk.Frame(self.notebook_top, style="White.TFrame")  # Usa estilo
        self.notebook_top.add(self.tab_local, text="LOCAL/ATIVO")
        self.build_tab_local()

        # Notebook 2
        self.notebook_bottom = ttk.Notebook(self.parent)
        self.notebook_bottom.place(relx=0.02, rely=0.32, relwidth=0.96, relheight=0.68)

        self.tab_ponto = ttk.Frame(self.notebook_bottom, style="White.TFrame")  # Usa estilo
        self.notebook_bottom.add(self.tab_ponto, text="EQUIPAMENTO")
        self.build_tab_ponto()


    def build_tab_local(self):
        frame = self.tab_local
        padding = {'padx': 5, 'pady': 5}
        
        # --- campos local/ativo ---
        # Labels
        ttk.Label(frame, text="EMPRESA:", font=self.font_label, style="White.TLabel").grid(row=0, column=3, sticky="w", **padding)
        self.entry_empresa = ttk.Entry(frame, font=self.font_entry)
        self.entry_empresa.grid(row=0, column=4, **padding)

        ttk.Button(frame, text="Carregar", command=self.salvar_ponto, style="My.TButton").grid(row=0, column=5, sticky="w", **padding)

        ttk.Label(frame, text="UNIDADE:", font=self.font_label, style="White.TLabel").grid(row=1, column=3, sticky="w", **padding)
        self.entry_unidade = ttk.Entry(frame, font=self.font_entry)
        self.entry_unidade.grid(row=1, column=4, **padding)

        ttk.Label(frame, text="SETOR:", font=self.font_label, style="White.TLabel").grid(row=2, column=3, sticky="w", **padding)
        self.entry_setor = ttk.Entry(frame, font=self.font_entry)
        self.entry_setor.grid(row=2, column=4, **padding)

        ttk.Label(frame, text="MÁQUINA:", font=self.font_label, style="White.TLabel").grid(row=3, column=3, sticky="w", **padding)
        self.entry_maquina = ttk.Entry(frame, font=self.font_entry)
        self.entry_maquina.grid(row=3, column=4, **padding)

        ttk.Label(frame, text="EQUIPAMENTO:", font=self.font_label, style="White.TLabel").grid(row=4, column=3, sticky="w", **padding)
        self.entry_equipamento = ttk.Entry(frame, font=self.font_entry)
        self.entry_equipamento.grid(row=4, column=4, **padding)

        ttk.Label(frame, text="NÚMEROS DOS PONTOS:", font=self.font_label, style="White.TLabel").grid(row=4, column=5, sticky="w", **padding)

        # Checkbuttons
        self.num_1, self.num_2, self.num_3, self.num_4 = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        ttk.Checkbutton(frame, text="1", variable=self.num_1, style="White.TCheckbutton").grid(row=4, column=6, **padding)
        ttk.Checkbutton(frame, text="2", variable=self.num_2, style="White.TCheckbutton").grid(row=4, column=7, **padding)
        ttk.Checkbutton(frame, text="3", variable=self.num_3, style="White.TCheckbutton").grid(row=4, column=8, **padding)
        ttk.Checkbutton(frame, text="4", variable=self.num_4, style="White.TCheckbutton").grid(row=4, column=9, **padding)
        # ttk.Label(frame, text="DIREÇÃO:", font=self.font_label).grid(row=6, column=4, sticky="w", **padding)
        # self.h_var, self.v_var, self.a_var = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        # ttk.Checkbutton(frame, text="H",  variable=self.h_var,  style="My.TCheckbutton").grid(row=6, column=5, **padding)
        # ttk.Checkbutton(frame, text="V",  variable=self.v_var,  style="My.TCheckbutton").grid(row=6, column=6, **padding)
        # ttk.Checkbutton(frame, text="A", variable=self.a_var, style="My.TCheckbutton").grid(row=6, column=7, **padding)
        # btn_frame = ttk.Frame(frame)
        # btn_frame.grid(row=4, column=10, columnspan=3, pady=15)
        ttk.Button(frame, text="Salvar", command=self.salvar_local, style="My.TButton").grid(row=5, column=3, **padding)
        ttk.Button(frame, text="Limpar", command=self.limpar_local, style="My.TButton").grid(row=5, column=4, **padding)

    def salvar_local(self):
        empresa     = self.entry_empresa.get().strip()
        unidade     = self.entry_unidade.get().strip()
        setor       = self.entry_setor.get().strip()
        maquina     = self.entry_maquina.get().strip()
        equipamento = self.entry_equipamento.get().strip()
        caixas = []
        if self.num_1.get(): caixas.append("1")
        if self.num_2.get(): caixas.append("2")
        if self.num_3.get(): caixas.append("3")
        if self.num_4.get(): caixas.append("4")
        # caixas = []
        # if self.h_var.get(): caixas.append("H")
        # if self.v_var.get(): caixas.append("V")
        # if self.a_var.get(): caixas.append("A")
        caminhos = []
        if caixas:
            for a in caixas:
                # for c in caixas:
                ultimo = f"{equipamento}|AV0{a}"
                caminhos.append('/'.join([empresa, unidade, setor, maquina, ultimo]))

        df_novas = pd.DataFrame([{"Caminho ponto": p} for p in caminhos])
        df_exist = self.df_arvore if os.path.isfile(CSV_PATH) else pd.DataFrame()
        df_concat = pd.concat([df_exist, df_novas], ignore_index=True)
        df_concat.to_csv(CSV_PATH, index=False, sep=";")

        # atualiza memória local
        self.df_arvore = df_concat
        self.ativos_unicos = sorted({p.split('/')[-2] for p in df_concat['Caminho ponto']})

        # atualiza combo_ativo diretamente
        self.atualizar_ativos()

        if callable(self.on_save): 
            self.on_save()

        mb.showinfo("Confirmação", "ATIVO SALVO")

    def limpar_local(self):
        for w in (self.entry_empresa, self.entry_unidade,
                  self.entry_setor, self.entry_maquina,
                  self.entry_equipamento):
            w.delete(0, tk.END)
        self.num_1.set(False); self.num_2.set(False); self.num_3.set(False); self.num_4.set(False)
        # self.h_var.set(False); self.v_var.set(False); self.a_var.set(False)

    def build_tab_ponto(self):
        frame = self.tab_ponto
        padding = {'padx': 5, 'pady': 5}
        ttk.Label(frame, text="SELECIONE O ATIVO:", font=self.font_label, style="White.TLabel").grid(row=0, column=0, sticky="w", **padding)
        self.combo_ativo = ttk.Combobox(frame, values=self.ativos_unicos, state='normal', font=self.font_entry)
        self.combo_ativo.grid(row=0, column=1, **padding)

        # ROTACAO (RPM)
        ttk.Label(frame, text="Rotação (RPM):", font=self.font_label, style="White.TLabel").grid(row=0, column=4, sticky="w", padx=5, pady=5)

        # Variável para tipo de rotação
        self.rpm_tipo_var = tk.StringVar(value="fixa")

        # Radiobuttons para fixa/variável
        self.rb_rpm_fixa = ttk.Radiobutton(frame, text="Fixa", variable=self.rpm_tipo_var, value="fixa", command=self.atualiza_campos_rpm, style="White.TRadiobutton")
        self.rb_rpm_fixa.grid(row=0, column=5, sticky="w", padx=5, pady=5)

        self.rb_rpm_variavel = ttk.Radiobutton(frame, text="Variável", variable=self.rpm_tipo_var, value="variavel", command=self.atualiza_campos_rpm, style="White.TRadiobutton")
        self.rb_rpm_variavel.grid(row=0, column=6, sticky="w", padx=5, pady=5)

        # self.rb_rpm_fixa = ttk.Radiobutton(frame, text="Fixa", variable=self.rpm_tipo_var, value="fixa", command=self.atualiza_campos_rpm)
        # self.rb_rpm_fixa.grid(row=0, column=5, sticky="w", padx=5, pady=5)

        # self.rb_rpm_variavel = ttk.Radiobutton(frame, text="Variável", variable=self.rpm_tipo_var, value="variavel", command=self.atualiza_campos_rpm)
        # self.rb_rpm_variavel.grid(row=0, column=6, sticky="w", padx=5, pady=5)

        # Entry para RPM fixa
        self.entry_rpm_fixa = ttk.Entry(frame, font=self.font_entry)
        self.entry_rpm_fixa.grid(row=0, column=7, sticky="w", padx=5, pady=5)

        # Entrys para limites RPM variável (inicialmente escondidos)
        self.entry_rpm_lim_inferior = ttk.Entry(frame, font=self.font_entry)
        self.entry_rpm_lim_superior = ttk.Entry(frame, font=self.font_entry)

        # Labels para limites RPM variável
        self.lbl_rpm_lim_inferior = ttk.Label(frame, text="Limite Inferior:", font=self.font_label, style="White.TLabel")
        self.lbl_rpm_lim_superior = ttk.Label(frame, text="Limite Superior:", font=self.font_label, style="White.TLabel")

        # Posiciona os widgets limites, mas esconde
        self.lbl_rpm_lim_inferior.grid(row=0, column=6, sticky="w", padx=5, pady=5)
        self.entry_rpm_lim_inferior.grid(row=0, column=7, sticky="w", padx=5, pady=5)
        self.lbl_rpm_lim_superior.grid(row=0, column=8, sticky="w", padx=5, pady=5)
        self.entry_rpm_lim_superior.grid(row=0, column=9, sticky="w", padx=5, pady=5)

        self.lbl_rpm_lim_inferior.grid_remove()
        self.entry_rpm_lim_inferior.grid_remove()
        self.lbl_rpm_lim_superior.grid_remove()
        self.entry_rpm_lim_superior.grid_remove()

        # Valor de Trip abaixo da rotação
        ttk.Label(frame, text="Valor de Trip:", font=self.font_label, style="White.TLabel").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.entry_valor_trip = ttk.Entry(frame, font=self.font_entry)
        self.entry_valor_trip.grid(row=3, column=1, padx=5, pady=5)

        # Rolamento
        self.rolamento_var = tk.BooleanVar()
        self.chk_rolamento = ttk.Checkbutton(frame, text="Rolamento", variable=self.rolamento_var, command=self.toggle_rolamento, style="White.TCheckbutton")
        self.chk_rolamento.grid(row=4, column=0, sticky="w", padx=5, pady=5)

        self.frame_rolamento = ttk.Frame(frame, style="White.TFrame")
        self.frame_rolamento.grid(row=5, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        self.frame_rolamento.grid_remove()

        ttk.Label(self.frame_rolamento, text="BPFI:", font=self.font_label, style="White.TLabel").grid(row=0, column=0, sticky="w")
        self.entry_bpfi = ttk.Entry(self.frame_rolamento, width=10, font=self.font_entry)
        self.entry_bpfi.grid(row=0, column=1, padx=5)

        ttk.Label(self.frame_rolamento, text="BPFO:", font=self.font_label, style="White.TLabel").grid(row=0, column=2, sticky="w")
        self.entry_bpfo = ttk.Entry(self.frame_rolamento, width=10, font=self.font_entry)
        self.entry_bpfo.grid(row=0, column=3, padx=5)

        ttk.Label(self.frame_rolamento, text="BSF:", font=self.font_label, style="White.TLabel").grid(row=1, column=0, sticky="w")
        self.entry_bsf = ttk.Entry(self.frame_rolamento, width=10, font=self.font_entry)
        self.entry_bsf.grid(row=1, column=1, padx=5)

        ttk.Label(self.frame_rolamento, text="FTF:", font=self.font_label, style="White.TLabel").grid(row=1, column=2, sticky="w")
        self.entry_ftf = ttk.Entry(self.frame_rolamento, width=10, font=self.font_entry)
        self.entry_ftf.grid(row=1, column=3, padx=5)

        # Engrenagem
        self.engrenagem_var = tk.BooleanVar()
        self.chk_engrenagem = ttk.Checkbutton(frame, text="Engrenagem", variable=self.engrenagem_var, command=self.toggle_engrenagem, style="White.TCheckbutton")
        self.chk_engrenagem.grid(row=6, column=0, sticky="w", padx=5, pady=5)

        self.frame_engrenagem = ttk.Frame(frame, style="White.TFrame")
        self.frame_engrenagem.grid(row=7, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        self.frame_engrenagem.grid_remove()

        ttk.Label(self.frame_engrenagem, text="Estágio:", font=self.font_label, style="White.TLabel").grid(row=0, column=0, sticky="w")
        self.entry_estagio = ttk.Entry(self.frame_engrenagem, width=5, font=self.font_entry)
        self.entry_estagio.grid(row=1, column=0, padx=5)

        ttk.Label(self.frame_engrenagem, text="Z - Motora:", font=self.font_label, style="White.TLabel").grid(row=0, column=1, sticky="w")
        self.entry_z_eng_motora = ttk.Entry(self.frame_engrenagem, width=10, font=self.font_entry)
        self.entry_z_eng_motora.grid(row=1, column=1, padx=5)

        ttk.Label(self.frame_engrenagem, text="Z - Motriz:", font=self.font_label, style="White.TLabel").grid(row=0, column=2, sticky="w")
        self.entry_z_eng_motriz = ttk.Entry(self.frame_engrenagem, width=10, font=self.font_entry)
        self.entry_z_eng_motriz.grid(row=1, column=2, padx=5)

        ttk.Label(self.frame_engrenagem, text="RPM Entrada:", font=self.font_label, style="White.TLabel").grid(row=0, column=3, sticky="w")
        self.entry_rpm_entrada = ttk.Entry(self.frame_engrenagem, width=10, font=self.font_entry)
        self.entry_rpm_entrada.grid(row=1, column=3, padx=5)

        ttk.Label(self.frame_engrenagem, text="RPM Saída:", font=self.font_label, style="White.TLabel").grid(row=0, column=4, sticky="w")
        self.entry_rpm_saida = ttk.Entry(self.frame_engrenagem, width=10, font=self.font_entry)
        self.entry_rpm_saida.grid(row=1, column=4, padx=5)

        # Correia
        self.correia_var = tk.BooleanVar()
        self.chk_correia = ttk.Checkbutton(frame, text="Correia", variable=self.correia_var, command=self.toggle_correia, style="White.TCheckbutton")
        self.chk_correia.grid(row=8, column=0, sticky="w", padx=5, pady=5)

        self.frame_correia = ttk.Frame(frame, style="White.TFrame")
        self.frame_correia.grid(row=9, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        self.frame_correia.grid_remove()

        ttk.Label(self.frame_correia, text="BPF:", font=self.font_label, style="White.TLabel").grid(row=0, column=0, sticky="w")
        self.entry_bpf = ttk.Entry(self.frame_correia, width=10, font=self.font_entry)
        self.entry_bpf.grid(row=0, column=1, padx=5)

        # Pás
        self.pas_var = tk.BooleanVar()
        self.chk_pas = ttk.Checkbutton(frame, text="Pás", variable=self.pas_var, command=self.toggle_pas, style="White.TCheckbutton")
        self.chk_pas.grid(row=10, column=0, sticky="w", padx=5, pady=5)

        self.frame_pas = ttk.Frame(frame, style="White.TFrame")
        self.frame_pas.grid(row=11, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        self.frame_pas.grid_remove()

        ttk.Label(self.frame_pas, text="Número de Pás:", font=self.font_label, style="White.TLabel").grid(row=0, column=0, sticky="w")
        self.entry_num_pas = ttk.Entry(self.frame_pas, width=10, font=self.font_entry)
        self.entry_num_pas.grid(row=0, column=1, padx=5)

        # Motor
        self.motor_var = tk.BooleanVar()
        self.chk_motor = ttk.Checkbutton(frame, text="Motor", variable=self.motor_var, command=self.toggle_motor, style="White.TCheckbutton")
        self.chk_motor.grid(row=12, column=0, sticky="w", padx=5, pady=5)

        self.frame_motor = ttk.Frame(frame, style="White.TFrame")
        self.frame_motor.grid(row=13, column=0, columnspan=3, sticky="w", padx=20, pady=5)
        self.frame_motor.grid_remove()

        ttk.Label(self.frame_motor, text="Número de Polos:", font=self.font_label, style="White.TLabel").grid(row=0, column=0, sticky="w")
        self.entry_num_polos = ttk.Entry(self.frame_motor, width=10, font=self.font_entry)
        self.entry_num_polos.grid(row=0, column=1, padx=5)

        ttk.Label(self.frame_motor, text="Número de Barras:", font=self.font_label, style="White.TLabel").grid(row=0, column=2, sticky="w")
        self.entry_num_barras = ttk.Entry(self.frame_motor, width=10, font=self.font_entry)
        self.entry_num_barras.grid(row=0, column=3, padx=5)

        # Botões Salvar/Cancelar ponto
        # btn_frame = ttk.Frame(frame)
        # btn_frame.grid(row=14, column= 0, columnspan=3, pady=20)
        ttk.Button(frame, text="Salvar", command=self.salvar_ponto, style="My.TButton").grid(row=14, column=0, **padding)
        ttk.Button(frame, text="Limpar", command=self.cancelar_ponto, style="My.TButton").grid(row=14, column=1, **padding)

        # Inicializa campos RPM para mostrar só o fixo
        self.atualiza_campos_rpm()

    def atualiza_campos_rpm(self):
        if self.rpm_tipo_var.get() == "fixa":
            # Mostra só o campo fixo
            self.entry_rpm_fixa.grid()
            self.lbl_rpm_lim_inferior.grid_remove()
            self.entry_rpm_lim_inferior.grid_remove()
            self.lbl_rpm_lim_superior.grid_remove()
            self.entry_rpm_lim_superior.grid_remove()
        else:
            # Mostra só os limites
            self.entry_rpm_fixa.grid_remove()
            self.lbl_rpm_lim_inferior.grid()
            self.entry_rpm_lim_inferior.grid()
            self.lbl_rpm_lim_superior.grid()
            self.entry_rpm_lim_superior.grid()

    # Toggle frames
    def toggle_rolamento(self):
        if self.rolamento_var.get():
            self.frame_rolamento.grid()
        else:
            self.frame_rolamento.grid_remove()

    def toggle_engrenagem(self):
        if self.engrenagem_var.get():
            self.frame_engrenagem.grid()
        else:
            self.frame_engrenagem.grid_remove()

    def toggle_correia(self):
        if self.correia_var.get():
            self.frame_correia.grid()
        else:
            self.frame_correia.grid_remove()

    def toggle_pas(self):
        if self.pas_var.get():
            self.frame_pas.grid()
        else:
            self.frame_pas.grid_remove()

    def toggle_motor(self):
        if self.motor_var.get():
            self.frame_motor.grid()
        else:
            self.frame_motor.grid_remove()

    # Salvamento PONTO
    def salvar_ponto(self):
        # Primeiro, pega o ativo selecionado
        ativo = self.combo_ativo.get().strip() or None

        # encontra um caminho completo que contenha esse ativo
        df_arvore = pd.read_csv(CSV_PATH)
        mask = df_arvore['Caminho ponto']\
            .apply(lambda x: x.split('/')[-2] == ativo)
        if mask.any():
            full_path = df_arvore.loc[mask, 'Caminho ponto'].iloc[0]
            asset_dir = full_path.replace("/",os.path.sep)[:-5]
            # asset_dir = '/'.join(full_path.rsplit('/', 1)[:-1])
        else:
            asset_dir = os.getcwd()  # fallback ou exibir erro

        # Campos comuns
        nome = ativo
        valor_trip = self.entry_valor_trip.get().strip() or None

        # Rotação RPM
        rpm_tipo = self.rpm_tipo_var.get()
        if rpm_tipo == "fixa":
            rpm = self.entry_rpm_fixa.get().strip() or None
            rpm_lim_inf = None
            rpm_lim_sup = None
        else:
            rpm = None
            rpm_lim_inf = self.entry_rpm_lim_inferior.get().strip() or None
            rpm_lim_sup = self.entry_rpm_lim_superior.get().strip() or None

        # Rolamento
        if self.rolamento_var.get():
            bpfi = self.entry_bpfi.get().strip() or None
            bpfo = self.entry_bpfo.get().strip() or None
            bsf = self.entry_bsf.get().strip() or None
            ftf = self.entry_ftf.get().strip() or None
        else:
            bpfi = bpfo = bsf = ftf = None

        # Engrenagem
        if self.engrenagem_var.get():
            estagio = self.entry_estagio.get().strip() or None
            z_motora = self.entry_z_eng_motora.get().strip() or None
            z_motriz = self.entry_z_eng_motriz.get().strip() or None
            rpm_entrada = self.entry_rpm_entrada.get().strip() or None
            rpm_saida = self.entry_rpm_saida.get().strip() or None
        else:
            estagio = z_motora = z_motriz = rpm_entrada = rpm_saida = None

        # Correia
        if self.correia_var.get():
            bpf = self.entry_bpf.get().strip() or None
        else:
            bpf = None

        # Pás
        if self.pas_var.get():
            num_pas = self.entry_num_pas.get().strip() or None
        else:
            num_pas = None

        # Motor
        if self.motor_var.get():
            num_polos = self.entry_num_polos.get().strip() or None
            num_barras = self.entry_num_barras.get().strip() or None
        else:
            num_polos = None
            num_barras = None

        dados = {
            "PONTO": nome,
            # "H": self.h_var.get(),  # True se marcado, False se não
            # "V": self.v_var.get(),
            # "A": self.a_var.get(),
            "rpm_tipo": rpm_tipo,
            "rpm": rpm,
            "rpm_lim_inferior": rpm_lim_inf,
            "rpm_lim_superior": rpm_lim_sup,
            "valor_trip": valor_trip,
            "bpfi": bpfi,
            "bpfo": bpfo,
            "bsf": bsf,
            "ftf": ftf,
            "estagio": estagio,
            "z_motora": z_motora,
            "z_motriz": z_motriz,
            "rpm_entrada": rpm_entrada,
            "rpm_saida": rpm_saida,
            "bpf": bpf,
            "num_pas": num_pas,
            "num_polos": num_polos,
            "num_barras": num_barras
        }

        # grava na pasta do ativo
        asset_dir = os.path.join(BASE_DIR, asset_dir)
        self.append_to_csv(dados, directory=asset_dir)

        mb.showinfo("Confirmação", "DADOS DO EQUIPAMENTO SALVOS")

    def append_to_csv(self, data, directory, sep=";"):
        os.makedirs(directory, exist_ok=True)
        filename = os.path.join(directory, "SPYAI_cadastro.csv")
        df_new = pd.DataFrame([data])

        if os.path.isfile(filename):
            df_existing = pd.read_csv(filename, sep=sep)
            df_concat = pd.concat([df_existing, df_new], ignore_index=True)
            df_concat.to_csv(filename, index=False, sep=sep)
        else:
            df_new.to_csv(filename, index=False, sep=sep)
            print(f"PONTO salvo in {directory}.")


    def cancelar_ponto(self):
        self.entry_rpm_fixa.delete(0, tk.END)
        self.entry_rpm_lim_inferior.delete(0, tk.END)
        self.entry_rpm_lim_superior.delete(0, tk.END)
        self.entry_valor_trip.delete(0, tk.END)

        self.rolamento_var.set(False)
        self.toggle_rolamento()
        self.entry_bpfi.delete(0, tk.END)
        self.entry_bpfo.delete(0, tk.END)
        self.entry_bsf.delete(0, tk.END)
        self.entry_ftf.delete(0, tk.END)

        self.engrenagem_var.set(False)
        self.toggle_engrenagem()
        self.entry_estagio.delete(0, tk.END)
        self.entry_z_eng_motora.delete(0, tk.END)
        self.entry_z_eng_motriz.delete(0, tk.END)
        self.entry_rpm_entrada.delete(0, tk.END)
        self.entry_rpm_saida.delete(0, tk.END)

        self.correia_var.set(False)
        self.toggle_correia()

        self.pas_var.set(False)
        self.toggle_pas()
        self.entry_num_pas.delete(0, tk.END)

        self.motor_var.set(False)
        self.toggle_motor()
        self.entry_num_polos.delete(0, tk.END)

        print("Campos PONTO limpos.")

    def atualizar_ativos(self):
        if os.path.isfile(CSV_PATH):
            self.df_arvore = pd.read_csv(CSV_PATH, sep=";")
            paths = self.df_arvore['Caminho ponto'].dropna().unique().tolist()
            ativos = []
            for p in paths:
                parts = p.split('/')
                if len(parts) >= 2:
                    ativos.append(parts[-2])
            self.ativos_unicos = sorted(set(ativos))
            self.combo_ativo['values'] = self.ativos_unicos


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Cadastro SPYAI")
    gui = GUI_cadastro(root)

    # Aplica estilo para botões e checkbuttons com fontes maiores
    style = ttk.Style()
    style.configure("My.TButton", font=("Times New Roman", 10))
    style.configure("My.TCheckbutton", font=("Times New Roman", 10))

    root.mainloop()