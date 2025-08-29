import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import tkinter.messagebox as mb
from gui_analise import GUI_analise
from gui_cadastro import GUI_cadastro
from gui_classification import GUI_classification
from gui_rul import GUI_rul
from dynamponto import Ponto
import threading
import tkinter.messagebox as mb

# Obtém o diretório onde o script está sendo executado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "Banco de dados")
#BASE_DIR = os.path.join(BASE_DIR, "BD")

# Construa o caminho completo para o arquivo CSV
CSV_PATH = os.path.join(BASE_DIR, "SPYAI_arvore_vibration.csv")
CAMINHO_IMAGEM = os.path.join(BASE_DIR, "logo.png")

class GUI_vibration:
    def __init__(self, root):
        super().__init__()  # Inicializa a classe base Tk
        self.root = root
        self.root.title("SPYAI - Sistema de Monitoramento")
        try:
            root.attributes('fullscreen', True)
        except:
            root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

            # Ícone da janela
        try:
            logo_icon = tk.PhotoImage(file=CAMINHO_IMAGEM)
            root.iconphoto(True, logo_icon)
        except Exception as e:
            print("Erro ao carregar ícone:", e)

        if os.path.isfile(CSV_PATH):
            df = pd.read_csv(CSV_PATH, sep=r';', engine = 'python')
        else:
            df = pd.DataFrame(columns=["Caminho ponto"])
        self.df = df

        self.paths = [p.split("|") for p in df["Caminho ponto"].unique()]
        self.current_path = None
        self.selected_button = None

        self.sidebar_width = 0.20
        self.left_expanded = True
        self.right_expanded = True

        self.status_frame = tk.Frame(self.root, bg="#2c3e50")
        self.status_frame.place(relx=0, rely=0, relwidth=1, relheight=0.10)

        self.status_label = tk.Label(self.status_frame, text="Cadastro", fg="white", bg="#2c3e50", font=("Arial", 24, "bold"))
        self.status_label.pack(side="left", padx=20, pady=20)

        # self.spyai_label = tk.Label(self.status_frame, text="spyAI", fg="white", bg="#2c3e50", font=("Arial", 20, "bold"))
        # self.spyai_label.pack(padx=20, pady=20)

        self.create_left_panel()
        self.create_right_sidebar()
        self.create_main_area()
        self.create_toggle_buttons()
        self.open_cadastro()

    def create_left_panel(self):
        self.left_panel = tk.Frame(self.root, bg="#2c3e50")
        self.left_panel.place(relx=0, rely=0.10, relwidth=self.sidebar_width, relheight=0.90)

        tk.Label(self.left_panel, text="ATIVOS", font=("Arial", 14, "bold"), fg="white", bg="#2c3e50").pack(pady=40)

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(self.left_panel, textvariable=self.search_var)
        search_entry.pack(padx=10, pady=10, fill="x")
        search_entry.bind("<KeyRelease>", self.search_tree)

        self.tree = ttk.Treeview(self.left_panel)
        self.tree.pack(padx=10, pady=10, expand=True, fill="both")
        self.tree.bind("<Double-1>", self.on_tree_select)

        style = ttk.Style()
        style.map("Treeview", background=[("selected", "#fff9c4")], foreground=[("selected", "black")])

        self.build_tree(self.paths)

    def build_tree(self, path_list):
        self.tree.delete(*self.tree.get_children())
        node_dict = {}
        for path in path_list:
            current = ""
            for part in path:
                parent = node_dict.get(current, "")
                full = current + os.path.sep + part
                if full not in node_dict:
                    node = self.tree.insert(parent, "end", text=part)
                    node_dict[full] = node
                current = full
        for node in node_dict.values():
            self.tree.item(node, open=True)

    def search_tree(self, event=None):
        kw = self.search_var.get().lower()
        filtered = [p for p in self.paths if any(kw in part.lower() for part in p)]
        self.build_tree(filtered)

    def on_tree_select(self, event):
        item = self.tree.focus()
        parts = self.get_full_path(item).split("|")
        target_dir = os.path.join(BASE_DIR, *parts)

        if not os.path.isdir(target_dir):
            mb.showerror("Erro", f"Pasta não encontrada:\n{target_dir}")
            return

        self.current_path = target_dir
        newWindow = tk.Toplevel(self.root)
        width = 750
        height = 250
        self.center_window(newWindow, width, height)
        # newWindow.geometry('100x+50+50') 
        # newWindow.title("   " )
        frame_graphic = tk.Frame( newWindow ) # Janela Principal
        frame_graphic.grid(row=0,column=0,padx=5, pady=5)            
        # Em processamentoH
        label_frame1=tk.LabelFrame( frame_graphic )
        label_frame1.grid(row=0,column=0,padx=(50,50), pady=(50,50),sticky='w') 
        txt = " Por favor, aguarde: IA em Treinamento"  
        label=tk.Label(label_frame1,text= txt , font=("courier", 20, "bold" )  )
        label.grid(row=0,column=0,padx=(20,20), pady=(50,50),sticky='w')
        frame_graphic.update()      
        self.calc_ponto(target_dir)
        #self.root.wait_window( newWindow )
        newWindow.destroy()         
        self.open_analise()
    
    # Centraliza as janelas
    def center_window(self, window, width, height): 
        """Centraliza a janela na tela."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')

    def calc_ponto(self, path):
        ponto = Ponto(path)
        ponto.analise_ponto()
        ponto.classification_ponto() 
        ponto.rul_ponto()
        
    def reload_tree(self):
        self.df = pd.read_csv(CSV_PATH, sep=r';')
        self.paths = [p.split("|") for p in self.df["Caminho ponto"].unique()]
        self.build_tree(self.paths)

    def get_full_path(self, item):
        parts = []
        while item:
            parts.insert(0, self.tree.item(item)["text"])
            item = self.tree.parent(item)
        return "|".join(parts)

    def create_right_sidebar(self):
        self.right_sidebar = tk.Frame(self.root, bg="#2c3e50")
        self.right_sidebar.place(relx=1 - self.sidebar_width, rely=0.10,
                                 relwidth=self.sidebar_width, relheight=0.90)

        tk.Label(self.right_sidebar, text="MENU", fg="white", bg="#2c3e50",
                 font=("Arial", 14, "bold")).pack(pady=40)

        self.menu_buttons = {}
        for label, cmd in [
            ("ANÁLISE",       self.open_analise),
            ("CLASSIFICAÇÃO", self.open_classification),
            ("RUL",           self.open_rul),
            ("CADASTRO",      self.open_cadastro),
        ]:
            btn = tk.Button(self.right_sidebar, text=label,
                            bg="#34495e", fg="white", font=("Arial", 10),
                            bd=1, relief="solid",
                            activebackground="#1abc9c", activeforeground="black",
                            command=cmd)
            btn.pack(fill="x", pady=5)
            self.menu_buttons[label] = btn

    def highlight_button(self, label):
        for lbl, btn in self.menu_buttons.items():
            if lbl == label:
                btn.config(bg="#fff9c4", fg="black")
            else:
                btn.config(bg="#34495e", fg="white")

    def create_main_area(self):
        self.main_frame = tk.Frame(self.root, bg="white")
        self.update_main_frame_position()

    def create_toggle_buttons(self):
        self.left_toggle_btn = tk.Button(self.root, text="<<", bg="#1abc9c", fg="white",
                                         font=("Arial", 10, "bold"), command=self.toggle_left_sidebar)
        self.left_toggle_btn.place(relx=self.sidebar_width - 0.03, rely=0.11, width=30, height=30)

        self.right_toggle_btn = tk.Button(self.root, text=">>", bg="#1abc9c", fg="white",
                                          font=("Arial", 10, "bold"), command=self.toggle_right_sidebar)
        self.right_toggle_btn.place(relx=1 - self.sidebar_width, rely=0.11, width=30, height=30)

    def toggle_left_sidebar(self):
        if self.left_expanded:
            self.left_panel.place_configure(relwidth=0)
            self.left_toggle_btn.place(relx=0, rely=0.11)
            self.left_toggle_btn.configure(text=">>")
        else:
            self.left_panel.place_configure(relwidth=self.sidebar_width)
            self.left_toggle_btn.place(relx=self.sidebar_width - 0.03, rely=0.11)
            self.left_toggle_btn.configure(text="<<")
        self.left_expanded = not self.left_expanded
        self.update_main_frame_position()

    def toggle_right_sidebar(self):
        if self.right_expanded:
            self.right_sidebar.place_configure(relwidth=0)
            self.right_toggle_btn.place(relx=0.97, rely=0.11)
            self.right_toggle_btn.configure(text="<<")
        else:
            self.right_sidebar.place_configure(relwidth=self.sidebar_width)
            self.right_toggle_btn.place(relx=1 - self.sidebar_width, rely=0.11)
            self.right_toggle_btn.configure(text=">>")
        self.right_expanded = not self.right_expanded
        self.update_main_frame_position()

    def update_main_frame_position(self):
        lw = self.sidebar_width if self.left_expanded else 0
        rw = self.sidebar_width if self.right_expanded else 0
        self.main_frame.place(relx=lw, rely=0.10, relwidth=1 - lw - rw, relheight=0.90)

    def clear_main(self):
        for w in self.main_frame.winfo_children():
            w.destroy()

    def open_analise(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ponto para análise")
            return
        self.clear_main()
        self.status_label.config(text="Análise")
        self.highlight_button("ANÁLISE")
        GUI_analise(self.main_frame, path=self.current_path)


    def open_classification(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ponto para classificação")
            return
        self.clear_main()
        self.status_label.config(text="Classificação")
        self.highlight_button("CLASSIFICAÇÃO")
        GUI_classification(self.main_frame, path=self.current_path)


    def open_cadastro(self):
        self.clear_main()
        self.status_label.config(text="Cadastro")
        self.highlight_button("CADASTRO")
        GUI_cadastro(self.main_frame, on_save=self.reload_tree)

    def open_rul(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ponto para gerar o RUL")
            return
        self.clear_main()
        self.status_label.config(text="RUL")
        self.highlight_button("RUL")
        GUI_rul(self.main_frame, path=self.current_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI_vibration(root)
    root.mainloop()

