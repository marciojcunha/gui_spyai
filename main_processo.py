import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import tkinter.messagebox as mb
from dynamprocesso import Processo

# Obtém o diretório onde o script está sendo executado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(BASE_DIR, "Banco de dados")

# Construa o caminho completo para o arquivo CSV
CSV_PATH = os.path.join(BASE_DIR, "SPYAI_arvore_processo.csv")

class GUI_processo:
    def __init__(self, root):
        self.root = root
        self.root.title("SPYAI - Sistema de Monitoramento")
        try:
            root.attributes('fullscreen', True)
        except:
            root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))

        if os.path.isfile(CSV_PATH):
            df = pd.read_csv(CSV_PATH, sep=r'[;,]')
        else:
            df = pd.DataFrame(columns=["Caminho ponto"])
        self.df = df

        self.paths = [p.split( os.path.sep) for p in df["Caminho ponto"].unique()]
        self.current_path = None
        self.selected_button = None

        self.sidebar_width = 0.20
        self.left_expanded = True
        self.right_expanded = True

        self.status_frame = tk.Frame(self.root, bg="#2c3e50")
        self.status_frame.place(relx=0, rely=0, relwidth=1, relheight=0.10)

        # self.spyai_label = tk.Label(self.status_frame, text="spyAI", fg="white", bg="#2c3e50", font=("Arial", 20, "bold"))
        # self.spyai_label.pack(padx=20, pady=20)

        self.create_left_panel()
        self.create_right_sidebar()
        self.create_main_area()
        self.create_toggle_buttons()

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
                full = current + "\\" + part
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
        parts = self.get_full_path(item).split( os.path.sep)
        PROCESSO_DIR = os.path.join(BASE_DIR, parts[0])
        if os.path.exists(PROCESSO_DIR):
            files = os.listdir(PROCESSO_DIR)
            found = next((f for f in files if parts[1].lower() in f and f.endswith('.csv')), None)
        if found:
            df_processo = pd.read_csv(os.path.join(PROCESSO_DIR, found))
        else:
            mb.showerror("Erro", f"Pasta não encontrada:\n{PROCESSO_DIR}")

        self.calc_equipamento(df_processo)

    def calc_equipamento(self, dataframe):
        processo = Processo(dataframe)
        processo.preprocessamento() # dashboards
        # selecionar o período de análise
        # processo.anomalia_global()
        # # selecionar os targets
        # processo.anomalia_local()
        # processo.rank()
        # processo.correlation()
        # processo.estatistica()
     
    def reload_tree(self):
        self.df = pd.read_csv(CSV_PATH, sep=r'[;,]')
        self.paths = [p.split( os.path.sep) for p in self.df["Caminho ponto"].unique()]
        self.build_tree(self.paths)

    def get_full_path(self, item):
        parts = []
        while item:
            parts.insert(0, self.tree.item(item)["text"])
            item = self.tree.parent(item)
        return "\\".join(parts)

    def create_right_sidebar(self):
        self.right_sidebar = tk.Frame(self.root, bg="#2c3e50")
        self.right_sidebar.place(relx=1 - self.sidebar_width, rely=0.10,
                                 relwidth=self.sidebar_width, relheight=0.90)

        tk.Label(self.right_sidebar, text="MENU", fg="white", bg="#2c3e50",
                 font=("Arial", 14, "bold")).pack(pady=40)

        self.menu_buttons = {}
        for label, cmd in [
            ("DASHBOARD", self.open_dashboard),
            ("ANÁLISE GLOBAL", self.open_analise_global),
            ("ANÁLISE LOCAL", self.open_analise_local),
            ("ANÁLISE ESTATÍSTICA", self.open_analise_estatistica),
            ("RANKING", self.open_rank),
            ("CORRELAÇÃO", self.open_correlation)
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

    def open_analise_global(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ativo para análise global")
            return
        self.clear_main()
        self.status_label.config(text="Análise Global")
        self.highlight_button("ANÁLISE GLOBAL")

    def open_analise_local(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ativo para análise local")
            return
        self.clear_main()
        self.status_label.config(text="Análise Local")
        self.highlight_button("ANÁLISE LOCAL")

    def open_rank(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ativo para gerar o RUL")
            return
        self.clear_main()
        self.status_label.config(text="RANKING")
        self.highlight_button("RANKING")
    
    def open_correlation(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ativo para gerar o RUL")
            return
        self.clear_main()
        self.status_label.config(text="Correlação")
        self.highlight_button("CORRELAÇÃO")

    def open_dashboard(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ativo para gerar o RUL")
            return
        self.clear_main()
        self.status_label.config(text="Dashboard")
        self.highlight_button("DASHBOARD")
    
    def open_analise_estatistica(self):
        if not self.current_path:
            mb.showwarning("Aviso", "Selecione antes um ativo para gerar o RUL")
            return
        self.clear_main()
        self.status_label.config(text="Análise Estatística")
        self.highlight_button("ANÁLISE ESTATÍSTICA")
    
if __name__ == "__main__":
    root = tk.Tk()
    app = GUI_processo(root)
    root.mainloop()

