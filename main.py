import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import subprocess
import tkinter.messagebox as mb
from platform import system

# Caminho para a imagem (altere para o caminho da sua imagem)
path = os.getcwd()
path_version = os.path.join(path,'SPYAI_01_08_25')
path_venv = os.path.join(os.path.dirname(path_version), 'SPYAI_venv')
path_imagem = os.path.join('Banco de dados', 'logo.png')
#path_imagem = os.path.join('BD', 'logo.png')
CAMINHO_IMAGEM = os.path.join(path_version, path_imagem)

# Configurações da interface
BG_COLOR = "#2c3e50"
FONT_TITULO = ("Arial", 18, "bold")
FONT_BOTAO = ("Arial", 16, "bold")
BOTAO_WIDTH = 25
BOTAO_HEIGHT = 2

def abrir_vibration():
    root.destroy()
    try:
        if system() == 'Windows' : 
            py_windows = os.path.join('Scripts', 'python.exe')
            path_exe = os.path.join(path_venv, py_windows)
        else:
            py_linux = os.path.join('bin', 'python')
            path_exe = os.path.join(path_venv, py_linux)
        path_vibration = os.path.join(path_version, 'main_vibration.py')
        subprocess.run([path_exe, path_vibration], check=True)
    except Exception as e:
        mb.showerror("Erro", "Erro ao abrir o módulo de vibração")

def abrir_processo():
    root.destroy()
    try:
        if system() == 'Windows' : 
            py_windows = os.path.join('Scripts', 'python.exe')
            path_exe = os.path.join(path_venv, py_windows)
        else:
            py_linux = os.path.join('bin', 'python')
            path_exe = os.path.join(path_venv, py_linux)
        path_process = os.path.join(path_version, 'main_process.py')
        subprocess.run([path_exe, path_process], check=True)
    except Exception as e:
        mb.showerror("Erro", "Erro ao abrir o módulo de processo")

def GUI():
    global root
    root = tk.Tk()
    root.title("spyAI")
    root.configure(bg=BG_COLOR)

    # Ícone da janela
    try:
        logo_icon = tk.PhotoImage(file=CAMINHO_IMAGEM)
        root.iconphoto(True, logo_icon)
    except Exception as e:
        print("Erro ao carregar ícone:", e)

    # Centralizar a janela
    largura = 500
    altura = 600
    x = (root.winfo_screenwidth() - largura) // 2
    y = (root.winfo_screenheight() - altura) // 2
    root.geometry(f"{largura}x{altura}+{x}+{y}")
    root.resizable(False, False)

    # Frame central
    frame_central = tk.Frame(root, bg=BG_COLOR)
    frame_central.pack(expand=True)

    # Imagem
    try:
        imagem = Image.open(CAMINHO_IMAGEM)
        imagem = imagem.resize((200, 200), Image.Resampling.LANCZOS)
        imagem_tk = ImageTk.PhotoImage(imagem)
        label_imagem = tk.Label(frame_central, image=imagem_tk, bg=BG_COLOR)
        label_imagem.image = imagem_tk 
        label_imagem.pack(pady=(20, 10))
    except Exception as e:
        label_erro = tk.Label(frame_central, text="Imagem não encontrada", fg="white", bg=BG_COLOR)
        label_erro.pack(pady=(20, 10))

    # Texto spyAI
    label_titulo = tk.Label(frame_central, text="spyAI", font=FONT_TITULO, fg="white", bg=BG_COLOR)
    label_titulo.pack(pady=(0, 30))

    # Botões
    btn_vibration = tk.Button(frame_central, text="SPYAI Vibration", font=FONT_BOTAO,
                              width=BOTAO_WIDTH, height=BOTAO_HEIGHT, bg="#34495e", fg="white",
                              activebackground="#1abc9c", relief="raised", command=abrir_vibration)
    btn_vibration.pack(pady=10)

    btn_process = tk.Button(frame_central, text="SPYAI Process", font=FONT_BOTAO,
                            width=BOTAO_WIDTH, height=BOTAO_HEIGHT, bg="#34495e", fg="white",
                            activebackground="#1abc9c", relief="raised", command=abrir_processo)
    btn_process.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    GUI()
