import pandas as pd
from dynamponto import Ponto
import os

df = pd.read_csv('C:/Users/letic/OneDrive/Documentos/SPYAI/Banco de dados/SPYAI_arvore_vibration.csv')

points = df["Caminho ponto"]
pontos = []
for i in range(len(points)):
    pontos.append(points[i].replace("|", os.path.sep))

for ponto in pontos:
    ponto = Ponto(ponto)
    ponto.analise_ponto()
    ponto.classification_ponto() 
    ponto.rul_ponto()
