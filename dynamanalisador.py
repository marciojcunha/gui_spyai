import numpy as np
import pandas as pd
import os
from dynamfeatures import Features_classification
import dynamsignal as ds

class Analisador:
    def __init__(self, path, dados_numpy=None):
        self.path = path
        self.npz_path = os.path.join(self.path, "SPYAI_analise.npz")
        dir = os.path.dirname(self.path)
        self.cadastro_path = os.path.join(os.path.dirname(dir), "SPYAI_cadastro.csv")
        self.dados_cadastro = pd.read_csv(self.cadastro_path, sep=';|,', engine='python')

        if dados_numpy is not None:
            self.dados_numpy = dados_numpy
        else:
            files = os.listdir(self.path)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found is None:
                raise FileNotFoundError("Arquivo Export_Time*.npz não encontrado")
            self.dados_numpy = np.load(os.path.join(path, found), allow_pickle=True)

        arr = self.dados_numpy['data']
        cols = self.dados_numpy['columns']

        #cols_to_delete = [3, 4, 7, 8, 9]
        cols_to_delete = []
        self.arr = np.delete(arr, cols_to_delete, axis=1)
        self.cols = np.delete(cols, cols_to_delete)

        self.idx_caminho = np.where(self.cols == 'Caminho de ponto')[0][0]
        self.idx_dts = np.where(self.cols == 'DTS')[0][0]
        self.idx_unidade = np.where(self.cols == 'Unidade')[0][0]
        self.idx_amostras = np.where(self.cols == 'Amostras')[0][0]
        self.idx_tempo_max = np.where(self.cols == 'Tempo máx.')[0][0]
        self.idx_dado = np.where(self.cols == 'Dado')[0][0]

        self.dados_cadastro = pd.DataFrame(self.dados_cadastro)
        self.features_test = None
        self.valor_trip = self.dados_cadastro['valor_trip'].values
        self.bpfi = self.dados_cadastro['bpfi'].values
        self.bpfo = self.dados_cadastro['bpfo'].values
        self.bsf = self.dados_cadastro['bsf'].values
        self.ftf = self.dados_cadastro['ftf'].values
        self.estagio = self.dados_cadastro['estagio'].values
        self.z_motora = self.dados_cadastro['z_motora'].values
        self.z_motriz = self.dados_cadastro['z_motriz'].values
        self.rpm_entrada = self.dados_cadastro['rpm_entrada'].values
        self.rpm_saida = self.dados_cadastro['rpm_saida'].values
        self.bpf = self.dados_cadastro['bpf'].values
        self.num_pas = self.dados_cadastro['num_pas'].values
        self.num_polos = self.dados_cadastro['num_polos'].values
        self.num_barras = self.dados_cadastro['num_barras'].values

    def analisar(self, data='all'):
        if data == 'all':
            indices = range(len(self.arr))
        else:
            if isinstance(data, str) or isinstance(data, np.str_):
                data = [data]
            data = set(data)
            indices = [i for i, linha in enumerate(self.arr) if linha[self.idx_dts] in data]

        registros = []

        for idx in indices:
            linha = self.arr[idx]

            caminho_ponto = linha[self.idx_caminho]
            dts_str = linha[self.idx_dts]
            unidade = linha[self.idx_unidade]
            amostras = int(linha[self.idx_amostras])
            tempo_max = linha[self.idx_tempo_max]

            sinal_raw = linha[self.idx_dado : self.idx_dado + amostras].astype(float)
            sinal = sinal_raw[~np.isnan(sinal_raw)]

            dt = tempo_max / amostras
            fs = 1.0 / dt
            intervalo = [int(self.dados_cadastro['rpm'].iloc[0] - 100), int(self.dados_cadastro['rpm'].iloc[0] + 100)]
            rpm = ds.utilities.rpm_estimation(sinal, dt, intervalo)

            tag, ponto = self.extrair_tag_ponto(caminho_ponto)

            feat = Features_classification(fs, rpm,
                        rot_ang= np.random.uniform(0,2*np.pi),
                        num_dentes_pinhao=self.z_motora,
                        num_dentes_coroa=self.z_motriz,
                        freq_rotac_pinhao=rpm / 60,
                        num_pas=self.num_pas,
                        num_polos=self.num_polos,
                        num_barras_rotor=self.num_barras,
                        bpfi=self.bpfi,
                        bpfo=self.bpfo,
                        bsf=self.bsf,
                        ftf=self.ftf,
                        bpf=self.bpf)
            feat.calcular_features(sinal, normalized=False)
            feats = feat.features

            registro = {
                'TAG': tag,
                'Ponto': ponto,
                'DATA': dts_str,
                'dt': dt,
                'Fs': fs,
                'Unidade': unidade,
                'RPM': rpm,
                **feats
            }

            registros.append(registro)

        if not registros:
            print("Nenhum registro para processar.")

        # Atualiza dados anteriores, se existirem
        if os.path.exists(self.npz_path):
            self.df_analise = np.load(self.npz_path, allow_pickle=True)
            self.dados_analise = {k: self.df_analise[k] for k in self.df_analise.files}
        else:
            self.dados_analise = None

        dados_df = pd.DataFrame(registros)
        path_csv = self.npz_path.replace('.npz', '.csv')
        dados_df.to_csv(path_csv, index=False)
        np.savez(self.npz_path, data=dados_df.values, columns=dados_df.columns)

        print(f"Análise concluída: {len(registros)} registros processados e salvos em {self.npz_path}")

    def extrair_tag_ponto(self, caminho):
        partes = caminho.strip( os.path.sep).split( os.path.sep)
        tag = partes[-2] if len(partes) >= 2 else 'Desconhecido'
        ponto = partes[-1] if len(partes) >= 1 else 'Desconhecido'
        return tag, ponto