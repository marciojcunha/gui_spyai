import pandas as pd
import numpy as np
from scipy.stats import zscore

class Processo:
    def __init__(self, dados_processo):
        self.dados_processo = dados_processo

    def preprocessamento(self):
        df = self.dados_processo.copy()

        # 1. Remover colunas duplicadas (mantém a primeira ocorrência)
        colunas_antes = df.shape[1]
        df = df.loc[:, ~df.columns.duplicated()]
        colunas_apos = df.shape[1]
        qtd_colunas_duplicadas = colunas_antes - colunas_apos

        # 2. Remover primeira coluna (índice 0)
        primeira_coluna = df.columns[0]
        df_sem_primeira = df.drop(columns=primeira_coluna)

        # 3. Contar e remover linhas com valores NaN ou 0
        qtd_nan_zero_por_coluna = df_sem_primeira.isna().sum() + (df_sem_primeira == 0).sum()
        df_sem_primeira = df_sem_primeira.dropna()
        df_sem_primeira = df_sem_primeira[(df_sem_primeira != 0).all(axis=1)]

        # 4. Separar colunas numéricas e categóricas
        col_data = 'data' if 'data' in df_sem_primeira.columns else None
        dados_processo_numerico = df_sem_primeira.select_dtypes(include=[np.number]).copy()

        # Adicionar a coluna 'data' aos dados numéricos se existir
        if col_data:
            dados_processo_numerico[col_data] = df_sem_primeira[col_data]

        dados_processo_categorico = df_sem_primeira.select_dtypes(exclude=[np.number]).copy()
        if col_data:
            dados_processo_categorico = dados_processo_categorico.drop(columns=[col_data], errors='ignore')

        qtd_col_numericas = dados_processo_numerico.shape[1]
        qtd_col_categoricas = dados_processo_categorico.shape[1]

        # 5. Detectar e remover outliers nas colunas numéricas (Z-score > 3), exceto 'data'
        outliers_por_coluna = {}
        cols_to_check = [col for col in dados_processo_numerico.columns if col != 'data']
        mask_outliers = pd.Series([False] * dados_processo_numerico.shape[0], index=dados_processo_numerico.index)

        for col in cols_to_check:
            z_scores = np.abs(zscore(dados_processo_numerico[col]))
            outliers = z_scores > 3
            outliers_por_coluna[col] = outliers.sum()
            mask_outliers = mask_outliers | outliers

        dados_processo_numerico = dados_processo_numerico[~mask_outliers]
        dados_processo_categorico = dados_processo_categorico.loc[dados_processo_numerico.index]

        # 6. Verificar quantos parâmetros existem após a coluna "data" para cada data
        if col_data:
            df_data = df_sem_primeira.copy()
            df_data[col_data] = pd.to_datetime(df_data[col_data], errors='coerce')
            df_data = df_data.dropna(subset=[col_data])
            parametros_por_data = df_data.groupby(col_data).apply(lambda x: x.shape[1] - 1).reset_index()
            parametros_por_data.columns = ['data', 'qtd_parametros']
        else:
            parametros_por_data = pd.DataFrame(columns=['data', 'qtd_parametros'])

        # 7. Montar resumo em DataFrame
        resumo = {
            'Colunas duplicadas removidas': [qtd_colunas_duplicadas],
            'Total colunas numéricas': [qtd_col_numericas],
            'Total colunas categóricas': [qtd_col_categoricas],
        }

        for col, qtd in qtd_nan_zero_por_coluna.items():
            resumo[f'Nan/Zero em {col}'] = [qtd]
        for col, qtd in outliers_por_coluna.items():
            resumo[f'Outliers em {col}'] = [qtd]

        resumo_df = pd.DataFrame(resumo)

        return resumo_df, dados_processo_numerico, dados_processo_categorico

    def anomalia_global():
        pass

    def anomalia_local():
        pass

    def rank():
        pass

    def correlation():
        pass

    def estatistica():
        pass
    