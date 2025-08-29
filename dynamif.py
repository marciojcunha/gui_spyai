import pandas as pd
import numpy as np
import os

class If_global: 
    def __init__(self, idx_ponto, path_v, path_h, path_a):
        self.path_h = path_h
        self.path_v = path_v
        self.path_a = path_a
        self.idx_ponto = idx_ponto #recebe o index do ponto a ser analisado

        self.diagnostico_if = {}  

        if self.path_h != None:
            npz_analise_h = os.path.join(path_h, "SPYAI_analise.npz")
            self.dados_h = np.load(npz_analise_h, allow_pickle=True)
            column_names = list(self.dados_h['columns'])
            # Encontra o índice da coluna 'DATA'
            idx_col_data = np.where(self.dados_h['columns'] == 'DATA')[0][0]

            # Obtém a coluna 'DATA'
            col_data = self.dados_h['data'][:, idx_col_data]

            # Encontra a(s) linha(s) onde o valor da coluna 'DATA' é igual a self.idx_ponto
            linha = np.where(col_data == self.idx_ponto)[0]
            feat = pd.DataFrame(self.dados_h['data'][linha], columns=column_names)
            feat = feat.iloc[0]

            i = 1
            self.avaliar_velocidade_iso_10816_3(feat, direcao = 'horizontal')
            self.avaliar_envelope_peak_to_peak(feat, direcao = 'horizontal')
            self.avaliar_aceleracao_peak_to_peak(feat, direcao = 'horizontal')
            self.avaliar_aceleracao_rms(feat, direcao = 'horizontal')
            self.avaliar_fator_crista(feat, direcao = 'horizontal')
            self.avaliar_curtose(feat, direcao = 'horizontal')
            self.avaliar_banda0(feat, direcao = 'horizontal')
            self.avaliar_banda1(feat, direcao = 'horizontal')
            self.avaliar_banda2(feat, direcao = 'horizontal')
            self.avaliar_banda3(feat, direcao = 'horizontal')
            self.avaliar_banda4(feat, direcao = 'horizontal')
            self.avaliar_banda5(feat, direcao = 'horizontal')
            self.avaliar_banda6(feat, direcao = 'horizontal')
            k = 2
        
        if self.path_v != None:
            npz_analise_h = os.path.join(path_h, "SPYAI_analise.npz")
            self.dados_h = np.load(npz_analise_h, allow_pickle=True)
            column_names = list(self.dados_h['columns'])
            # Encontra o índice da coluna 'DATA'
            idx_col_data = np.where(self.dados_h['columns'] == 'DATA')[0][0]

            # Obtém a coluna 'DATA'
            col_data = self.dados_h['data'][:, idx_col_data]

            # Encontra a(s) linha(s) onde o valor da coluna 'DATA' é igual a self.idx_ponto
            linha = np.where(col_data == self.idx_ponto)[0]
            feat = pd.DataFrame(self.dados_h['data'][linha], columns=column_names)
            feat = feat.iloc[0]

            npz_analise_v = os.path.join(path_v, "SPYAI_analise.npz")
            self.avaliar_velocidade_iso_10816_3(self.path_v, direcao = 'vertical')
            self.avaliar_envelope_peak_to_peak(self.path_v, direcao = 'vertical')
            self.avaliar_aceleracao_peak_to_peak(self.path_v, direcao = 'vertical')
            self.avaliar_aceleracao_rms(self.path_v, direcao = 'vertical')
            self.avaliar_fator_crista(self.path_v, direcao = 'vertical')
            self.avaliar_curtose(self.path_v, direcao = 'vertical')
            self.avaliar_banda0(self.path_v, direcao = 'vertical')
            self.avaliar_banda1(self.path_v, direcao = 'vertical')
            self.avaliar_banda2(self.path_v, direcao = 'vertical')
            self.avaliar_banda3(self.path_v, direcao = 'vertical')
            self.avaliar_banda4(self.path_v, direcao = 'vertical')
            self.avaliar_banda5(self.path_v, direcao = 'vertical')
            self.avaliar_banda6(self.path_v, direcao = 'vertical')
        
        if self.path_a != None:
            npz_analise_a = os.path.join(path_a, "SPYAI_analise.npz")
            self.avaliar_velocidade_iso_10816_3(self.path_a, direcao = 'axial')
            self.avaliar_envelope_peak_to_peak(self.path_a, direcao = 'axial')
            self.avaliar_aceleracao_peak_to_peak(self.path_a, direcao = 'axial')
            self.avaliar_aceleracao_rms(self.path_a, direcao = 'axial')
            self.avaliar_fator_crista(self.path_a, direcao = 'axial')
            self.avaliar_curtose(self.path_a, direcao = 'axial')
            self.avaliar_banda0(self.path_a, direcao = 'axial')
            self.avaliar_banda1(self.path_a, direcao = 'axial')
            self.avaliar_banda2(self.path_a, direcao = 'axial')
            self.avaliar_banda3(self.path_a, direcao = 'axial')
            self.avaliar_banda4(self.path_a, direcao = 'axial')
            self.avaliar_banda5(self.path_a, direcao = 'axial')
            self.avaliar_banda6(self.path_a, direcao = 'axial')

        i = 2
        
    # saidas:
    # 0: "Nível Global: N - Normal (aceitável para operação contínua)"
    # 1: "Nível Global: A1 - Alerta I -  Satisfatória (aceitável para operação limitada - defeito incipiente)"
    # 2: "Nível Global: A2 - Alerta II - Insatisfatória (requer correção em manutenção corretiva planejada - defeito relevante)"
    # 3: "Nível Global: AR - Alto Risco - Inaceitável (parada imediata recomendada - condição de falha)"
    

    def avaliar_velocidade_iso_10816_3 (self, feat, direcao):
        """
        Avalia a severidade da vibração segundo a norma ISO 10816-3.
        Parâmetros:
            vibracao_rms (float): valor RMS da vibração em mm/s.
        Retorna:
            str: Nível de severidade.
        """

        if feat['rms_v'] <= 2.8:
            self.diagnostico_if['norma_iso_10816_3_'+str(direcao)] = 0 
        elif feat['rms_v'] <= 4.5:
            self.diagnostico_if['norma_iso_10816_3_'+str(direcao)] = 1 
        elif feat['rms_v'] <= 7.1:
            self.diagnostico_if['norma_iso_10816_3_'+str(direcao)] = 2
        else:
            self.diagnostico_if['norma_iso_10816_3_'+str(direcao)] = 3

    def avaliar_envelope_peak_to_peak (self, feat, direcao):
        """
        Avalia a severidade da vibração para análise de envelope.
        Parâmetros:
            vibracao_gE (float): valor pico a pico da vibração em gE no intervalo de 0 - 1khz, com filtro 500-10khz.
        Retorna:
            str: Nível de severidade.
        """

        # verificar níveis !!!

        if feat['env_peak'] <= 0.75:
            self.diagnostico_if['env_pkpk_'+str(direcao)] = 0 
        elif feat['env_peak'] <= 2:
            self.diagnostico_if['env_pkpk_'+str(direcao)] = 1 
        elif feat['env_peak'] <= 4:
            self.diagnostico_if['env_pkpk_'+str(direcao)] = 2 
        else:
            self.diagnostico_if['env_pkpk_'+str(direcao)] = 3 

    def avaliar_aceleracao_peak_to_peak (self, feat, direcao):
        """
        Avalia a severidade da vibração para análise do sinal em aceleração - pico a pico.
        Parâmetros:
            vibracao_g (float): valor pico a pico da vibração em g.
        Retorna:
            str: Nível de severidade.
        """

        # verificar níveis !!!
        if feat['acc_peak'] <= 4.0:
            self.diagnostico_if['acc_pkpk_'+str(direcao)] = 0 
        elif feat['acc_peak'] <= 7.5:
            self.diagnostico_if['acc_pkpk_'+str(direcao)] = 1 
        elif feat['acc_peak'] <= 15:
            self.diagnostico_if['acc_pkpk_'+str(direcao)] = 2 
        else:
            self.diagnostico_if['acc_pkpk_'+str(direcao)] = 3 

    def avaliar_aceleracao_rms (self, feat, direcao):
        """
        Avalia a severidade da vibração para análise do sinal em aceleração - rms.
        Parâmetros:
            vibracao_g (float): valor rms da vibração em g.
        Retorna:
            str: Nível de severidade.
        """

        # verificar níveis !!!

        if feat['rms_acc'] <= 2:
            self.diagnostico_if['rms_acc_'+str(direcao)] = 0
        elif feat['rms_acc'] <= 3.5:
            self.diagnostico_if['rms_acc_'+str(direcao)] = 1
        elif feat['rms_acc'] <= 7:
            self.diagnostico_if['rms_acc_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_acc_'+str(direcao)] = 3

    def avaliar_fator_crista (self, feat, direcao):
        """
        Avalia a severidade da vibração para o fator de crista.
        Parâmetros:
            fator de crista (float): valor do fator de crista.
        Retorna:
            str: Nível de severidade.
        """

        if feat['crest_factor'] <= 3:
            self.diagnostico_if['crest_factor_'+str(direcao)] = 0
        elif feat['crest_factor'] <= 5:
            self.diagnostico_if['crest_factor_'+str(direcao)] = 1
        elif feat['crest_factor'] <= 7:
            self.diagnostico_if['crest_factor_'+str(direcao)] = 2
        else:
            self.diagnostico_if['crest_factor_'+str(direcao)] = 3

    def avaliar_curtose (self, feat, direcao):
        """
        Avalia a severidade da vibração para a curtose.
        Parâmetros:
            curtose (float): valor de curtose.
            Sinais normais (sem falhas): Tendem a ter uma distribuição próxima da gaussiana (kurtose ≈ 3)
            Sinais com impactos: Apresentam kurtose maior que 3, pois contêm picos curtos e de alta energia (impactos)
        Retorna:
            str: Nível de severidade.
        """

        if feat['kurtosis'] <= 3.5:
            self.diagnostico_if['kurtosis'+str(direcao)] = 0
        elif feat['kurtosis'] <= 5:
            self.diagnostico_if['kurtosis'+str(direcao)] = 1
        elif feat['kurtosis'] <= 8:
            self.diagnostico_if['kurtosis'+str(direcao)] = 2
        else:
            self.diagnostico_if['kurtosis'+str(direcao)] = 3

    def avaliar_banda0 (self, feat, direcao):
        """
        Banda 0: 0.3 a 0.78 x rpm (sub-síncronos)
        Avalia a vibração na banda 0
        Valor limite em aproximadamente 40% do nível global (10-1khz)
        """

        if feat['rms_bands_0'] <= 1.8 :
            self.diagnostico_if['rms_bands_0_'+str(direcao)] = 0
        elif feat['rms_bands_0'] <= 2.8:
            self.diagnostico_if['rms_bands_0_'+str(direcao)] = 1
        elif feat['rms_bands_0'] <= 5.7:
            self.diagnostico_if['rms_bands_0_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_0_'+str(direcao)] = 3

    def avaliar_banda1 (self, feat, direcao):
        """
        Banda 1: 0.8 a 1.2 x rpm (1x)
        Avalia a vibração na banda 1
        Valor limite em aproximadamente 80% do nível global (10-1khz)
        """

        if feat['rms_bands_1'] <= 3.6 :
            self.diagnostico_if['rms_bands_1_'+str(direcao)] = 0
        elif feat['rms_bands_1'] <= 5.7:
            self.diagnostico_if['rms_bands_1_'+str(direcao)] = 1
        elif feat['rms_bands_1'] <= 11.4:
            self.diagnostico_if['rms_bands_1_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_1_'+str(direcao)] = 3

    def avaliar_banda2 (self, feat, direcao):
        """
        Banda 2: 1.8 a 2.2 x rpm (2x)
        Avalia a vibração na banda 2
        Valor limite em aproximadamente 60% do nível global (10-1khz)
        
        """

        if feat['rms_bands_2'] <= 2.7 :
            self.diagnostico_if['rms_bands_2_'+str(direcao)] = 0
        elif feat['rms_bands_2'] <= 4.3:
            self.diagnostico_if['rms_bands_2_'+str(direcao)] = 1
        elif feat['rms_bands_2'] <= 8.5:
            self.diagnostico_if['rms_bands_2_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_2_'+str(direcao)] = 3

    def avaliar_banda3 (self, feat, direcao):
        """
        Banda 3: 2.3 a 3.6 x rpm (3x)
        Avalia a vibração na banda 3
        Valor limite em aproximadamente 50% do nível global (10-1khz)
        
        """

        if feat['rms_bands_3'] <= 2.3 :
            self.diagnostico_if['rms_bands_3_'+str(direcao)] = 0
        elif feat['rms_bands_3'] <= 3.6:
            self.diagnostico_if['rms_bands_3_'+str(direcao)] = 1
        elif feat['rms_bands_3'] <= 7.1:
            self.diagnostico_if['rms_bands_3_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_3_'+str(direcao)] = 3

    def avaliar_banda4 (self, feat, direcao):
        """
        Banda 4: 3.6 a 12.2 x rpm (folgas/rolamento estágio avançado)
        Avalia a vibração na banda 4
            
        """

        # avaliar os valores !!!!!
        falha = []
        if feat['rms_bands_4'] <= 1.3 :
            self.diagnostico_if['rms_bands_4_'+str(direcao)] = 0
        elif feat['rms_bands_4'] <= 1.8:
            self.diagnostico_if['rms_bands_4_'+str(direcao)] = 1
        elif feat['rms_bands_4'] <= 2.7:
            self.diagnostico_if['rms_bands_4_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_4_'+str(direcao)] = 3

    def avaliar_banda5 (self, feat, direcao):
        """
        Banda 5: 12.3 a 16.6 x rpm (rolamento estágio intermediário)
        Avalia a vibração na banda 5
            
        """

        # avaliar os valores !!!!!
    
        if feat['rms_bands_5'] <= 1.0 :
            self.diagnostico_if['rms_bands_5_'+str(direcao)] = 0
        elif feat['rms_bands_5'] <= 1.4:
            self.diagnostico_if['rms_bands_5_'+str(direcao)] = 1
        elif feat['rms_bands_5'] <= 2.0:
            self.diagnostico_if['rms_bands_5_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_5_'+str(direcao)] = 3


    def avaliar_banda6 (self, feat, direcao):
        """
        Banda 6: 16.7 a 25 x rpm (rolamento estágio incipiente)
        Avalia a vibração na banda 6
            
        """

        # avaliar os valores !!!!!
        
        if feat['rms_bands_6'] <= 0.9 :
            self.diagnostico_if['rms_bands_6_'+str(direcao)] = 0
        elif feat['rms_bands_6'] <= 1.2:
            self.diagnostico_if['rms_bands_6_'+str(direcao)] = 1
        elif feat['rms_bands_6'] <= 1.8:
            self.diagnostico_if['rms_bands_6_'+str(direcao)] = 2
        else:
            self.diagnostico_if['rms_bands_6_'+str(direcao)] = 3
    

class If_classification: 
    def __init__(self, idx_ponto, path_v, path_h, path_a):
        self.path_v = path_v
        self.path_h = path_h
        self.path_a = path_a
        self.idx_ponto = idx_ponto #recebe o index do ponto a ser analisado

        self.diagnostico_if = {}  

        if self.path_h != None:
            npz_analise_h = os.path.join(path_h, "SPYAI_analise.npz")
            self.dados_h = np.load(npz_analise_h, allow_pickle=True)
            column_names = list(self.dados_h['columns'])
            # Encontra o índice da coluna 'DATA'
            idx_col_data = np.where(self.dados_h['columns'] == 'DATA')[0][0]

            # Obtém a coluna 'DATA'
            col_data = self.dados_h['data'][:, idx_col_data]

            # Encontra a(s) linha(s) onde o valor da coluna 'DATA' é igual a self.idx_ponto
            linha = np.where(col_data == self.idx_ponto)[0]
            feat = pd.DataFrame(self.dados_h['data'][linha], columns=column_names)
            feat = feat.iloc[0]

            self.diagnosticar_desbalanceamento(self.path_h, direcao = 'horizontal')
            self.diagnosticar_desalinhamento(self.path_h, direcao = 'horizontal')
            self.diagnosticar_folga(self.path_h, direcao = 'horizontal')
            self.diagnosticar_cavitacao(self.path_h, direcao = 'horizontal')
            self.diagnosticar_passagempas(self.path_h, direcao = 'horizontal')
            self.diagnosticar_fluxoturb(self.path_h, direcao = 'horizontal')
            self.diagnosticar_oilproblems(self.path_h, direcao = 'horizontal')


        if self.path_v != None:
            npz_analise_v = os.path.join(path_v, "SPYAI_analise.npz")
            self.dados_v = np.load(npz_analise_v, allow_pickle=True)
            column_names = list(self.dados_v['columns'])
            # Encontra o índice da coluna 'DATA'
            idx_col_data = np.where(self.dados_v['columns'] == 'DATA')[0][0]

            # Obtém a coluna 'DATA'
            col_data = self.dados_v['data'][:, idx_col_data]

            # Encontra a(s) linha(s) onde o valor da coluna 'DATA' é igual a self.idx_ponto
            linha = np.where(col_data == self.idx_ponto)[0]
            feat = pd.DataFrame(self.dados_v['data'][linha], columns=column_names)
            feat = feat.iloc[0]

            self.diagnosticar_desbalanceamento(self.path_v, direcao = 'vertical')
            self.diagnosticar_desalinhamento(self.path_v, direcao = 'vertical')
            self.diagnosticar_folga(self.path_v, direcao = 'vertical')
            self.diagnosticar_cavitacao(self.path_v, direcao = 'vertical')
            self.diagnosticar_passagempas(self.path_v, direcao = 'vertical')
            self.diagnosticar_fluxoturb(self.path_v, direcao = 'vertical')
            self.diagnosticar_oilproblems(self.path_v, direcao = 'vertical')

        if self.path_a != None:
            npz_analise_a = os.path.join(path_a, "SPYAI_analise.npz")
            self.dados_a = np.load(npz_analise_a, allow_pickle=True)
            column_names = list(self.dados_a['columns'])
            # Encontra o índice da coluna 'DATA'
            idx_col_data = np.where(self.dados_a['columns'] == 'DATA')[0][0]

            # Obtém a coluna 'DATA'
            col_data = self.dados_a['data'][:, idx_col_data]

            # Encontra a(s) linha(s) onde o valor da coluna 'DATA' é igual a self.idx_ponto
            linha = np.where(col_data == self.idx_ponto)[0]
            feat = pd.DataFrame(self.dados_a['data'][linha], columns=column_names)
            feat = feat.iloc[0]

            self.diagnosticar_desbalanceamento(self.path_a, direcao = 'axial')
            self.diagnosticar_desalinhamento(self.path_a, direcao = 'axial')
            self.diagnosticar_folga(self.path_a, direcao = 'vertical')
            self.diagnosticar_cavitacao(self.path_a, direcao = 'vertical')
            self.diagnosticar_passagempas(self.path_a, direcao = 'vertical')
            self.diagnosticar_fluxoturb(self.path_a, direcao = 'vertical')
            self.diagnosticar_oilproblems(self.path_a, direcao = 'vertical')
        
      

    # saidas:
    # 0: condição normal
    # 1: valor global atingido porém não foi diagnosticado defeito pela classe
    # 2: desbalanceamento
    # 3: folga
    # 4: desalinhamento
    # 5: rolamento
    # 6: engrenagem
    # 7: lubrificação
    # 8: passagem de pás
    # 9: turbulência
    # 10: instabilidade do lubrificante


    def diagnosticar_desbalanceamento(self, path_direcao, direcao): 
        
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]
            harmonicos = feat[['max_harmonics_1.5', 'max_harmonics_2', 'max_harmonics_2.5', 'max_harmonics_3', 'max_harmonics_4']]
            if feat['rms_v'] > 2.8 :
                if feat['max_harmonics_1'] > 3.6: #nivel rms global ou energia na banda de 1x
                    if any(x>0.8*feat['max_harmonics_1'] for x in harmonicos) : #or sum(x>0.4 for x in harmonicos_h_folga) >= 2:
                        #print("Horizontal: Desbalanceamento descartado, 2-8x maior que 1x, verifique outra falha")            
                        self.diagnostico_if['desbalanceamento_'+str(direcao)] = 1
                    else:
                        if self.path_v == [] or self.path_h == []:
                            self.diagnostico_if['desbalanceamento'] = 2
                            #print("Horizontal: Desbalanceamento possível, mas somente uma direção radial detectada, não é possível verificar folga estrutural")    
                        else: 
                            if pd.read_csv(self.path_h).iloc[self.idx_ponto]['max_harmonics_1'] > 2*pd.read_csv(self.path_v).iloc[self.idx_ponto]['max_harmonics_1']: # 2 a 4 vezes maior
                                self.diagnostico_if['desbalanceamento_'+str(direcao)] = 3 # Folga estrutural
                                #print("Horizontal: Caracteristica de folga estrutural")
                            else: 
                                self.diagnostico_if['desbalanceamento_'+str(direcao)] = 2
                                #print("Horizontal: Desbalanceamento possível")
                else: self.diagnostico_if['desbalanceamento_'+str(direcao)] = 1 #print("Horizontal: Valor global atingido porém não está relacionado ao desbalanceamento")                         
            else: self.diagnostico_if['desbalanceamento_'+str(direcao)] = 0 #print("Horizontal: Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")       

    def diagnosticar_desalinhamento(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]
            harmonicos = feat[['max_harmonics_4', 'max_harmonics_5', 'max_harmonics_6', 'max_harmonics_7', 'max_harmonics_8']]
            if feat['rms_v'] > 2.8 :
                if feat['max_harmonics_2'] > 2.7: # energia na banda de 2x
                    if sum(x>0.5*feat['max_harmonics_2'] for x in harmonicos) >= 2 : #any(x>x3h for x in harmonicos_h) or any(x>0.5*x2h for x in harmonicos_h)
                        self.diagnostico_if['desalinhamento_'+str(direcao)] = 1
                        #print("Horizontal: Desalinhamento paralelo descartado, 4-8x maior que 1x, 2x e 3x, verifique outra falha, como folga")            
                    else: 
                        if feat['max_harmonics_2'] > 0.8*feat['max_harmonics_1'] :
                            self.diagnostico_if['desalinhamento_'+str(direcao)] = 4
                            #print("Horizontal: Desalinhamento Paralelo possível")
                        else: self.diagnostico_if['desalinhamento_'+str(direcao)] = 1 #print("Horizontal: Valor global atingido porém não está relacionado ao desalinhamento")                           
                else: self.diagnostico_if['desalinhamento_'+str(direcao)] = 1 #print("Horizontal: Valor global atingido porém não está relacionado ao desalinhamento")                         
            else: self.diagnostico_if['desalinhamento_'+str(direcao)] = 0 #print("Horizontal: Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    def diagnosticar_folga(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]
            harmonicos = feat[['max_harmonics_1', 'max_harmonics_2', 'max_harmonics_3']]
            harmonicos_folga = feat[['max_harmonics_4', 'max_harmonics_5', 'max_harmonics_6', 'max_harmonics_7', 'max_harmonics_8']]
            sub_harmonicos = feat[['max_harmonics_0.5', 'max_harmonics_1.5', 'max_harmonics_2.5']]
            if feat['rms_v'] > 2.8 : #nivel rms global
                if sum(x>0.5 for x in sub_harmonicos) >= 2 : 
                    self.diagnostico_if['folga_'+str(direcao)] = 3
                    #print("Horizontal: Folga mecânica possível - subharmônicos detectados")
                else:                            
                    if sum(x>2 for x in harmonicos) >= 2 : 
                        if sum(x>0.3 for x in harmonicos_folga) >= 2 : #any(x>x3h for x in harmonicos_h) or any(x>0.5*x2h for x in harmonicos_h)
                            self.diagnostico_if['folga_'+str(direcao)] = 3
                            #print("Horizontal: Folga mecânica possível")            
                        else: 
                            self.diagnostico_if['folga_'+str(direcao)] = 1 #print("Horizontal: Folga mecânica descartada, verifique outra falha")            
                    else: self.diagnostico_if['folga_'+str(direcao)] = 1 #print("Horizontal: Valor global atingido porém não está relacionado a folga")                         
            else: self.diagnostico_if['folga_'+str(direcao)] = 0 #print("Horizontal: Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    # precisa modificar os features para receber as freq. de rolamento
    # modificar as features para retornar valor maximo e nao soma
    # colocar um IF caso nao exista a freq. de rolamento (BPFO, BPFI, BSF e FTF) ou passar valor 0
    def diagnosticar_rolamento(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]
            harmonicos = feat[['rol3']] # freq. BPFO, BPPFI, BSF e FTF
            
            if feat['env_peak'] > 2 : #nivel gE - pkpk
                if sum(x>1 for x in harmonicos) >= 2 or feat[['rol1']] >= 4 or feat[['rol2']] >= 2: 
                    self.diagnostico_if['rolamento_'+str(direcao)] = 5
                    #print("Defeito no rolamento possível - harmônicos detectados")
                else:                            
                    self.diagnostico_if['rolamento_'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito no rolamento")                         
            else: self.diagnostico_if['rolamento_'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    # criar o feature com o nivel de vibracao nas GMFs em velocidade
    # colocar um IF caso nao exista a freq. de rolamento (GMF) ou passar valor 0
    def diagnosticar_engrenagem(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]
            gmf_vel = feat[['rol3']] # valores das frequencias de engrenamento em mm/s
            gmf_acc = feat[['rol3']] # valores das frequencias de engrenamento em g
            
            if feat['acc_peak'] > 4 : #nivel g global pk-pk
                if any(x>10.2 for x in gmf_vel) >= 1 or any(x>10 for x in gmf_acc) >= 1 :  #10.2 mm/s (pk-pk, recomendacao rexnord falk), nivel g - pkpk (norma ANSI/AGMA 6000-B96)
                    self.diagnostico_if['engrenamento_'+str(direcao)] = 6
                    #print("Defeito no engrenamento possível - harmônicos de gmf detectados")
                else:                            
                    self.diagnostico_if['engrenamento_'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito no engrenamento")                         
            else: self.diagnostico_if['engrenamento_'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    # mudar o lub_2 para média dos 100 menores valores no espectro de envelope até 500 hz
    # mudar o lub_1, remove as harmonicas e as 10 maiores assicronas, depois faz o valor medio das 10 maiores freq. acima de 500 hz
    # pensar em mais alguma 
    def diagnosticar_lubrificacao(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]

            if feat['acc_peak'] > 4 : #nivel g global pk-pk
                if feat['lub_2'] > 0.01 and feat['lub_1'] > 0.5 :
                    self.diagnostico_if['lubrificacao'+str(direcao)] = 7
                    #print("Defeito de lubrificacao possível - aumento do carpete detectado")
                else:                            
                    self.diagnostico_if['lubrificacao'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito de lubrificacao")                         
            else: self.diagnostico_if['lubrificacao'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    # conferir os limites da cavitacao
    def diagnosticar_cavitacao(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]

            if feat['acc_peak'] > 4 : #nivel g global pk-pk
                if feat['cav_2'] > 0.01 and feat['cav_1'] > 0.5 :
                    self.diagnostico_if['cavitacao'+str(direcao)] = 7
                    #print("Defeito de cavitacao possível - aumento do carpete detectado")
                else:                            
                    self.diagnostico_if['cavitacao'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito de cavitacao")                         
            else: self.diagnostico_if['cavitacao'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    # mudar o parametro para valor maximo de amplitude entre as harmonicas (melhor para comparar)
    def diagnosticar_passagempas(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]

            if feat['acc_peak'] > 4 : #nivel g global pk-pk
                if feat['pas_1'] > 2.8 : #acima do nivel global da norma
                    self.diagnostico_if['passagem de pás'+str(direcao)] = 8
                    #print("Defeito de passagem de pas possível - harmônicos de passagem de pás detectado")
                else:                            
                    self.diagnostico_if['passagem de pás'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito de passagem de pás")                         
            else: self.diagnostico_if['passagem de pás'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    # nao vamos precisar do tub_2 por enquanto, pode desativa-lo  xxxxxxxxxxx
    # calcular o valor maximo 
    def diagnosticar_fluxoturb(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]

            if feat['rms_v'] > 2.8 : #nivel vel rms global
                if feat['rms_bands_0'] > 0.7* feat['rms_bands_1'] and feat['turb_1'] > 0.1: 
                    self.diagnostico_if['turbulencia'+str(direcao)] = 9
                    #print("Defeito de turbulência possível - sinal randomico abaixo da freq. de rot. detectado")
                else:                            
                    self.diagnostico_if['turbulencia'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito de passagem de pás")                         
            else: self.diagnostico_if['turbulencia'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

    def diagnosticar_oilproblems(self, path_direcao, direcao): 
        if isinstance(path_direcao, (str)):
            feat = pd.read_csv(path_direcao) ### AJUSTAR COMO VAI CARREGAR
            feat = feat.iloc[self.idx_ponto]

            if feat['rms_v'] > 2.8 : #nivel vel rms global
                if feat['oil_1'] > 0.7* feat['rms_bands_1'] or feat['oil_1'] > 1.5: 
                    self.diagnostico_if['oilproblems'+str(direcao)] = 10
                    #print("Defeito de instabilidade no lubrificante possível - harmônicos detectado")
                else:                            
                    self.diagnostico_if['oilproblems'+str(direcao)] = 1 #print("Valor global atingido porém não está relacionado a defeito de passagem de pás")                         
            else: self.diagnostico_if['oilproblems'+str(direcao)] = 0 #print("Condição normal para o nível global")                                   
        else: pass #print("Posição horizontal não coletada")

