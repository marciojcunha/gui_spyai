import dynamsignal as ds
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import dynamif
import dynamsignal as ds
from dynamfeatures import Features_classification
import os
    
class Quality_Signal:
        
    def unit_verification(dados):
        cols = dados['columns']
        arr = dados['data']

        idx_amostras = np.where(cols == 'Amostras')[0][0]
        idx_unidade = np.where(cols == 'Unidade')[0][0]
        idx_dado = np.where(cols == 'Dado')[0][0]

        # Filtrar linhas onde a unidade é diferente de 'g'
        unidades = arr[:, idx_unidade]
        mask_g = (unidades != 'g')

        # Itera somente nas linhas que precisam conversão
        indices = np.where(mask_g)[0]

        for i in indices:
            linha = arr[i]
            amostras = int(linha[idx_amostras])
            sinal = linha[idx_dado : idx_dado + amostras].astype(float)
            sinal = sinal / 10
            linha[idx_dado : idx_dado + amostras] = sinal

        return dados
    
    def define_base(index, df_numpy, df_analise,path_h,path_v,path_a):

        num_amostras = df_numpy[:,5]

        # Extrai o vetor do sinal bruto
        sinal = df_numpy[:,10:int(num_amostras)]

        # Extrair arrays do npz df_numpy
        def extrair_tag_ponto(caminho):
            partes = caminho.strip( os.path.sep).split( os.path.sep)
            tag = partes[-2] if len(partes) >= 2 else 'Desconhecido'
            ponto = partes[-1] if len(partes) >= 1 else 'Desconhecido'
            return tag, ponto

        tag, ponto = extrair_tag_ponto(df_numpy[:,0])
        dts_str = df_numpy[:,1]
        unidade = df_numpy[:,2]
        fs = df_numpy[:,2]
        dt = df_numpy[:,2]
        rpm = df_numpy[:,2]

        if sinal is None:
            raise ValueError("Array 'sinal' não encontrado em df_numpy.")

        if dynamif.If_global(index,sinal,path_h,path_v,path_a) == 0:
            sinal_base = sinal
        else:
            sinal_base = Quality_Signal.baseline_signal(sinal, dt, rpm)

        fs = 1 / dt
        feats = Features_classification(sinal, fs=fs).features

        registro = {
            'TAG': tag,
            'Ponto': ponto,
            'data': dts_str,
            'dt': dt,
            'Fs': fs,
            'Unidade': unidade,
            'RPM': rpm,
            **feats
        }

        # Copiar para dict mutável para atualização
        df_analise_data = dict(df_analise)

        # Atualizar/Adicionar o registro, assumindo a chave pela data (dts_str)
        df_analise_data[dts_str] = registro

        # Retorna os sinais e o dict atualizado
        return sinal, sinal_base, df_analise_data


    def baseline_signal(data, dt, rpm, ovl = 4.5, units='g') :
        # data: machine vibration signal (m/s²)
        # dt = aquisition time (s)
        # ovl = overall velocity limit 
        # gerar desbalanceamento e desalinhamento
        fs = 1/dt 
        f0 = rpm/60 
        fm = f0/2
        npto_old = len(data) 
        i = int(fs/fm) 
        fs_new = i*fm
        dt_new = 1/fs_new 
        sinal = ds.utilities.detrend(data,dt, deg = 2)
        x = ds.utilities.changedeltat(sinal,dt,dt_new)[1] 
        npto = len(x)
        df = 1/(npto*dt_new)
        i_0 = int(f0/df) 
        i_m = int(i_0 / 2)
        # Limpar harmonicos
        X = np.fft.fft(x/npto)
        df = fs_new/npto
        i_0 = int(f0/df) 
        i_0 = i_0-2 + np.argmax(np.abs(X[i_0-2:i_0+3])) 
        i_m = int(i_0 / 2)
        # Diminui subharmônicoa 
        for i in range(2,i_m+2) : 
            X[i] /= 4            
        # Acerta as harmonicas até 5 f_0 
        X1 = np.abs(2*np.sqrt( np.sum(X[i_0-2:i_0+3]**2) )*1000/(2*np.pi*fm))
        if units == 'g' : 
            X1 *=10           
        ratio = 2.6/X1 # norma ISO
        npto2 = int(npto/2)
        X[i_0-2:i_0+3] *= ratio
        X1 = np.abs(2*np.sqrt( np.sum(X[i_0-2:i_0+3]**2) ))
        ratio /=4
        for i in range(2*i_0,6*i_0,i_0) :
            Xi = np.abs(2*np.sqrt( np.sum(X[i-2:i+3]**2) ))  
            #if Xi > X1*ratio : 
            X[i-2:i+3] =  X[i-2:i+3]*X1*ratio/Xi
            X[i-i_m-2:i-i_m+3] /= 4 # cort 6 dB as subharmonicas  
            ratio /=1.5
             
       
        i_1000 = int(1000/df)     
        if i_1000 > npto2 : 
            i_1000 = npto2
        # pegar no envelope frequências não harmonicas da rotação 
        x_env = ds.timeparameters.envelope(x,500,dt=dt) 
        n = len(x_env)
        X_env = np.abs(np.fft.fft(hann(n)*x_env/n))
        X_env[0:int(10/df)] = 0 
        X_env[i_1000:] = 0 
        max = X_env[i_0] 
        ganho = np.max(X_env)/max
        if ganho > 2: 
            ganho = np.sqrt(ganho)
        X_env[i_0-3:i_0+4] = 0 
        X_env[i_m:i_1000:i_m] = 0 
        i_max = np.argmax(X_env) 
        while X_env[i_max] > max : 
            X[i_max-2:i_max+3] /=4 
            X_env[i_max-3:i_max+4] = 0
            i_max = np.argmax(X_env) 

        X[i_1000:npto2] /=2 
        x = np.zeros(npto).astype('complex')
        x[0:npto2] = X[0:npto2]
        x[npto-1:npto2+1:-1] = np.conj(X[1:npto2-1])
        x= np.real(np.fft.fft(np.conj(x)))
        x = ds.utilities.changedeltat(x,dt_new,dt)[1]
        x = x[0:npto_old]
        v = 1000*ds.utilities.time_integration(x,dt )
        if units == 'g' : 
            v*=10
        rms_v = ds.timeparameters.rms(v)
        if rms_v > ovl :
            x *= ovl/rms_v   

        return x