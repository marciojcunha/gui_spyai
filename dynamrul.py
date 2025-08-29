import numpy as np
import pandas as pd
from scipy.signal.windows import hann
from math import gamma
from scipy.optimize import fsolve
import scipy.stats as st
import matplotlib.pyplot as plt
import dynamsignal as ds
import os
import datetime
from scipy.stats import ks_2samp
              

class utilities : 

    def date_to_numeric(date, type = 'meses') : 
        aux = '%s'%(date) 
        j = aux.find('-')
        if j < 0 : 
            j = aux.find('/')   
        if j < 3 : # data dia - mês - ano ... 
            dia = int(aux[0:2])
            mes = int(aux[3:5])
            ano = int(aux[6:10])
        else : 
            ano = int(aux[0:4])
            mes = int(aux[5:7])
            dia = int(aux[8:10])

        hora = 0
        minuto = 0 
        segundo = 0 
        if len(aux) > 10 : # tem horas minutos e segundos
            hora = int(aux[11:13])
            minuto = int(aux[14:16])
            segundo = int(aux[17:19])
        t = datetime.datetime(ano, mes, dia, hora, minuto, segundo) 
        seconds = t.timestamp()
        if type == 'meses' : 
            return seconds/2628288 # transforma para mês
        if type == 'dias' :
            return seconds/86400 # transforma para dia
        if type == 'horas' : 
            return seconds/3600 # transforma para hora


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

    def baseline_signal_old(data, dt, rpm, ovl = 3) :
        # data: machine vibration signal ( g )
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
        # filtra passa alta em 10 Hz
        x = ds.utilities.frequency_filter(x,10,dt_new,type='highpass') 
        npto = len(x)
        df = 1/(npto*dt_new)
        # Conferir harmonicos
        X = np.fft.fft(x/npto)
        df = fs_new/npto
        i_0 = int(f0/df) 
        i_0 = i_0-2+np.argmax(np.abs(X[i_0-2:i_0+3]))
        i_m = int(i_0 / 2)
        i_m = i_m-2+np.argmax(np.abs(X[i_m-2:i_m+3]))
        # Diminue as harmonicas e sub-harmonicas até 5 f_0 (- 6 dB)
        npto2 = int(npto/2)
        X1_max = 2*np.pi*2*rpm/600000 
        X1 = np.abs(X[i_0])+1e-8
        if X1 > X1_max : 
            X[i_0] *= X1_max/X1  
            X1 = X1_max
        # abaixo de f0 não pode ter amplitude maior do que 20% de X1
        for i in range(0,i_0-2) : 
            if np.abs(X[i]) > .2*X1 : 
                X[i] *=  .2*X1/(np.abs(X[i])+1e-8)   
        for i in range(2*i_0,6*i_0,i_0) : # o super harmônico não pode ser 50% maior do que o antrior 
            if np.max(np.abs(X[i-2:i+3])) > .5*np.max(np.abs(X[i-i_0-2:i-i_0+3])) : 
                X[i-2:i+3] *=  .5*np.abs(X[i-i_0-2:i-i_0+3])/(np.abs(X[i-2:i+3])+1e-8)   
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

        X[i_1000:npto2] /=ganho 
        x = np.zeros(npto).astype('complex')
        x[0:npto2] = X[0:npto2]
        x[npto-1:npto2+1:-1] = np.conj(X[1:npto2-1])
        x= np.real(np.fft.fft(np.conj(x)))
        x = ds.utilities.changedeltat(x,dt_new,dt)[1] 
        x = x[0:npto_old]
        v = 10000*ds.utilities.time_integration(x,dt )
        rms_v = ds.timeparameters.rms(v)
        if rms_v > ovl :
            x *= ovl/rms_v     
        return x
    

    def signal_dft_clean(data, dt ,freq, harm_old, harm_new ) :
        # clean signal 
        x = np.copy(data) 
        npto = len(data) 
        t = np.linspace(0,(npto-1)*dt,npto) 
        x_o = np.zeros(npto)
        x_n = np.zeros(npto)
        for k_ext in range(len(freq)) : 
            w = 2*np.pi*freq[k_ext]            
            sn = np.sin(w*t) 
            cs = np.cos(w*t)
            x_o += (np.real(harm_old[k_ext])*cs + np.imag(harm_old[k_ext])*sn)
            x_n += (np.real(harm_new[k_ext])*cs + np.imag(harm_new[k_ext])*sn)
        x = x - x_o + x_n       
        return x

    def hist_user(x, bins, hist_nbins = None ):
        if hist_nbins is None : 
            #nbins = utilities.fd_optimal_bins(x)
            nbins = bins.size - 1
            hist = np.zeros(nbins)
            for i in range(0, nbins - 1):
                hist[i] = x[x < bins[i + 1]].size
            hist[i + 1] = x[x > bins[i + 1]].size
            hist[1 : nbins - 1] = hist[1 : nbins - 1] - hist[0 : nbins - 2]
            return hist/len(x) 
        nbins = hist_nbins  
        #nbins = utilities.fd_optimal_bins(x)  
        bins = np.linspace(x.min(), x.max(), num=nbins + 1)    
        hist = np.zeros(nbins)
        for i in range(0, nbins - 1):
            hist[i] = x[x < bins[i + 1]].size
        hist[i + 1] = x[x > bins[i + 1]].size
        hist[1 : nbins - 1] = hist[1 : nbins - 1] - hist[0 : nbins - 2]
        hist /= len(x)
        return bins,hist    



    
    def interpola_reta(x,y, x_pred) : 
        y0 = y - y[0] 
        x0 = x -x[0]
        #x0,y0 = rul.sequencia(x0,y0)
        p = np.polyfit(x0, y0, 1) 
        e2 = np.sum((y0-np.polyval(p,x0))**2)
        y_pred = y[0]+np.polyval(p,(x_pred-x[0])) 
        return e2, y_pred

    def interpola_parabola(x,y, x_pred) : 
        y0 = y - y[0] 
        x0 = x -x[0]
        #x0,y0 = rul.sequencia(x0,y0)
        p = np.polyfit(x0, y0, 2) 
        e2 = np.sum((y0-np.polyval(p,x0))**2)
        y_pred = y[0]+np.polyval(p,(x_pred-x[0])) 
        return e2, y_pred

    def interpola_exp(x,y, x_pred) :
        y_bas = np.abs(y[0])+1e-8
        y0 = (y/y_bas)
        x0 = ((x)-x[0])/x[-1]
        I = y0>1 
        if np.sum(I) < 2 : # falhou
            y_pred = y[-1]*np.ones(len(x_pred))
            return 1e12, y_pred
        x0 = x0[I] 
        y0 = y0[I]  
        y0 = np.log(y0) 
        p = np.array(np.polyfit(x0,y0,1))
        beta = p[0] 
        alfa = np.exp(p[1]) 
        y_t = y_bas*alfa*np.exp(beta*(x-x[0])/x[-1])  
        e2 = np.sum((y-y_t)**2)
        y_pred = y_bas*alfa*np.exp(beta*(x_pred-x[0])/x[-1])  
        return e2, y_pred
    
    def best_fit(x,y) : 
        # Esta função escolhe estatisticamente a melhor função func entre uma reta, uma parábola, uma root,
        # uma exponencial 
        # Retorna: 
        #           0 - reta 
        #           1 - parábola 
        #           2 - exponencial
        npto = len(x)
        if npto < 3 : return 0 # só pode ser uma reta
        # ganha menor erro médio quadrático 
        e20 = utilities.interpola_reta(x,y,y[-1])[0]
        e21 = utilities.interpola_parabola(x,y,y[-1])[0]+1e12
        e22 = utilities.interpola_exp(x,y,y[-1])[0] 
        if e20 > e21 : 
            if e22 < e21 : return 2
            else: return 1
        if e20 > 22 : 
            return 2
        return 0  


    def prev_par_bootstrap(x,y,x_prev, st) : 
        i_func = utilities.best_fit(x,y)
        # hora do Monte Carlo
        npto = len(x_prev) 
        results = np.zeros((100,npto)) 
        for i in range(100) : 
            yda = np.sort(y + np.random.normal(loc=0, scale=st, size = len(y)) )
            xda = x +np.random.normal(0,.1, size=len(x))
            if i_func == 0 : results[i,:] = utilities.interpola_reta(xda,yda,x_prev)[1] 
            elif i_func == 1 : results[i,:] = utilities.interpola_parabola(xda,yda,x_prev)[1] 
            else :  results[i,:] = utilities.interpola_exp(xda,yda,x_prev)[1] 
        return results

    def prev_param_IC(tempo, features, delta_time , st_data) :             
        n_datas = 6 
        n_times = len(delta_time)
        IC_80 = np.zeros(3*n_times)
        IC_90 = np.zeros(3*n_times)
        IC_95 = np.zeros(3*n_times)  
        st = st_data/2
        tempo_prev = tempo[-1]+delta_time
        feat = np.copy( features )
        tempo_ant = np.copy(tempo) 
        bs = utilities.prev_par_bootstrap(tempo_ant,feat,tempo_prev, st)
        IC_80 = np.quantile(bs, [0.1,.5,.9] , axis= 0 ).flatten()  
        IC_90 = np.quantile(bs, [0.05,.5,.95] , axis= 0 ).flatten()   
        IC_95 = np.quantile(bs, [0.025,.5,.975] , axis= 0 ).flatten()  
        IC = np.concatenate( (IC_80,IC_90, IC_95) ) 
        return IC 


    def prepara_retorno(npto,data, data_inf,data_sup) : 
        y_ret = [] 
        if npto == 1 : 
            y_ret.append('000000000')
            return
        data_sup[0] = data[0] 
        data_inf[0] = data[0]
        for i in range(npto) : 
            aux1 = int(1000*(data-data_inf)/data)
            aux1[aux1<0] = 0 
            aux1[aux1>999] = 999  
            s1 = '%d'%(aux1)
            if aux1< 100 : s1 = '0%d'%(aux1) 
            if aux1 < 10: s1 = '00%d'%(aux1)
            aux2 = int(1000*(data-data[0])/(data[0]+1e-6))
            aux2[aux2<0] = 0 
            aux2[aux2>999] = 999  
            s2 = '%d'%(aux2)
            if aux2< 100 : s2 = '0%d'%(aux2) 
            if aux2 < 10: s2 = '00%d'%(aux2)
            aux3 = int(1000*(data_sup-data)/data)
            aux3[aux3<0] = 0 
            aux3[aux3>999] = 999  
            s3 = '%d'%(aux3)
            if aux3< 100 : s3 = '0%d'%(aux3) 
            if aux3 < 10: s3 = '00%d'%(aux3)
            y_ret.append(s1+s2+s3)
        return y_ret    


class basic_defects :
    def velocity_10_1000(data, dt) : 
        # velocidade 10:1000
        acel  = np.copy( data )               
        acel = ds.utilities.detrend(acel,dt, deg = 2)
        acel_filt = ds.utilities.frequency_filter(acel,[10,1000], dt,type='bandpass')
        v = ds.utilities.time_integration(acel_filt,dt)
        rms = 10000*ds.timeparameters.rms(v)
        return rms
    
    def harm1(data, dt, rpm ) : 
        # energia do sinal na velocidade de rotação         
        acel  = np.copy( data )    
        rps = rpm/60 
        nfft = len(acel) 
        df = 1/(nfft*dt)
        i_0 = round(rps/df) 
        df_new = rps/i_0 
        dt_new = 1/(nfft*df_new)
        acel_harm = ds.utilities.changedeltat(acel,dt,dt_new)[1]   # corrigir dt para elimimar leakge   
        Acel = ds.utilities.fft_spectrum(acel_harm)**2 
 
        i_0 = int(rps/df_new) 
        if i_0 < 2 : # máquina parada
            return 0
        rms = 0 
        for i in range(i_0-2,i_0+3) : # somar a energia das bandas laterais devido a fft
            rms += Acel[i] 
        rms = 10000*np.sqrt(rms)/(2*np.pi*rps)     
        return rms                 

    def acel_high( data , dt) : 
        acel  = np.copy( data )                 
        aux = ds.utilities.frequency_filter(acel,1000,dt, type='highpass') # aceleração na alta > 1000 Hz 
        rms = ds.timeparameters.rms(aux)
        return rms
    
    def JB_harm6( data , dt, rpm) : # 0.5X 2X, 3X, 4X, 6X and 8X
        freq_bas = rpm/60 
        freq_max = 9*freq_bas 
        fc = 1/dt 
        if fc/2 < freq_max : 
            freq_max = fc/2 
        # energia do sinal na velocidade de rotação         
        acel  = np.copy( data )    
        rps = rpm/60 
        nfft = len(acel) 
        df = 1/(nfft*dt)
        i_0 = round(rps/df) 
        df_new = rps/i_0 
        dt_new = 1/(nfft*df_new)
        acel_harm = ds.utilities.changedeltat(acel,dt,dt_new)[1]   # corrigir dt para elimimar leakge  
        DEP = ds.utilities.fft_spectrum(acel_harm)**2 
        df = df_new 
        i_0 = int(rps/df)
        i = int(i_0/2)
        if i < 2 : 
            i = 2
        rms = DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2] 
        i = 2*i_0
        rms += (DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2]) 
        i = 3*i_0
        rms += (DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2])         
        i = 4*i_0
        rms += (DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2])         
        i = 6*i_0
        rms += (DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2])         
        i = 4*i_0
        rms += (DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2])  
        i = i_0 
        aux = (DEP[i-2]+DEP[i-1]+DEP[i]+DEP[i+1]+DEP[i+2])  
        return np.sqrt(rms/aux)       

    
    def low_order_rms( data , dt, rpm) : 
        # https://philarchive.org/archive/PINIOB : 
        freq_bas = rpm/60 
        freq_max = 5.5*freq_bas 
        fc = 1/dt 
        if fc/2 < freq_max : 
            freq_max = fc/2 
        # energia do sinal na velocidade de rotação         
        acel  = np.copy( data )    
        rps = rpm/60 
        nfft = len(acel) 
        df = 1/(nfft*dt)
        i_0 = round(rps/df) 
        df_new = rps/i_0 
        dt_new = 1/(nfft*df_new)
        acel_harm = ds.utilities.changedeltat(acel,dt,dt_new)[1]   # corrigir dt para elimimar leakge  
        acel_harm = ds.utilities.detrend(acel_harm,dt, deg = 2)
        acel_filt = ds.utilities.frequency_filter(acel_harm,[10,freq_max], dt,type='bandpass')
        v = ds.utilities.time_integration(acel_filt,dt)
        DEP = ds.utilities.fft_spectrum(v)**2 
        df = df_new 
        i_10 = int(10/df) 
        DEP[0:i_10] = 0 
        i_5h = int(5.5*freq_bas/df) 
        DEP[i_5h:] = 0         
        dep = 0 
        freq = np.arange(0, freq_max, df, dtype = float)
        i_freq = np.argmin( np.abs(freq-freq_bas) )
        dep += np.sum(DEP[i_freq-2:i_freq+3]) 
        DEP[i_freq-2:i_freq+2] = 0 
        for i in range(5) : 
            i_freq = np.argmax(DEP) 
            dep += np.sum(DEP[i_freq-2:i_freq+3]) 
            DEP[i_freq-2:i_freq+3] = 0             
        dep = np.sqrt(dep) 
        return dep  

    def rms_no_harm_1000(data,dt,rpm) :      # 6 - no_harm_5_low - rms não harmônicas acima do quinto harmônico [m/s²] até 1000 Hz         
        acel  = np.copy( data )    
        rps = rpm/60 
        nfft = len(acel) 
        df = 1/(nfft*dt)
        i_0 = round(rps/df) 
        #df_new = rps/i_0 
        #dt_new = 1/(nfft*df_new)
        #acel_harm = ds.utilities.changedeltat(acel,dt,dt_new)[1]   # corrigir dt para elimimar leakge
        freq_env = 2000 
        while freq_env > .2/dt : 
            freq_env *= .8 
        a_env = ds.timeparameters.envelope(acel,2000,dt=dt)                   
        df = 1/(nfft*dt)
        i_0 = round(rps/df) 
        i_1000 = int(1000/df)

        A_env = np.abs(np.fft.fft(2*a_env/nfft))[0:i_1000+2] 
        A = ds.utilities.fft_spectrum(acel)[0:i_1000+2]
        A = 2*A*A # Valor médio quadrático  
        A_env[0:5*i_0+2] = 0 
        # zerar componentes harmônicas
        for k in range(i_0,i_1000,i_0) : 
            A_env[k-2:k+3] = 0 
        # zerar harmonicos de 60 
        i_60 = int(60/df) 
        for k in range(i_60,i_1000,i_60) : 
            A_env[k-2:k+3] = 0        
        # acumular energia de 10 não harmônicos
        soma = 0 
        for k in range(10) : 
            i = np.argmax(A_env) 
            soma += np.sum(A[i-2:i+3]) 
            A_env[i-2:i+3] = 0 
        return np.sqrt(soma)    

    def electric_motor(sinal, dt, rpm):
        freq_bas = rpm/60
        if freq_bas > 50:
            freq_min, freq_max = 18 * freq_bas, 43 * freq_bas
        elif freq_bas > 12:
            freq_min, freq_max = 30 * freq_bas, 60 * freq_bas
        else:
            freq_min, freq_max = 0, 0
        freq_max = min(freq_max, 0.5 / dt)
        acel = np.copy(sinal)
        nfft = len(acel)
        nfft2 = int(nfft / 2.56)
        df = 1 / (nfft * dt)
        i_60 = round(60 / df)
        df = 60/i_60 
        dt_new = 1/(df*nfft) 
        acel_harm = ds.utilities.changedeltat(acel,dt,dt_new)[1]        
        DEP = 2 * np.abs(np.fft.fft(acel_harm / nfft)[:nfft2]) ** 2
        df = 1/(nfft*dt_new)
        i_min = int(freq_min / df)
        i_max = int(freq_max / df)
        dep = sum(DEP[i + 2 * i_60] for i in range(-2, 3))
        DEP[:i_min] = 0
        DEP[i_max:] = 0
        for _ in range(3):
            i_freq = np.argmax(DEP)
            DEP[i_freq - 2:i_freq + 3] = 0
            for j in range(i_freq - 3 * i_60, i_freq + 2 * i_60 + 1, i_60):
                dep += np.sum(DEP[j - 2:j + 3])
            DEP[i_freq - 2*i_60-2:i_freq + 2 * i_60 + 3] = 0    
        return np.sqrt(dep)
    
    def calc_symptoms_basic( data, dt, rpm  ) : 
        simp = [ basic_defects.harm1(data, dt, rpm )  ] 
        simp.append(  basic_defects.velocity_10_1000(data, dt ) )
        return simp

    def calc_symptoms_all( data, dt, rpm  ) : 
        simp = [ basic_defects.harm1(data, dt, rpm )  ] 
        simp.append(  basic_defects.velocity_10_1000(data, dt ) )
        simp.append(  basic_defects.low_order_rms(data, dt, rpm ) )
        simp.append(  basic_defects.JB_harm6(data, dt, rpm ) )
        simp.append(  basic_defects.acel_high(data, dt ) )
        simp.append(  basic_defects.rms_no_harm_1000(data, dt, rpm ) )
        simp.append(  basic_defects.electric_motor(data, dt, rpm ) )
        return simp

    def calc_symptoms_low_frequency( data, dt, rpm  ) : 
        simp = [ basic_defects.harm1(data, dt, rpm )  ] 
        simp.append(  basic_defects.velocity_10_1000(data, dt ) )
        simp.append(  basic_defects.low_order_rms(data, dt, rpm ) )
        simp.append(  basic_defects.JB_harm6(data, dt, rpm ) )
        simp.append(  basic_defects.rms_no_harm_1000(data, dt, rpm ) )
        simp.append(  basic_defects.electric_motor(data, dt, rpm ) ) 
        simp.append( 0 )  
        return simp     



class annomaly : 
    def anomaly_load(dir_path,unit,tag) : 
        filename = dir_path + os.path.sep + 'Files' + os.path.sep + 'spectrum' + os.path.sep +  unit + '_' + tag + '.npy'
        isExist = os.path.exists(filename)
        if isExist : 
            return np.load(filename,allow_pickle=True ) 
        return None
    


    def iniatilize_histogram(dir_path,data,unit,tag,x_data,dt,i_sample,n_samples,n_bins=10,noise_level=1 ) :
        x = np.copy(x_data)

        if n_bins < 8: 
            n_bins=8      
        npto = len(x)        
        #xf = ds.utilities.frequency_filter(x,[10,1000],dt,type='bandpass') 
        xf = ds.utilities.frequency_filter(x,10,dt,type='highpass')          
        v = xf #1000*ds.utilities.time_integration(xf,dt)
        Freq = np.linspace(10,1000,num=1800)
        df = Freq[2]-Freq[1] 
        Vc,Vs = ds.utilities.dft(v,dt,Freq)
        V = Vc + 1j*Vs 
        freq_values = np.zeros(100).astype(complex)
        freq = np.zeros(100)
        harm_old = np.zeros(100).astype(complex)
        harm_new = np.zeros(100).astype(complex)
        i_5 = int(5/df) 
        V[0:i_5] = 0  
        dep = np.abs(V)
        ii = 0
        for i in range(10):
            index = np.argmax(dep) 
            if index < 8 : 
                index = 8 
            if index > 1795 : 
                index = 1795    
            for j in range(index-3,index+4) : 
                if np.abs(V[j]) > 1e-6 : 
                    freq[ii] = Freq[j] 
                    harm_old[ii] = V[j]
                    freq_values[ii] = V[j] 
                    ii += 1           
            dep[index-3:index+4] = 0 
            V[index-3:index+4] = 0
        freq = freq[0:ii] 
        harm_old = harm_old[0:ii] 
        freq_values = freq_values[0:ii]+1e-12
        v = v*hann(npto)
        v_clean = utilities.signal_dft_clean(v,dt,freq,harm_old,harm_new)    
        rms = ds.timeparameters.rms(v_clean)
        x_noise =np.random.uniform(0,noise_level*rms,size=npto )
        bin_time, hist_time = utilities.hist_user(v+x_noise,None,hist_nbins=n_bins)    
        if i_sample > 1 :
            data = data.tolist()        
            data[0] = i_sample 
            data
        else : 
            data.append(i_sample)
            data.append(dt)
            data.append(n_bins)
            data.append(n_samples)
            data.append(noise_level)  
        data.append(freq) 
        data.append(freq_values)  
        data.append(x_noise)      
        data.append(bin_time)
        data.append(hist_time) 
        path = dir_path + os.path.sep + 'Files' + os.path.sep + 'spectrum'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = dir_path + os.path.sep + 'Files' + os.path.sep + 'spectrum' + os.path.sep + unit + '_' + tag + '.npy'
        dados_np = np.array( data , dtype='object' )                   
        np.save(filename,dados_np)        
        
    def histograme_compare(data,x2) :
        p_value, err_max = annomaly.compute_histogram_pvalue(data,x2)
        return p_value, err_max


    def compute_histogram_pvalue(data, x2):
        """
        Parameters
        ----------
        data : String of base comparation dates 
             
        x2 : Array of Float64
            SAmple_compare.

        dt : time sample interval [s]

        Returns
        -------
        p_value of hypothesis_test : Float
        
        It calculates the P-value between sample_base and sample_compare
        it will return the p-value as H1 or H0.

        """
        n_hist = int( data[0]) 
        dt = float(data[1])
        npto = len(x2)         
        p_value = np.zeros( n_hist )
        index_data = 5
        index_p = 0
        xf = ds.utilities.frequency_filter(x2,10,dt,type='highpass')      
        v = xf #1000*ds.utilities.time_integration(xf,dt)
        for k_ext in range( n_hist ) :
            freq = data[index_data]
            freq_values = data[index_data+1]
            Vc,Vs = ds.utilities.dft(v,dt,freq)
            V = Vc + 1j*Vs 
            dep = np.abs(V)+1e-12
            n = len(freq)
            harm_old = np.zeros(n).astype(complex)
            harm_new = np.zeros(n).astype(complex)
            zero_c = 0+0*1j
            for i in range(n) : 
                dif = 1 -  np.abs(freq_values[i])/(dep[i]) 
                #aux = (dep[i] - freq_values[i])/dep[i] 
                if dif < 1e-12 : 
                    harm_old[i] = zero_c
                    #harm_new[i] = V[i]
                else : 
                    harm_old[i] = freq_values[i]
                    harm_new[i] = dif*V[i]    
            v_clean = utilities.signal_dft_clean(v*hann(npto),dt,freq,harm_old,harm_new)    
            noise = data[index_data+2]
            bins_time = data[index_data+3] 
            h1_time = data[index_data+4]
            h2_time = utilities.hist_user(v_clean+noise, bins_time) 
            p_value[index_p] = ks_2samp(h1_time,h2_time, method ='asymp').pvalue
            index_data+=5
            index_p += 1 
        p_max = np.max(p_value) 
        err_max = np.sum(np.abs(h1_time-h2_time))
        return p_max, err_max # máximo dos mínimos      

    def signal_anomaly(data, dt) : 
        x = ds.utilities.detrend(data,dt,deg=2)
        erro = 10*np.log10( ds.timeparameters.rms(data) ) - 10*np.log10( ds.timeparameters.rms(x) )
        if np.abs(erro) > 3 : 
            return True
        return False
    
    def is_anomaly(dir_path,unit,tag,x,dt) : 
        n_bins = 0 
        data = annomaly.anomaly_load(dir_path,unit,tag)
        if data is None : 
            return -1, -1
        if annomaly.signal_anomaly(x,dt) : 
            return -1, -1
        err, err_max = annomaly.histograme_compare(data,x)
        i_sample = int(data[0])
        n_samples = data[3]                
        if i_sample < n_samples : # and err > .6 : 
            n_bins = int(data[2])   
            n_bins = data[3]
            noise_level = data[4]
            annomaly.iniatilize_histogram(dir_path,data,unit,tag,x,dt,i_sample+1,n_samples,n_bins=n_bins, noise_level=noise_level)
        #return err
        n_bins = data[2] 
        if err > .8 : 
            return 1, n_bins
        if err > .4 : 
            return 2, n_bins      
        return 3, n_bins
    

# -------------------------- Classe Para Estimar vida residual de ativos via monitoramento de vibrações e distribuição de weibull  -----
class rul_weibull : 


    def estima_k(x) :
        try : 
            dist = getattr(st,'weibull_min')
            paramt = dist.fit(x , floc=0)
            param=np.asarray(paramt).astype(float)
            k = param[0]
            if k < 10 : 
                return k   
            return 0     
        except :    
            return (np.mean(x)-np.min(x))/(np.std(x)+1e-6)

    def estima_SnS0(x, k, Sb, std) :
        #Sb = np.mean(x)
        #std = np.std(x) 
        Sn = Sb - k*std
        if Sn < 0 :
            Sn = 0
        invG = rul_weibull.inverse_gamma(k, initial_guess=1.0) 
        S0 = Sn+(Sb-Sn)*invG  
        if Sn > S0 : 
            Sn = .9*S0
        return Sn,S0     

    def estima_S0(k, Sn, Sb, std) :
        invG = rul_weibull.inverse_gamma(k, initial_guess=1.0) 
        s0 = Sn+(Sb-Sn)*invG  
        if Sn > s0 : 
            s0 = 1.01*Sn
        return s0     


    def estima_lamda(X,Y,k,S0,Sn) : 
        for i in range(len(X)) : 
            if Y[i] >= Sn : 
                break    
        teta = X[i:]         
        S = Y[i:] 
        if len(S) < 1 :
            return None  
        if S0 - Sn <1e-6 : 
            return None  
        try : 
            aux = ((S-Sn)/(S0-Sn))**k
            aux[aux>10] = 10
            y = 1+1/np.exp(aux)
        except : 
            return None     
        x = np.copy(teta) 
        aux = np.sum(x*y)/(np.sum(x*x)+1e-6) 
        if aux < 1e-2 : 
            lamb = 100
        else : 
            lamb = 1/aux
        return lamb        


    
    def inverse_gamma(y, initial_guess=1.0):
        inverse = fsolve(lambda x: gamma(x) - y, initial_guess, xtol=1e-3)[0]
        return inverse

    
    def life_20dB(data_x,data_y,y_mean) :
        x = np.copy(data_x)
        y = np.copy(data_y) 
        log_y = np.log(y/(y[0]+1e-8)) 
        x_rel = x-x[0]
        try : 
            p = np.polyfit(x_rel,log_y,1)
        except :
            return -1, -1
        erro = np.sum( np.abs( np.polyval(p,x_rel) - log_y  ) )  
        y_20dB = np.log(10/(y[0]+1e-8))  
        p = np.array(p)
        t_20dB = x[0] + (y_20dB -p[1])/(p[0]+1e-8)
        if t_20dB < x[0]: 
            return -1,-1
        return t_20dB, erro

    def Raquel_procedure(x,y, data_base, data_mean, data_std , mean1, std1) :
        TMMF = []  
        beta = rul_weibull.estima_k(data_base) 
        K = []
        if beta > .8 and beta < 10 : 
            K.append(beta)
        beta = (data_mean-np.min(y))/data_std 
        if beta > .8 and beta < 10 :    
            K.append(beta)
        beta = (mean1/std1) 
        if beta > .8 and beta < 10 :    
            K.append(beta)
        if len(K) > 1 : 
            K.append(np.mean(K))                
        Aw = .05 
        Pg = .95

        for i in range(len(K)) : 
            k = K[i]
            k = (np.mean(y)-np.min(y))/(np.std(y)+1e-6)
            Sn, S0 = rul_weibull.estima_SnS0(y,k,data_mean,data_std)
            lamb = rul_weibull.estima_lamda(x,y,k,S0,Sn) 
            if lamb is None or S0 - Sn < 1e-4 :
                continue 
            for i in range(len(x)-1) : 
                if y[i] >  Sn : 
                    break
            teta_a = lamb*(1 -Aw/Pg ) 
            if teta_a < 1 : 
                teta_a = 1
            TMMF.append(teta_a)
        return TMMF
    

    def sequencia(x,y, time_lag,std_err) : 
        # descobrie a maior sequencia decrescente de y 
        npto= len(y)  
        if npto == 0 : 
            return [[-1], [-1]]
        max = y[npto-1]
        if y[npto-2] > 2*max and y[npto-3] > max : # queda nos níveis de vibração 
            max = y[npto-1] + 3*std_err # tenta 3 sigma para cima 
            if y[npto-2] > 2*max and y[npto-3] > max :
               return [x[-1]], [y[-1]]
            y[npto-1] += 3*std_err
        I = [npto-1] 
        for i in range(npto-2,-1,-1) : 
            if y[i] < max : 
                I.append(i) 
                max = y[i] 
        I = np.array(I, dtype=int) 
        I = np.flip(I) 
        x = x[I] 
        y = y[I]
        if len(x) < 3 : 
            return x,y
        x_seq = [x[-1]] 
        y_seq = [y[-1]] 
        ii = 0 
        for i in range(len(x)-2,-1,-1) : 
            if x_seq[ii]-x[i] > time_lag : 
                break
            ii += 1 
            x_seq.append(x[i]) 
            y_seq.append(y[i]) 
        if ii > 0 :    
            x_seq = x_seq[::-1]
            y_seq = y_seq[::-1] 
        return x_seq,y_seq        



    def weibull(annomaly, tempo,sintoma, i_rul_ini, features_pred , time_pred, std_err) :
        TMMF = np.zeros(40)
        index = 0 
        time_lag = 4*np.mean( tempo[1:-1] - tempo[0:-2] )
        n_symptoms = sintoma.shape[1] 
        Index = len(tempo)
        annomaly = np.array(annomaly, dtype=int) 
        tempo = np.array(tempo, dtype=float)
        sintoma = np.array(sintoma[0:Index,:], dtype=float) 
        i_init = i_rul_ini - 3
        # ajustar os sintomas em uma exponencial mais 20 dB do sintoma 
        if i_init< 0 : 
            i_init = 0        
        y_mean_good = sintoma[ annomaly < 2,:] 
        ok_1 = True    
        if len(y_mean_good) < 2 : 
            y_mean_good = sintoma
            ok_1 = False       
        for k_ext in range(2) :
            x_anomaly_base = np.copy(tempo[i_init:])
            y_anomaly_base =  sintoma[i_init:,k_ext]
            x_anomaly,y_anomaly = rul_weibull.sequencia(x_anomaly_base,y_anomaly_base, time_lag,std_err[k_ext])
            if len(x_anomaly) < 3 :
                continue 
            if ok_1 : 
                y_mean = np.mean(y_mean_good[:,k_ext])
            else : 
                y_mean = np.min(y_mean_good[:,k_ext])/2 # valor mínimo menos 6 dB    
            t20, erro = rul_weibull.life_20dB(x_anomaly,y_anomaly,y_mean) 
            if erro > 0 : 
                if t20 < 1 : 
                    TMMF[index] = 1 + np.random.uniform(0,4)
                    index += 1
                else : 
                    TMMF[index] = t20    
                index += 1
        # Procedimento da Raquel 
        for k_ext in range(n_symptoms) : 
            dados = np.copy(sintoma[0:Index,k_ext])
            i_init = i_rul_ini - 1
            dados_rul =  np.copy(dados[i_init:])
            tempo_rul = np.copy(tempo[i_init:])
            try :
                st = std_err[k_ext]
                x_anomaly, y_anomaly = rul_weibull.sequencia(tempo_rul,dados_rul,time_lag, st) 
                if len(x_anomaly) > 3 : 
                    features_pred[k_ext,:] =  utilities.prev_param_IC(x_anomaly,y_anomaly, time_pred , st)
            except : 
                
                x_anomaly, y_anomaly = rul_weibull.sequencia(tempo_rul,dados_rul,time_lag,std_err[k_ext])    
                return None    
             
            yb = np.mean(sintoma[:,k_ext])
            ystd = np.std(sintoma[:,k_ext])
            i_min = np.argmin(sintoma[:,k_ext])
            y_min = sintoma[i_min,k_ext]
            sN = y_min
            for j in range(1) :
                if j == 0 : 
                    sy = ystd
                else : 
                    sy = ystd*np.random.uniform(.2,1)  
                k=(yb-y_min)/(sy+1e-6)
                s0=rul_weibull.estima_S0(k,sN, yb, sy)
                x_anomaly = np.concatenate(([tempo[i_min]],x_anomaly)) - tempo[i_min]
                y_anomaly = np.concatenate(([y_min],y_anomaly))+.0001
                lamb = rul_weibull.estima_lamda(x_anomaly,y_anomaly,k,s0,sN) 
                try :
                    if lamb is not None :
                        TMMF[index] = lamb + tempo[i_min]
                except : 
                    lamb = rul_weibull.estima_lamda(x_anomaly,y_anomaly,k,s0,sN)        
                index += 1  
        if len(x_anomaly) < 3 : # está começando
            TMMF *= 1.42   
        if np.sum( TMMF < tempo[-1]) > 0 :  
            TMMF[index] =  1.2*tempo[-1]  
            index += 1    
        TMMF = TMMF[0:index]     
        return TMMF




    def Ajusta_lamb(TMMF, beta = 2.5) :
        # API-581-2008 - 8.3.5 POF using the User Supplied MTTF
        aux = rul_weibull.inverse_gamma(1+1/beta) 
        lamb = TMMF/(aux+1e-8) 
        return beta,lamb    

    def Ajusta_lamb_TMMFs(TMMF, tempo_atual, beta = 2.5) :
        # API-581-2008 - 8.3.6 POF calculated using Specific Bundle History
        r = len(TMMF) 
        soma = tempo_atual**beta
        for tmmf in TMMF : 
            soma += tmmf**(1/beta) 
        lamb = (soma/r)**(1/beta)  
        if lamb < .5 : 
            return 2.5, .5
        return beta, lamb  
         


    def Ajusta_TMMF(TMMF) : 
        xo = np.sort(TMMF)
        npto = len(TMMF) 
        MR = (np.arange(1,npto+1,step=1)-.3)/(npto+.4) # median rank 
        Y = np.log(-np.log(1-MR))
        X = np.log(xo) 
        if np.sum(X) < .1 : 
            return 2.5,.5
        try :  
            par = np.array(np.polyfit(X,Y,1))
        except :  
            return 2.5,.5
        beta = par[0]
        lamb = np.exp(-par[1]/beta)
        if lamb < .5 : 
            return 2.5,.5
        return beta,lamb    
    

    def POF(TMMF, tempo) : 
        if np.sum(TMMF<2) == 0 : # nenhum parâmetro estourou 
            if len(TMMF) > 3 : 
                TMMF = ds.utilities.chauvenet(TMMF)[0]
        beta,lamb = rul_weibull.Ajusta_TMMF(TMMF)        
        if beta > 5 : 
            beta,lamb = rul_weibull.Ajusta_lamb_TMMFs(TMMF, tempo[-1], beta = 5) 
        try : 
            pof = 100*(1-np.exp(-(tempo/lamb)**beta))
        except : 
            pof = np.repeat(98,len(tempo))
        return pof    

    def POF_IC(TMMF, tempo, time_lag = 4) : 
        # https://reliawiki.com/index.php/Weibull_Confidence_Bounds problemas com uma matriz de covariância
        # Vou de Monte Carlo mesmo
        n_TMMF = len(TMMF) 
        if n_TMMF > 4 : 
            TMMF = ds.utilities.chauvenet(TMMF)[0]
        n_TMMF = len(TMMF) 
        dt = time_lag/int(time_lag-1) 
        t = np.arange(tempo,tempo+time_lag+dt,dt, dtype=float) 
        t = t[0:int(time_lag)]  
        u = np.zeros((200,len(t)))
        for k in range (200) : # Fazer um Monte Carlo para estimar beta e eta
            tmmf = np.copy(np.array(TMMF,dtype=float))
            tmmf *= np.random.uniform(.7,1.3,size=n_TMMF)
            beta,eta = rul_weibull.Ajusta_TMMF(tmmf) 
            aux  = 100*(1-np.exp(-(t/eta)**beta))
            aux[aux< 5] = 5
            aux[aux> 95] = 95
            u[k] = aux
        return u


    def POF_None(pof,tempo, IC = 10) : 
        # calcular IC 
        dt = 4/19 
        t = np.arange(tempo,tempo+4+dt,dt, dtype=float) 
        t = t[0:20]  
        t_life = 180 
        if tempo > t_life :
            t_life = 1.2*tempo
        beta = (pof[0]-5)*(.6-2.5)/(60-2.5)+2.5
        if beta < .6 : 
            beta  = .6 
        if beta > 2.5 : 
            beta = 2.5        
        lamb = tempo/((-np.log(1-pof[0]/100))**(1/beta)) 
        aux  = 100*(1-np.exp(-(t/lamb)**beta))
        aux[aux>90] = 90 
        aux[aux<5] = 5 

        IC_inf = 1-IC/100 
        IC_sup = 1+IC/100        
        pof_inf = IC_inf*aux
        pof_sup = IC_sup*aux
        pof_medio = aux
        return t,pof_inf, pof_medio, pof_sup


class rul : 
        # Entradas : 
        # path : caminho para salvar e carregar os dados históricos necessários para o funcionamento do RUL 
        # time_unit : unidade de tempo ( 'meses', 'dias', 'horas' )
        # time_lag : intervalo médio entre as medições em time_unit
        # production : Valores dos parâmetros de produção realcionadas com a anomalia  
        # noise_level : Nível de ruído para ajuste do detector de anomalias  

        # Retorna: 
        #  um dicionario, onde 
        #               ok: 'Falso': Não foi possível calcular o RUL 
        #                   'Atenção' : Não foi possível fazer previsão de RUL. Só tem o POF atual na msg 
        #                   'Verdadeiro' : RUL calculado com os intervalos de confiança repectivos
        #               msg: Mensagem texto 
        #               pof_80: matriz 3x6time_lag (float) - IC -80% para o pof nospróximos seis times_lag  
        #               pof_90: matriz 3x6time_lag (float) - IC -90% para o pof nospróximos seis times_lag 
        #               pof_95: matriz 3x6time_lag (float) - IC -95% para o pof nospróximos seis times_lag         
        #               annomaly : Null - se ok for True, msg avisando que o sistema está inicializando
        #                        [anomalia ref. sinal base, anomalia ref. sinal anterior] 
        #               features_pred: matriz 3xn_features x 3*3*6time_lag: direção x feature x 
        #               [pof_80[3*6time_lag],pof_80[3*6time_lag],pof_95[3*6time_lag]      

    def __init__(self, path,  time_unit,time_lag, production = None, noise_level = .5 ): 
        super( rul , self).__init__() 
        self.dirpath = path
        self.rpm = None 
        self.dt = None 
        self.dados = None 
        self.time_unit = time_unit
        self.time_lag = time_lag 
        self.n_times = time_lag
        self.time_pred = np.arange(1,self.n_times+1)
        self.production = production
        self.noise_level = noise_level
        self.n_dir = 0
        self.i_dir = 0 
        self.Index = 0
        self.mes = None 
        self.anomalia = None 
        self.anomalia_ant = None
        self.sinal = None 
        self.features = None
        self.TMMF = None
        self.dt = None 
        self.mes = None
        self.pof = None
        self.data = None
        self.n_features = 12
        self.n_TMMF = 20
        self.features_pred = np.zeros((3,self.n_features,9*self.n_times))

    def save_rul(self,df) : 
        filename = os.path.join( self.dirpath,'SPYAI_rul.npz' )  
        np.savez(filename, data=df.values, columns=df.columns)         

    def carrega_df_rul(self) : 
        filename = os.path.join( self.dirpath,'SPYAI_rul.npz' ) 
        if os.path.exists(filename) : 
            dados_numpy = np.load(filename, allow_pickle=True)
            df_rul = pd.DataFrame( dados_numpy['data']  ) 
            df_rul.columns = dados_numpy['columns']
        else : 
            df_rul = pd.DataFrame()
        return df_rul    

    def atualiza_df_TMMF_None( self, pof ) :  # atualiza df  
        try : 
            pof = -pof
            pof_max = np.max(pof)
            msg = "Probabilidade de falha atual é de %.1f %%"%(pof_max)
            valores = [self.data,'Atenção',msg, self.mes[-1],self.anomalia[-1],self.anomalia_ant] 
        except: 
            msg = "Erro nos Dados"
            valores = [self.data,'Erro',msg, self.mes[-1],self.anomalia[-1],0] 
        features = np.zeros(self.n_features) 
        features[0:len(self.features[-1])] = self.features[-1] 
        valores.append( features )      
        valores.append(np.zeros(3*self.n_times))
        valores.append(np.zeros(3*self.n_times))
        valores.append(np.zeros(3*self.n_times))
        aux = np.zeros(self.n_features) 
        aux[0] = 10000
        valores.append(aux)
        valores.append(np.array(self.features_pred, dtype=float).flatten())
        print( 'mensagem %s - rms = %.2f - 1X = %.2f'%(msg,features[1],features[0]) )
        return valores        

    def atualiza_df_TMMF( self, TMF ) :  # atualiza df  

        pof = rul_weibull.POF_IC(TMF, self.mes[-1], time_lag = self.time_lag)
        pof_medio = np.quantile(pof,[.5],axis=0)[0] 
        msg = "Probabilidade de falha nos próximos %d %s é de %.1f %%"%(self.time_lag,self.time_unit,pof_medio[-1])
        valores = [self.data,'Verdadeiro',msg, self.mes[-1],self.anomalia[-1],self.anomalia_ant]  
        features = np.zeros(self.n_features)
        features[0:len(self.features[-1])] = self.features[-1]
        valores.append( features )   
        # IC = 80%
        inf = np.quantile(pof,[.1], axis=0)[0]
        sup = np.quantile(pof,[.9], axis=0)[0] 
        aux = np.concatenate((inf,pof_medio,sup  ), axis=0)   
        valores.append(aux)
        # IC = 90%
        inf = np.quantile(pof,[.05],axis=0)[0]
        sup = np.quantile(pof,[.95],axis=0)[0] 
        aux = np.concatenate((inf,pof_medio,sup  ), axis=0)   
        valores.append(aux)
        # IC = 95%
        inf = np.quantile(pof,[.025],axis=0)[0]
        sup = np.quantile(pof,[.975],axis=0)[0] 
        aux = np.concatenate((inf,pof_medio,sup  ), axis=0)   
        valores.append(aux)
        aux = np.zeros(30) 
        aux[0:len(TMF)] = TMF
        valores.append(aux)
        valores.append(np.array(self.features_pred, dtype=float).flatten())
        print( 'mensagem %s - rms = %.2f - 1X = %.2f'%(msg,features[1],features[0]) )
        return valores        


    def iniciliza_rul( self, data ) :  # limpa o sinal e monta a linha 1 
        columns = ['data','ok','msg','mes','anomalia-base','anomalia-anterior','features','pof80%','pof90%','pof95%','TMMF','features_pred'] 
        valores = ['29/03/1960 04:40:00','Falso','Sinal Base', utilities.date_to_numeric(data,self.time_unit)  ,np.array([0,0,0],dtype=int),0]     
        valores.append(np.zeros(self.n_features)) 
        valores.append(np.zeros(3*self.n_times))
        valores.append(np.zeros(3*self.n_times))
        valores.append(np.zeros(3*self.n_times))
        aux = np.zeros(self.n_TMMF) 
        aux[0] = 10000
        valores.append(aux)
        valores.append(np.zeros(3*self.n_features*9*self.n_times))
            
        df = pd.DataFrame(columns=columns, )
        df.loc[-1] = valores 
        self.save_rul(df)
        return df      


    def Inicia_RUL( self, index, features) : 
        pof_max = -10
        sintoma = features[index]
        for i in range(2) :   
            dB_ratio = 10*np.log10(sintoma[i]+1e-8)
            if dB_ratio > 10.492 :
                dB_ratio = 10.492 
            if dB_ratio < 1.7609 : 
                dB_ratio = 1.7609 
            pof = 5+(dB_ratio-1.7609)*10.308   
            if pof > pof_max : 
                pof_max = pof
        return [-pof_max]  # negativo significa que não tem TMMF                       


    def estima_std(self, sinal ) : # Assume-se que o sintoma é sempre cresecente. Se ele diminui é variação 
                            # devido ao processo 
        soma = 0 
        cont = 0 
        for i in range( len(sinal) -1, 0 , -1 ) : 
            if sinal[i-1] > sinal[i] :
                if sinal[i-1] < 2*sinal[i] :  # variação máxima de 6 dB
                    soma += np.var(sinal[i-1:i+1]) 
                    cont += 1 
        if soma < 1e-8 :
            return sinal[-1]/3 
        else : 
            return np.sqrt(soma/cont)  



    def historico(self, features) : 
        index = self.Index
        self.anomalia_ant = -1 
        #ii = 0 
        if self.calcula_anomalia() < 0 : #não inicializou anomali ainda
            sinal = self.dados[self.i_dir] 
            dt = self.dt
            sinal_base = utilities.baseline_signal(sinal, dt, self.rpm)
            # plotar_graficos(sinal, sinal_base,dt, title = 'AV01VA - %s'%(data))
            #features = basic_defects.calc_symptoms_basic(sinal_base,dt,self.rpm) 
            n_bins =  int(1+3.322*np.log10(len(sinal_base))) #  Sturge’s Rule     
            annomaly.iniatilize_histogram(self.dirpath,[],'direction_base','%d'%(self.i_dir),sinal_base,dt,1,1,
                                     n_bins=n_bins,noise_level=self.noise_level)                  
            annomaly.iniatilize_histogram(self.dirpath,[],'direction_anterior','%d'%(self.i_dir),sinal_base,dt,1,1,
                                     n_bins=n_bins,noise_level=self.noise_level)   
            self.calcula_anomalia()
        if index < 3 : 
            return self.Inicia_RUL(index,features) 
        # criar apoio 
        ii = 0
        for i in range(len(self.mes)-1,0,-1) : 
            if self.features[i][0] < 1e-4 : 
                ii = i 
                break
            if ii > 10 : 
                ii = len(self.mes) - ii
                break 
            ii += 1
        if ii < 0 : 
            ii = 1    
        mes_aux = self.mes[ii:]
        i_rul_ini = ii      
        i_rul_cont = len(mes_aux)
        if i_rul_cont > 3 :
            n_sint = self.features.shape[1]
            std_err = np.zeros(n_sint) 
            for i in range( n_sint ) :  
                std_err[i] = self.estima_std( self.features[:,i] )
                #annomaly, tempo,sintoma, i_rul_ini, features_pred , time_lag, std_err
            TMF  = rul_weibull.weibull(self.anomalia[ii:,self.i_dir], mes_aux,features[ii:], 
                                       3, self.features_pred[self.i_dir,:,:] , self.time_pred, std_err)

            if TMF is None: 
                return self.Inicia_RUL(index,features)
            else :
                TMF [TMF<1] = 1
                return TMF.tolist()  
        return self.Inicia_RUL(index,features)            



    def calcula_anomalia(self) : 
        # calcular anomalia base e com relação ao anterior 
        aux = annomaly.is_anomaly(self.dirpath,  'direction_base','%d'%(self.i_dir), self.sinal,self.dt)[0]
        if aux < 0 :
            return -1 
        self.anomalia[-1,self.i_dir] = aux    
        aux, n_bins = annomaly.is_anomaly(self.dirpath, 'direction_anterior','%d'%(self.i_dir), self.sinal,self.dt)
        if aux > self.anomalia_ant : 
            self.anomalia_ant = aux 
        # atualiza anomalia anterior 
        annomaly.iniatilize_histogram(self.dirpath,[],'direction_anterior','%d'%(self.i_dir),self.sinal,self.dt,1,1,
                                     n_bins=n_bins,noise_level=self.noise_level)  
        return 1          

 
    def atualiza_base(self) : 
        TMF = np.array( self.TMMF, dtype=float)
        if len(TMF)-np.sum(TMF< 0) == 0 : # A sequencia não é crescente 
            valores = self.atualiza_df_TMMF_None(TMF) # Só tem POF
            self.df_rul.loc[len(self.df_rul)] = valores 
        else :
            valores = self.atualiza_df_TMMF(TMF[TMF>0])
            self.df_rul.loc[len(self.df_rul)] = valores 
        i = self.Index        
        pof80 = np.vstack(self.df_rul.iloc[i,7],dtype=float).reshape(3,-1)
        pof90 = np.vstack(self.df_rul.iloc[i,8],dtype=float).reshape(3,-1)
        pof95 = np.vstack(self.df_rul.iloc[i,9],dtype=float).reshape(3,-1)
        features_pred = np.array(self.df_rul.iloc[i,11], dtype=float).flatten()
        anomalia_base = np.max(self.df_rul.iloc[i,4]) 
        annom = [anomalia_base,self.df_rul.iloc[i,5]]
        dic = {'ok':self.df_rul.iloc[i,1] , 'msg': self.df_rul.iloc[i,2] , 'pof_80': pof80, 'pof_90': pof90,
                'pof_95': pof95, 'annomaly':annom, 'features_pred': features_pred }
        return dic
        #  dicionario, onde 
        #               ok: 'Falso': Não foi possível calcular o RUL 
        #                   'Atenção' : Não foi possível fazer previsão de RUL. Só tem o POF atual na msg 
        #                   'Verdadeiro' : RUL calculado com os intervalos de confiança repectivos
        #               msg: Mensagem texto 
        #               pof_80: matriz 3x6time_lag (float) - IC -80% para o pof nospróximos seis times_lag  
        #               pof_90: matriz 3x6time_lag (float) - IC -90% para o pof nospróximos seis times_lag 
        #               pof_95: matriz 3x6time_lag (float) - IC -95% para o pof nospróximos seis times_lag         
        #               annomaly : Null - se ok for True, msg avisando que o sistema está inicializando
        #                        [anomalia ref. sinal base, anomalia ref. sinal anterior] 
        #               features_pred: matriz 3xn_features x 3*3*6time_lag: direção x feature x 
        #               [pof_80[3*6time_lag],pof_80[3*6time_lag],pof_95[3*6time_lag]          
  
       


    def calcula_rul(self,data, rpm, F_aquis, dados_orig, features) :
        # data = data (dia/mes/ano hh:min:seg) da direção1 do ponto  
        # rpm : (float) rotação nominal do ponto de medição 
        # F_aquis: (float) lista com as frequências de aquisição de cada direção  
        # dados_orig: lista com numpy array (float) com os sinais de aceleração [g] [direção1, (direção2) e (direção3)]
        # features: lista comas as features utilizadas na classificação [direction1, (direction2),(direction3)]
        self.data = data
        n_dir = len(dados_orig) 
        dados = []     
        for i in range(n_dir) : # conferir se tem frequência de aquisição menor do que 2100 Hz 
            if F_aquis[i] < 2100 : # duplica o sinal para não dar problema no cálculo das features
                npto = len(dados_orig[i]) 
                sinal = np.zeros(2*npto) 
                ii = 0 
                for j in range(0,npto-1) : 
                    sinal[ii] = dados_orig[i][j]
                    sinal[ii+1] = .5*(dados_orig[i][j]+dados_orig[i][j+1]) 
                    ii += 2
                sinal[-1] = dados_orig[i][-1] 
                F_aquis[i] = 2*F_aquis[i]
            else : 
                sinal = dados_orig[i] 
            dados.append(sinal)
        self.dados = dados_orig                    
        # multiplica as features pelo rms do sinal 
        for i in range(n_dir) :
            rms = ds.timeparameters.rms(dados[i]) 
            for j in range(len(features[i])) :
                features[i][j] = features[i][j]*rms 
            
    
        features_org = [] 
        for i in range(n_dir) :
            for j in range(len(features[i])) :  
                features_org.append(features[i][j])         
        self.n_dir = n_dir 
        self.dt = 1/np.array(F_aquis, dtype=float) 
        
        self.rpm = rpm  
        df = self.carrega_df_rul() 
        if df.empty  : 
            self.iniciliza_rul( data) # limpa o sinal e monta a linha 1 
            df = self.carrega_df_rul() 
        for i in range(len(df)) : 
            if data == df.iloc[i,0] : # Já existe a data no banco de dados 
                pof80 = np.vstack(df.iloc[i,7],dtype=float).reshape(3,-1)
                pof90 = np.vstack(df.iloc[i,8],dtype=float).reshape(3,-1)
                pof95 = np.vstack(df.iloc[i,9],dtype=float).reshape(3,-1)
                anomalia_base = np.max(df.iloc[i,4]) 
                annom = [anomalia_base,df.iloc[5]]
                features_pred = np.array(self.df_rul.iloc[i,11], dtype=float).flatten()
                dic = {'ok':df.iloc[i,1] , 'msg': df.iloc[i,2] , 'pof_80': pof80, 'pof_90': pof90, 'pof_95': pof95, 
                    'annomaly':annom, 'features_pred': features_pred }
                return dic
        self.df_rul = df    
        self.anomalia = np.vstack(df.iloc[:,4].values, dtype=int).reshape(-1,3)
        self.anomalia =  np.concatenate( ( self.anomalia, np.array([[0,0,0]]) ) )
        self.mes = np.stack(df.iloc[:,3].values, dtype=float) 
        m = len(features_org)
        features_aux = np.zeros(self.n_features)
        features_aux[0:m] = features_org
        self.features= np.vstack(df.iloc[:,6].values, dtype=float).reshape(-1,self.n_features)

        self.features = np.concatenate( ( self.features, features_aux.reshape(1,-1)), axis=0 )
        self.Index = len(df)  
        vet_dt = self.dt

        mes = utilities.date_to_numeric(data) - df.iloc[0,3] 
        self.mes = np.concatenate((self.mes,[mes]), axis=0)
        self.TMMF = [0]
        ii = 0 
        for self.i_dir in range( self.n_dir ) :  
            self.dt = vet_dt[self.i_dir] 
            self.sinal = self.dados[self.i_dir]
            self.TMMF = self.TMMF + self.historico( self.features[:,ii:ii+len(features[self.i_dir])] )  
            ii += len(features[self.i_dir])
        self.TMMF = self.TMMF[1:]    
        dic = self.atualiza_base() 
        self.save_rul(self.df_rul)  
        return dic    

# path = '/home/lav/Leticia/SPYAI/SPYAI_01_08_25/Banco de dados/P100/SI1/EXA/EX0401/105.401/105.401ME/AV01/SPYAI_rul.npz'  
# rul = np.load(path, allow_pickle= True) 
# dados = rul['data']
# tempo = dados[1:,3]
# features = np.array(dados[1:,6].tolist() )  
# time_lag = 6
# IC_80,IC_90,IC_95 = utilities.prev_param_IC(tempo, features[:,0], time_lag ) 
i =1               
