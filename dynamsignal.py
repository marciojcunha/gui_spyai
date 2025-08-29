import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, signal
from scipy import interpolate
from scipy.signal import welch
from scipy.stats import norm
from scipy.signal.windows import hann
import datetime



class utilities : 
    
#    def __init__(self, master ) : 
#        super( utilities , self).__init__() 
#        self.sos = None

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


    def chauvenet(data):
        # Número de dados
        n = len(data)
        # Média e desvio padrão
        mean = np.mean(data)
        std_dev = np.std(data)
        # Z-scores
        z_scores = np.abs(data - mean) / std_dev
        # Critério de Chauvenet
        criterion = 1.0 / (2 * n)
        # Probabilidade de cada z-score
        probs = norm.sf(z_scores) * 2
        # Máscara para identificar os valores que não são outliers
        mask = probs >= criterion
        # Retornar dados filtrados
        return data[mask], mask
        

    def detrend( data, dt, deg = 1) : 
        # data = array com o sinal base
        #   dt = intervalo de aquisição [s]  
        # deg =  ordem do polinômio 
        npto = len(data) 
        t = np.linspace(0,(npto-1)*dt,npto) 
        coef = np.polyfit(t,data,deg=deg) 
        p = np.poly1d(coef) 
        x = np.copy(data) 
        x = x-p(t) 
        return x
    
    def autocorrelation( data, npto, dt, rpm) : 
        if npto > len(data) //2 :
            npto = len(data) //2
        freq_rot = rpm/60 
        freq_aquis = 1/dt 
        n_T = freq_aquis/freq_rot 
        i = int(npto/n_T) 
        if i < 1 :
            i = 1
        npto = int((i-1+1/np.sqrt(2))*n_T) 
        corr = np.zeros(npto) 
        for i in range(npto) : 
            x1=data[0+i:-1]
            n = len(x1) 
            x2 = data[0:n] 
            corr[i] = np.sum(x1*x2)/n 
        return corr



    
    def frequency_filter( data , Fc, dt,  poles = 4, type ='lowpass' ) :
        #   data = array com o sinal base para estimar a velocidade de rotação da máquina
        #   fc = float or two position array with cutoff frequencies 
        #   dt = intervalo de aquisição [s]  
        #   poles = number of irr filter poles 
        #   type{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}, 
        n=1
        freq_nyquist = .5/dt
        if type == 'bandpass' or type == 'bandstop' : 
            n = len(Fc) 
        if n ==1 : 
            while Fc > freq_nyquist : 
                Fc /= 2
            fc = 2*Fc*dt 
        else :
            fu = Fc[1] 
            while fu > freq_nyquist : 
                fu /= 2            
            fc = np.zeros(2)
            fc[0] = 2*Fc[0]*dt 
            fc[1] = 2*fu*dt  
        sos = signal.butter(poles, fc, btype=type, output='sos')       
        return signal.sosfilt(sos,np.copy(data))
    
    def dft(data,dt,freqs) : 
        npto = len(data)
        x = hann(npto)*np.copy(data)/(np.sqrt(2)*npto)
        t = np.linspace(0,(npto-1)*dt, num=npto)
        nfreq = len(freqs)
        a = np.zeros(nfreq) 
        b = np.zeros(nfreq)
        x0 = np.zeros(npto)
        for i in range(nfreq) :
            cos = np.cos(2*np.pi*freqs[i]*t) 
            sin = np.sin(2*np.pi*freqs[i]*t)
            a[i] =  np.trapz(cos*x)
            b[i] = np.trapz(sin*x)  
            if freqs[i] > 45.5 and freqs[i]< 72 :
                x0 = x0+a[i]*cos+b[i]*sin
        return a, b      
    
    def dft_dep(data,dt,freqs) : 
        npto = len(data)
        x = np.copy(data)/npto
        t = np.linspace(0,(npto-1)*dt, num=npto)
        nfreq = len(freqs)
        dep = np.zeros(nfreq) 
        for i in range(nfreq) : 
            cos = np.cos(2*np.pi*freqs[i]*t) 
            sin = np.sin(2*np.pi*freqs[i]*t)
            dep [i] =  (np.trapz(cos*x)**2 + np.trapz(sin*x)**2 ) 
        return 2*dep      

    def dep_welch(data, dt, nfft) : 
        if nfft > len(data) : 
            nfft = len(data)
        freq,DEP = welch(data,1/dt,nperseg=nfft,noverlap=int(nfft//2),nfft=nfft,scaling='spectrum')
        return freq,DEP

    def dep(data,dt,freqs) : 
        # retorno 
        nfft = len(data)
        if nfft > 16384:
            nfft = 16384
        freq,DEP = welch(data,1/dt,nperseg=nfft,noverlap=int(nfft//2),nfft=nfft,scaling='spectrum')
        nfreq = len(freqs)
        dep = np.zeros(nfreq) 
        df = freq[2]-freq[1]
        for i in range(nfreq) : 
            index = int(freqs[i]/df)
            if index < 2: 
                index = 2
            if index > len(DEP)-3 : 
                index = len(DEP)-3    
            dep[i] = np.sum(DEP[index-2:index+2])
        dep = np.sqrt(dep)
        return dep   
    
    def fft_spectrum( x , scaling = 'rms'  ) : 
    #   x : np  float array 
    #   scaling : 'peak' , 'peak-peak' , 'rms'  
        npto = len(x)
        X = np.abs(np.fft.fft(2*hann(npto)*x/npto))[0:int(npto/2)]
        if scaling == 'peak' : 
            return 2*X 
        if scaling == 'peak-peak' : 
            return 4*X 
        return np.sqrt(2)*X         


    def time_integration( data , dt ): 
        #   data = array com o sinal base para estimar a velocidade de rotação da máquina
        #   dt = intervalo de aquisição [s]          
        x = np.copy(data.flatten())
        sos = signal.butter(2, 10*dt, btype='highpass', output='sos') 
        x = signal.sosfilt(sos,x)
        x = signal.sosfilt(sos,x)

        npto = len(data) 
        xp = np.zeros(npto) 
        for i in range(1,npto): 
            xp[i] = xp[i-1]+(x[i]+x[i-1]) 
        for i in range(npto-2,-1,-1): 
            xp[i] = xp[i+1]+(x[i]+x[i+1])                    
        xp *= (dt/2)
        xp = signal.sosfilt(sos,xp)
        xp = signal.sosfilt(sos,xp)        
        return xp     

    def _rpm_spectrum_old(data, dt, rpm_search_interval ) : 
        x = np.copy(data).flatten()
        m = len(x)
        if m > 4*8192: 
            m = int(4*8192)
            x = x[0:m]
        else : 
            npto = 2 
            while npto < m : 
                npto += npto 
            xr = np.zeros(npto) 
            xr[0:m] = x 
            xr[m:] = x[0:npto-m] 
            x = xr
        freq_inf = rpm_search_interval[0]/60
        freq_sup = rpm_search_interval[1]/60    
        npto = len(x)      
        xr = np.zeros(npto) 
        index = 0
        fc = 1/dt
        freq_max = 8*freq_sup
        if freq_max > fc : 
            freq_max = fc+1
        npto = len(x) 
        npto2 = npto // 2
        while fc > freq_max :
            if index == 5 : 
                break 
            jan = hann(npto2)
            x = utilities.frequency_filter(x,.4*fc, dt) 
            xr[0:npto2] = jan*x[0:npto:2] 
            xr[npto2:] = jan*x[1:npto:2] 
            x = xr
            fc /=2 
            index += 1
        df = fc/npto 
        while df > .1 : 
            x = np.concatenate([x,x])
            npto += npto 
            df = fc/npto
        X = np.abs(np.fft.fft(2*x/npto)) 
        i_freq_min = int(freq_inf/df) 
        i_freq_max = int(freq_sup/df) 
        X_aux = np.zeros(i_freq_max)
        for i in range(i_freq_min,i_freq_max) :
            X_aux[i] = X[i]*X[2*i]*X[3*i] 
        i_freq = np.argmax(X_aux) 
        freq_rotation = i_freq*df
       
        return freq_rotation*60  

    def _rpm_spectrum(data, dt, rpm_search_interval ) : 
        x = np.copy(data).flatten()
        freq_inf = rpm_search_interval[0]/60
        freq_sup = rpm_search_interval[1]/60    
        npto = len(x)      
        xr = np.zeros(npto) 
        index = 0
        df = 1/(npto*dt)
        npto = len(x) 
        npto2 = int(npto / 2 )
        X = np.abs(np.fft.fft(2*hann(npto)*x/npto))[0:npto2]
        X = 10*np.log10((2*X*X+1e-12)/1e-12) 
        i_freq_min = int(freq_inf/df) 
        i_freq_max = int(freq_sup/df) 
        X_aux = np.zeros(i_freq_max)
        if i_freq_min == 0 : 
            i_freq_min = 1
        for i in range(i_freq_min,i_freq_max) :            
            X_aux[i] = np.sum( X[3*i:-1:i])/((npto2-3*i)/i) 
        i_freq = np.argmax(X_aux) 
        freq_rotation = i_freq*df
       
        return freq_rotation*60  


    def _rpm_envelope(data, dt, rpm_search_interval, freq_high ) : 
        x = np.copy(data).flatten()
        m = len(x)
        if m > 4*8192: 
            m = int(4*8192)
            x = x[0:m]
        else : 
            npto = 2 
            while npto < m : 
                npto += npto 
            xr = np.zeros(npto) 
            xr[0:m] = x 
            xr[m:] = x[0:npto-m] 
            x = xr
        fc = 1/dt    
        if freq_high > fc/4 : 
            freq_high = fc/4
        x = utilities.frequency_filter(x,freq_high,dt,poles=6,type='highpass') 
        x = np.abs(x)                
        freq_inf = rpm_search_interval[0]/60
        freq_sup = rpm_search_interval[1]/60    
        npto = len(x)      
        xr = np.zeros(npto) 
        index = 0
        freq_max = 8*freq_sup
        if freq_max > fc/2 : 
            freq_max = fc/2
        x = utilities.frequency_filter(x,freq_max,dt) 
        npto = len(x) 
        npto2 = npto // 2
        while fc > freq_max :
            if index == 5 : 
                break 
            jan = hann(npto2)
            x = utilities.frequency_filter(x,.4*fc, dt) 
            xr[0:npto2] = jan*x[0:npto:2] 
            xr[npto2:] = jan*x[1:npto:2] 
            x = xr
            fc /=2 
            index += 1
        df = fc/npto 
        while df > .1 : 
            x = np.concatenate([x,x])
            npto += npto 
            df = fc/npto
        X = np.abs(np.fft.fft(2*x/npto)) 
        i_freq_min = int(freq_inf/df) 
        i_freq_max = int(freq_sup/df) 
        X_aux = np.zeros(i_freq_max)
        for i in range(i_freq_min,i_freq_max) :
            X_aux[i] = X[i]*X[2*i]*X[3*i] 
        i_freq = np.argmax(X_aux) 
        freq_rotation = i_freq*df
       
        return freq_rotation*60  

    def _rpm_convolution(data , dt, rpm_search_interval):
        # Detrend 
        x = np.copy(data).flatten()
        m = len(x)
        if m > 8192: 
            m = 8192
            x = x[0:m]
        else : 
            npto = 2 
            while npto < m : 
                npto += npto 
            xr = np.zeros(npto) 
            xr[0:m] = x 
            xr[m:] = x[0:npto-m] 
            x = xr            
        freq_inf = rpm_search_interval[0]/60
        if freq_inf < 10 : 
            freq_inf = 10 
        freq_sup = rpm_search_interval[1]/60
        x = utilities.frequency_filter(x,freq_inf,dt,type='highpass')    
        freq_aquis = 1/dt
        freq_max = 4*freq_sup 
        fs = freq_aquis
        npto = len(x) 
        npto2 = npto //2 
        xf = np.zeros(npto)
        n_sig = 1
        ii = 1
        while fs > freq_max : 
            if npto/n_sig < 512 or ii == 5 : 
                break
            xf = utilities.frequency_filter(x,.4*fs,dt,poles=6) 
            x[0:npto2] = xf[0:npto:2] 
            x[npto2:] = xf[1:npto:2]
            fs /=2 
            n_sig *=2
            ii += 1
        x = x.reshape(n_sig,-1)
        freq_rotation = np.zeros(n_sig)
        T = 8*m*dt
        df = 1/T 
        pos_min = int( freq_inf / df )
        pos_max = int( freq_sup/df)
        m = len(x[0])
        for i_sig in range( n_sig ) : 
            # Calcular o vetor de autocorrelação         
            r = np.correlate(x[i_sig,:], x[i_sig,:], mode="full")
            r = r[m:0:-1]
            # Aplicar janela exponencial 4 de forma a garantir zeros no final de r
            exp4 = 1.0 / np.exp(np.linspace(0, (m - 1) * dt, num=m) * 4)
            r = r * exp4
            # Adicionar zeros em r visando aumentar a resolução em frequência
            y = np.zeros(16 * m)
            y[0:m] = r
            # Calcular a fft do sinal transiente
            X = np.abs(fftpack.fft(y))
            # Fazer produtorio entre os harmônicos para amenizar influência de componentes indesejáveis
            for i in range( pos_min,pos_max ) : 
                X[i] = X[i]*X[2*i]*X[3*i]
            index_max = pos_min+np.argmax(X[pos_min:pos_max])
            freq_rotation[i_sig] = (index_max)*df
        rotation_frequency_estimation = np.mean(freq_rotation)        
        return rotation_frequency_estimation*60   
        



    def rpm_estimation(data , dt, rpm_search_interval, estimator = 'spectrum', freq_envelope = 500):
        # Entradas :
        #   data = array com o sinal base para estimar a velocidade de rotação da máquina
        #   dt = intervalo de aquisição [s] 
        #   rpm_search_interval = string com o valor inferior e superior para busca da velocidade de rotação da máquina [rpm]
        # retorna: 
        #   current_rotation = velocidade de rotação estimada para a máquina [rpm] 
        
        if estimator == 'spectrum' : 
            return utilities._rpm_spectrum(data, dt, rpm_search_interval )
        if estimator == 'envelope' : 
            return utilities._rpm_envelope(data, dt, rpm_search_interval, freq_envelope )        
        if estimator == 'convolution' : 
            return utilities._rpm_convolution(data, dt, rpm_search_interval )        
        return utilities._rpm_spectrum(data, dt, rpm_search_interval )
    
    def changedeltat(dados, dt_old, dt_new ):
        # dados : série temporal adquirida com uma frequência de aquisição fa Hz
        # dt_old = intervalo de aquisição atual
        # dt_new = intervalo de aquisição de interesse
        # retorna:
        # t_new: array com o eixo do tempo  
        # dados_new: array com o dados reamostrado 

        N = len(dados) 
        N_old = N
        t_new    = np.linspace(0,(N-1)*dt_new, N )
        t_max = t_new[-1]
        ok = True
        x = np.copy(dados) 
        while ok: 
            t_old    = np.linspace(0,(N-1)*dt_old, N ) 
            if t_old[-1] < t_max : 
                x      = np.concatenate((x,dados) , axis=0)   
                N = len(x)  
            else : 
                ok = False       
        tck      = interpolate.splrep(t_old, x, s=0)
        t_new    = np.linspace(0,(N-1)*dt_new, N )
        x_new    = interpolate.splev(t_new, tck, der=0)
        if len(t_new>N_old) : 
             t_new = t_new[0:N_old]
             x_new = x_new[0:N_old]
        return t_new, x_new

    def dominio_angular(data, dt, nominal_rotation = -1, npto = 256) :
        if nominal_rotation<0 : # utilizar o estimador de velocidade de rotação  
            rotation_speed = utilities.rpm_estimation(data , dt, nominal_rotation = nominal_rotation)   
        else : # o usuário passou a velocidade de rotação real da máquina 
            rotation_speed = nominal_rotation
        x = np.copy(data)
        n = len(x) 
        if n < npto :
            return x
        T_rot = 2*np.pi/rotation_speed
        delta_n = int(T_rot/dt)
        dt_new = T_rot/delta_n
        #dt_new = T_rot/npto
        t_new,x_new = utilities.changedeltat(x, dt, dt_new )
        x_angular = np.zeros(delta_n)         
        for i in range(0,delta_n) : 
            x_angular[i] = np.mean(x_new[i:n:delta_n])  
        while delta_n < npto: 
            x_angular = np.concat((x_angular,x_angular))   
            delta_n *= 2 
        dt_angular = T_rot/npto    
        t_new,x_new = utilities.changedeltat(x_angular, dt_new, dt_angular )
        return t_new[0:npto], x_new[0:npto]  

    def reshape(signal, new_npto) : 
        npto = len(signal) 
        xr = np.copy(signal)
        if npto < new_npto : 
            xr = xr.reshape(1,-1)
        else : 
            npto = int(int( npto/new_npto)*new_npto)
            xr = xr[0:npto] 
            xr = xr.reshape(-1,new_npto)
        return xr     
    
    def ball_bearing_plot(signal, dt, BPFI, BPFO, BSF, FTF, rpm = -1) : 
        if rpm < 0 : 
            rpm = utilities.rpm_estimation(signal,dt,3600) 
        freq_bas = rpm/60 
        nfft = 16384 
        while nfft > len(signal) : 
            nfft /= 2
        nfft = int(nfft) 
        nfft2 = int(nfft/2)
        freq,dep = welch(signal,1/dt, nperseg=nfft, noverlap=nfft2,nfft=nfft,scaling='spectrum')         
        dep_inf = np.min(dep)
        dep_sup = np.max(dep)
        n = len(freq)

        fault_freq = [freq_bas*BPFI,freq_bas*BPFO,freq_bas*BSF,freq_bas*FTF] 
        fault_plot = -.1*np.ones((4,n)) 

        for i in range(4) : 
            for j in range(0,5) : 
                index = np.argmin(np.abs(freq-(j+1)*fault_freq[i]))
                fault_plot[i,index] = dep_sup    
        plt.plot( freq, dep, freq,fault_plot[0,:],freq, fault_plot[1,:], freq, fault_plot[2,:], freq, fault_plot[3,:])
        freq_max = 10*np.max(fault_freq) 
        if freq_max > freq[-1] : 
            freq_max = freq[-1] 
        plt.axis([5, freq_max, dep_inf, dep_sup])
        plt.grid() 
        plt.xlabel('Frequência [Hz]') 
        plt.ylabel('DEP [dB, ref = 1e-12]')
        plt.legend( ['DEP','BPFI','BPFO','BSF','FTF'] ) 
        plt.show()

    def signal_defect_detector(Sinal, dt):  
        sinal = np.copy(Sinal)
        rms = timeparameters.rms(sinal)  
        sinal_1 = utilities.detrend(sinal, dt) 
        rms_1 = timeparameters.rms(sinal_1)   
        if np.abs( (rms_1-rms)/(rms_1+1e-8) ) > .1 : 
            return True 
        # A = utilities.fft_spectrum(sinal) 
        # A_min = np.min(A) 
        # I = A[0:20] < A_min 
        # if np.sum(I) < 1 : 
        #     return True
        return False

       



class timeparameters : 
    def rms(data) : 
        rms = np.sqrt( np.sum(data*data)/len(data) )
        return rms
    
    def peak(data ) : 
        return( np.max(abs(data)) ) 

    def peak_sync( data, dt, T_lag = 10, nominal_rotation = -1 ) :
             
        if nominal_rotation<0 : # utilizar o estimador de velocidade de rotação  
            rotation_speed = utilities.rpm_estimation(data , dt, nominal_rotation = nominal_rotation)   
        else : # o usuário passou a velocidade de rotação real da máquina 
            rotation_speed = nominal_rotation 
            
        npto = len(data)   
        T = 1/(rotation_speed/60) 
        i_T = int( T_lag*T/dt ) # avaliar pico em pelo menos dez periodos de tempo 
        peak_mean = [] 
        
        i_inf = 0
        for i in range(i_T,npto,i_T) : 
            peak_mean.append( timeparameters.peak( data[i_inf:i] ) ) 
            i_inf = i
        return( np.mean( peak_mean ) ) 
    
    def peak_to_peak(data ) : 
        return( np.max(abs(data)) + np.min(abs(data)) ) 

    def crest_factor( data ) : 
        cf = timeparameters.peak(data)/timeparameters.rms(data)
        return cf

    def crest_factor_sync( data, dt, T_lag = 1, nominal_rotation = -1  ) : 
        cf = timeparameters.peak_sync(data,dt, T_lag= T_lag,nominal_rotation=nominal_rotation  )/timeparameters.rms(data) 
        return cf 
    
    def k_factor( data ) : 
        k = timeparameters.rms(data)*timeparameters.peak(data)
        return k

    def skewness( data ) : 
        data_mean = np.mean(data) 
        npto = len(data)
        sk = np.sum( ( (data - data_mean)**3)/npto )  / timeparameters.rms(data)**3
        return sk 

    def kurtosis( data ) : 
        data_mean = np.mean(data) 
        npto = len(data)
        k4 = np.sum( ( (data - data_mean)**4)/npto )  / timeparameters.rms(data)**4
        return k4 

    def k6( data ) : 
        data_mean = np.mean(data) 
        npto = len(data)
        k6 = np.sum( ( (data - data_mean)**6)/npto )  / timeparameters.rms(data)**6
        return k6
     
    def syncronic_mean( data , dt, npto = 256, nominal_rotation = 3600  ) : 
        data_sync = utilities.dominio_angular(data,dt,nominal_rotation, npto)
        return data_sync    
    
    def cepstrum( data , dt, npto = 4096  ) :
        #dep = signal.welch(data, fs=1/dt, window='hann', nperseg=npto, noverlap=None, nfft=npto)[1]
        dep = 2*np.log(np.fft.fft(data))
        ceps = np.abs(np.fft.ifft ( dep ) )**2
        ceps = np.flip(ceps)
        return ceps

    def envelope( data, fc, dt = None): 
        if dt is None :
            x_a = signal.hilbert( data ) 
            amp = np.abs(x_a) 
            phase = np.arctan2( np.imag(x_a), np.real(x_a) )
            return amp, phase
        x_f = utilities.frequency_filter(data,fc,dt, poles=6, type='highpass')
        cut_freq = 1/(4*dt)
        x_f = utilities.frequency_filter(x_f,cut_freq,dt, poles=6, type='lowpass')
        x_a_r = np.abs(x_f)
        return x_a_r        

class ballbearingfaults : 


    def rms_righ(signal,dt,Fc= 500) : 
        x = np.copy(signal) 
        cut_freq = 4/dt 
        if cut_freq < Fc  : 
            fc = cut_freq
        else : 
            fc = Fc    
        xf = utilities.frequency_filter(x,fc,dt,type='highpass')
        return timeparameters.rms( xf )
    

    def FC( signal , dt, rpm ) : 
        peak = timeparameters.peak_sync(signal,dt,nominal_rotation=rpm) 
        fc = 20*np.log10( (peak+1e-12 ) / (timeparameters.rms(signal)+1e-12) )
        return fc
    
    def envelope(signal, dt , fc = 500) : 
        se = timeparameters.envelope(signal, dt = dt, freq_corte = fc)
        se_rms = timeparameters.rms(se) 
        se_amp = timeparameters.peak_to_peak(se) 
        return se_rms, se_amp, se

    def BPFO(signal, dt, BPFO, rpm, estimator = 'DEP') : 
        freq_nominal = rpm/60 
        BPFO_freq = freq_nominal*BPFO 
        freq = [freq_nominal, 2*freq_nominal, BPFO_freq, 2*BPFO_freq, 3*BPFO_freq,4*BPFO_freq] 
        if estimator == 'DFT':
            dep = ballbearingfaults.dft_dep(signal,dt,freq)
        else :
            dep = ballbearingfaults.dep(signal,dt,freq)     
        return np.sum(np.sqrt(np.sum(dep)))
    
    def BPFI(signal, dt, BPFI, rpm , estimator = 'DEP') : 
        freq_nominal = rpm/60 
        BPFI_freq = freq_nominal*BPFI 
        freq_bas = [freq_nominal, 2*freq_nominal, BPFI_freq, 2*BPFI_freq, 3*BPFI_freq, 4*BPFI_freq]
        if estimator == 'DFT' :
            dep_bas = ballbearingfaults.dft(signal,dt, freq_bas)
        else : 
            dep_bas = ballbearingfaults.dep(signal,dt, freq_bas)    
        freq = []     
        for i in range(1,5) : 
            aux = i *BPFI_freq
            for j in range(-3,4,1) : 
                if j != 0 :
                    freq.append( (aux+j*freq_nominal) )
        if estimator == 'DFT' :
            dep_lateral = ballbearingfaults.dft_dep(signal,dt,freq)
        else : 
            dep_lateral = ballbearingfaults.dep(signal,dt,freq)    
        dep = np.sqrt( np.sum(dep_bas) + np.sum(dep_lateral)/6  )
        return dep    

    def BSF(signal, dt, BSF, FTF, rpm, estimator = 'DEP') : 
        freq_nominal = rpm/60 
        BPFI_freq = freq_nominal*BSF 
        FTF_freq = freq_nominal*FTF
        freq_bas = [freq_nominal, 2*freq_nominal, BPFI_freq, 2*BPFI_freq, 3*BPFI_freq, 4*BPFI_freq]
        if estimator == 'DFT' :
            dep_bas = ballbearingfaults.dft_dep(signal,dt, freq_bas)
        else : 
            dep_bas = ballbearingfaults.dep(signal,dt, freq_bas)    
        freq = []     
        for i in range(1,5) : 
            aux = i * BPFI_freq
            for j in range(-3,4,1) : 
                if j != 0 :
                    freq.append((aux+j*FTF_freq))
        if estimator == 'DFT' :
            dep_lateral = ballbearingfaults.dft_dep(signal,dt,freq)
        else : 
            dep_lateral = ballbearingfaults.dep(signal,dt,freq)    
        dep = np.sqrt( np.sum(dep_bas) + np.sum(dep_lateral)/6 )
        return dep    
    
    def FTF(signal, dt, FTF, rpm, estimator = 'DEP') : 
        freq_nominal = rpm/60 
        BPFO_freq = freq_nominal*FTF 
        freq = [freq_nominal, 2*freq_nominal, BPFO_freq, 2*BPFO_freq, 3*BPFO_freq,4*BPFO_freq] 
        if estimator == 'DFT':
            dep = ballbearingfaults.dft_dep(signal,dt,freq)
        else :
            dep = ballbearingfaults.dep(signal,dt,freq)     
        return np.sum(np.sqrt( np.sum(dep )))

class journalbearingsfaults: 

    def low_order_rms( signal , dt, rpm) : 
        # https://philarchive.org/archive/PINIOB
        freq_bas = rpm/60 
        freq_max = 5.5*freq_bas 
        fc = 1/dt 
        if fc/2 < freq_max : 
            freq_max = fc/2  
        x = np.concatenate((signal,signal)) 
        x = np.concatenate((x,x)) 
        nfft = len(x/2)
        freq,DEP = welch(x, fs=fc, nperseg=nfft, noverlap=int(nfft/2),nfft=nfft, scaling='spectrum')
        df = freq[2]-freq[1] 
        i_10 = int(10/df) 
        DEP[0:i_10] = 0 
        i_5h = int(5.5*freq_bas/df) 
        DEP[i_5h:] = 0         
        dep = 0 
        i_freq = np.argmin( np.abs(freq-freq_bas) )
        dep += np.sum(DEP[i_freq-2:i_freq+2]) 
        DEP[i_freq-2:i_freq+2] = 0 
        for i in range(5) : 
            i_freq = np.argmax(DEP) 
            dep += np.sum(DEP[i_freq-2:i_freq+2]) 
            DEP[i_freq-2:i_freq+2] = 0             
        dep = np.sqrt(dep) 
        return dep    

    def harm5( signal , dt, rpm) : 
        # https://philarchive.org/archive/PINIOB
        freq_bas = rpm/60 
        fc = 1/dt 
        freq_max = 5.5*freq_bas 
        fc = 1/dt 
        if fc/2 < freq_max : 
            freq_max = fc/2          
        x = np.concatenate((signal,signal)) 
        x = np.concatenate((x,x)) 
        npto = int(len(x)/2) 
        nfft = npto
        if nfft > npto: 
            npto = nfft    
        freq,DEP = welch(x, fs=fc, nperseg=nfft, noverlap=int(nfft/2),nfft=nfft, scaling='spectrum')
        df = freq[2]-freq[1] 
        i_10 = int(10/df) 
        DEP[0:i_10] = 0 
        i_5h = int(5.5*freq_bas/df) 
        DEP[i_5h:] = 0
        dep = np.zeros(5) 
        i_freq = np.argmin( np.abs(freq-freq_bas) )
        dep[0] = np.sum(DEP[i_freq-2:i_freq+2]) 
        DEP[i_freq-2:i_freq+2] = 0 
        for i in range(1,5) : 
            i_freq = np.argmax(DEP) 
            dep[i] = np.sum(DEP[i_freq-2:i_freq+2]) 
            DEP[i_freq-2:i_freq+2] = 0             
        dep = np.sqrt(dep) 
        return dep    

    def harm5_normalized( signal , dt, rpm) : 
        dep = journalbearingsfaults.harm5( signal, dt, rpm)
        dep_norm = dep[1:]/(dep[0]+1e-12)
        return dep_norm


    def oil_wirl_whip( signal, dt, rpm ) : 
        freq_bas = 2*rpm/60 
        fc = 1/dt 
        x = np.copy(signal) 
        npto = len(x) 
        npto2 = int(npto/2)
        xr = np.zeros(npto) 
        while fc > freq_bas : 
            x = utilities.frequency_filter(x,.4*fc, dt) 
            xr[0:npto2] = x[0:npto:2] 
            xr[npto:] = x[1:npto:2] 
            x = xr
            fc /=2 
        nfft = 2048
        if nfft > npto: 
            npto = nfft    
        freq,dep = welch(x, fs=fc, nperseg=nfft, noverlap=int(nfft/2),nfft=nfft, scaling='spectrum') 
        i_freq_inf = np.argmin(np.abs(freq-.35*freq_bas)) 
        i_freq_sup = np.argmin(np.abs(freq-.45*freq_bas))   
        if i_freq_sup == i_freq_inf : 
            i_freq_sup += 1 
        dep = np.sum(dep[i_freq_inf:i_freq_sup]) 
        return np.sqrt(dep)      






'''
Fs = 8192
t = np.linspace(0,1-1/8192,8192)
x = np.sin(2*np.pi*60.28*t) +np.sin(2*np.pi*61*t+.3)+np.random.normal(0,.1,size=8192)
X = utilities.fft_spectrum(x,'peak') 
plt.plot(X[0:200])
plt.show()
X = utilities.fft_spectrum(x,'peak-peak') 
plt.plot(X[0:200])
plt.show()
X = utilities.fft_spectrum(x,'rms') 
X2 = welch(x , scaling='spectrum', nperseg=8192)[1] 
plt.plot(X[0:200])
plt.plot(np.sqrt(X2))
plt.show()

t_new,x_new = timeparameters.syncronic_mean(x,1/Fs,npto=360,nominal_rotation=59.5*2*np.pi)
plt.plot(t_new,x_new)
plt.show()
i=1 
'''
                    
