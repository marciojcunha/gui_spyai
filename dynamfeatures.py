import dynamsignal as ds
import numpy as np
from scipy.signal.windows import hann
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

class Features_classification:
    def __init__(self, fs, rpm, **kwargs):
        self.num_dentes = kwargs['num_dentes_pinhao']
        self.freq_rotac_pinhao=  kwargs['freq_rotac_pinhao']
        self.num_pas= kwargs['num_pas']
        self.num_polos= kwargs['num_polos']
        self.num_barras_rotor=kwargs['num_barras_rotor']
        bpfi= kwargs['bpfi']
        bpfo= kwargs['bpfo']
        bsf = kwargs['bsf']
        ftf = kwargs['ftf']
        bpf = kwargs['bpf']
        self.freq_rolamento = [bpfo,bpfi,bsf,ftf]
        self.fs = float(fs)
        self.dt = 1 / self.fs
        self.rpm = float(rpm)
        self.rps = self.rpm / 60.0
        self.features = None
        self.dt_sinc = self.dt # vai mudar em self.rpm_sinc_data
        self.npto = None    
        self.npto2 = None
        self.sinal_sinc = None   
        self.DEP = None
        self.DEP2 = None
        self.df_sinc = None 
        self.freq_sinc = None  
        self.DEPV = None 
        self.DEPV2 = None     
        
    
    
    def rpm_sinc_data(self,data) : 
        f0 = self.rpm/60 
        npto_old = len(data) 
        df_old = 1/(npto_old*self.dt)
        i = int(f0/df_old) 
        df = f0/i 
        dt_new = 1/(df*npto_old) 
        sinal = np.copy(data)
        x = ds.utilities.changedeltat(sinal,self.dt,dt_new)[1] 
        self.dt_sinc = dt_new
        return x

    def calcular_features(self, sinal, normalized=True):
        npto = len(sinal) 
        if npto%2 != 0 : 
            sinal = np.concatenate( (sinal,[sinal[0]]), axis = 0 )
        if normalized:
            sinal /= (ds.timeparameters.rms(sinal)+1e-8)    
        self.sinal = sinal
        self.npto = len(sinal) 
        self.npto2 = int(npto/2)
        self.dt_sinc = self.dt # vai mudar em self.rpm_sinc_data
        self.sinal_sinc = self.rpm_sinc_data(self.sinal)   
        self.DEP = ds.utilities.fft_spectrum( self.sinal_sinc )
        self.DEP2 = self.DEP*self.DEP
        self.df_sinc = 1/(self.npto*self.dt_sinc) 
        self.freq_sinc = self.df_sinc*np.arange(0,self.npto2)  
        i_10 = int(10/self.df_sinc) 
        xf = ds.utilities.frequency_filter(self.sinal_sinc, 10, self.dt_sinc, type='highpass')
        vi = 10000 * ds.utilities.time_integration(xf, self.dt_sinc)
        dep_v = ds.utilities.fft_spectrum(vi)
        vi[0:i_10] = 0 

        # dep_v = 10000*np.copy(self.DEP)
        # dep_v[0:i_10] = 0
        # dep_v[i_10:] = dep_v[i_10:]/(2*np.pi*self.freq_sinc[i_10:])
        self.DEPV = dep_v  
        self.DEPV2 = dep_v*dep_v  
        self.features = {}   
        self.harm1()
        self.velocity_10_1000()
        self.acel_high()
        self.JB_harm6()
        self.low_order_rms()
        self.rms_no_harm_1000()
        self.high_order_rms()
        self.electric_motor()
        self.rms_velocity()
        self.envelope_peak_to_peak()
        self.acc_peak_to_peak()
        self.rms_acc()
        self.all_rms_bands()
        self.max_amplitudes_harmonics()
        self.kurtosis()
        self.fator_crista()
        self.fator_k()
        self.rol_1()
        self.rol_2()
        self.rol_3()
        self.rol_3_max()
        self.eng_1()
        self.eng_2()
        self.eng_3()
        self.eng_3a_max() 
        self.eng_3v_max()
        self.eng_4()
        self.cav_1()
        self.cav_2()
        self.lub_1()
        self.lub_2()
        self.pas_1()
        self.oil_1()
        #self.turb_1()



    def velocity_10_1000(self):
        i_10 = int(10/self.df_sinc) 
        i_1000 = int(1000/self.df_sinc) 
        rms = np.sqrt(np.sum( self.DEPV2[i_10:i_1000]))
        self.features['v_10_1000'] = rms

    def harm1(self):
        i_0 = int(self.rps / self.df_sinc)
        rms = np.sum(self.DEPV2[i_0 - 2:i_0 + 3])
        self.features['harm1'] = np.sqrt(rms)

    def acel_high(self):
        aux = np.sum(self.DEP2[int(1000/self.df_sinc):])
        self.features['acel_high'] = np.sqrt(aux)

    def JB_harm6(self):
        i_0 = int(self.rps / self.df_sinc)

        def sum_energy_around(idx):
            return np.sum(self.DEP2[idx - 2:idx + 3])

        rms = sum_energy_around(int(i_0 / 2))
        for multiplier in [2, 3, 4, 6, 8]:
            rms += sum_energy_around(multiplier * i_0)

        self.features['JB_harm6'] = np.sqrt(rms)

    def low_order_rms(self):
        freq_bas = self.rps
        df = self.df_sinc
        i_10 = int(10 / df)
        i_5h = int(5.5 * freq_bas / df)
        DEP = np.copy(self.DEP2)
        DEP[:i_10] = 0
        DEP[i_5h:] = 0
        freq = self.freq_sinc
        i_freq = np.argmin(np.abs(freq - freq_bas))
        dep = np.sum(DEP[i_freq - 2:i_freq + 3])
        DEP[i_freq - 2:i_freq + 3] = 0
        for _ in range(5):
            i_freq = np.argmax(DEP)
            dep += np.sum(DEP[i_freq - 2:i_freq + 3])
            DEP[i_freq - 2:i_freq + 3] = 0
        self.features['low_order_rms'] = np.sqrt(dep)

    def rms_no_harm_1000(self):
        dep2= np.copy(self.DEP2)
        df = self.df_sinc
        i_0 = round(self.rps / df)
        i_1000 = int(1000 / df)
        dep2[i_1000:] = 0 
        for k in range(i_0, i_1000, i_0):
            dep2[k - 2:k + 3] = 0
        i_60 = int(60 / df)
        for k in range(i_60, i_1000, i_60):
            dep2[k - 2:k + 3] = 0
        soma = np.sum(dep2)
        self.features['rms_no_harm_1000'] = np.sqrt(soma)

    def high_order_rms(self):
        freq_min = 16 * self.rps
        nfft2 = int(self.npto / 2.56)   
        DEP = np.copy(self.DEP2)
        df = self.df_sinc
        i_min = int(freq_min / df)
        DEP[:i_min] = 0
        dep = 0
        i_0 = int(self.rps/(df))
        for _ in range(4):
            i_freq = np.argmax(DEP)
            DEP[i_freq - 2:i_freq + 3] = 0
            for j in range(i_freq - 2 * i_0, i_freq + 2 * i_0 + 1, i_0):
                dep += np.sum(DEP[j - 2:j + 3])
            DEP[i_freq - 2*i_0-2:i_freq + 2 * i_0 + 3] = 0    
        self.features['high_order_rms'] = np.sqrt(dep)

    def electric_motor(self):
        freq_bas = self.rps
        if freq_bas > 50:
            freq_min, freq_max = 18 * freq_bas, 43 * freq_bas
        elif freq_bas > 12:
            freq_min, freq_max = 30 * freq_bas, 60 * freq_bas
        else:
            freq_min, freq_max = 0, 0
        freq_max = min(freq_max, 0.5 / self.dt)
        acel = np.copy(self.sinal)
        nfft = self.npto
        nfft2 = int(nfft / 2.56)
        df = 1 / (nfft * self.dt)
        i_60 = round(60 / df)
        df = 60/i_60 
        dt_new = 1/(df*nfft) 
        acel_harm = ds.utilities.changedeltat(acel,self.dt,dt_new)[1]        
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
        self.features['eletric_motor'] = np.sqrt(dep)

    def rms_velocity1(self):
        i_10 = int(10/self.df_sinc) 
        i_1000 = int(1000/self.df_sinc) 
        rms = np.sqrt(np.sum( self.DEP[i_10:i_1000]))
        self.features['rms_v'] = rms

    def rms_velocity(self):
        envelope = ds.timeparameters.envelope(self.sinal_sinc, fc=1000, dt=self.dt_sinc)        
        dep2= ds.utilities.fft_spectrum(envelope, scaling= 'peak-peak')
        df = self.df_sinc
        i_0 = round(self.rps / df)
        i_1000 = int(1000 / df)
        dep2[i_1000:] = 0 
        for k in range(i_0, i_1000, i_0):
            dep2[k - 2:k + 3] = 0
        i_60 = int(60 / df)
        for k in range(i_60, i_1000, i_60):
            dep2[k - 2:k + 3] = 0
        soma = np.sum(dep2)
        self.features['rms_v'] = np.sqrt(soma)


    def envelope_peak_to_peak(self, passa_alta=500):
        envelope = ds.timeparameters.envelope(self.sinal, fc=passa_alta, dt=self.dt)
        self.features['env_peak'] = ds.timeparameters.peak_to_peak(envelope)

    def acc_peak_to_peak(self):
        self.features['acc_peak'] = ds.timeparameters.peak_to_peak(self.sinal)

    def rms_acc(self):
        self.features['rms_acc'] = ds.timeparameters.rms(self.sinal)

    def rms_band(self, vel_dep, band_low, band_high):
        lowcut = int(band_low * self.rps/self.df_sinc)
        highcut = int(band_high * self.rps/self.df_sinc)
        if highcut == lowcut : 
            highcut += 2
        rms = np.sqrt(np.sum(vel_dep[lowcut:highcut]))
        return rms

    def all_rms_bands(self): 
        v = np.copy(self.DEPV2) 
        bands = [
            (0.3, 0.78), (0.8, 1.2), (1.8, 2.2),
            (2.3, 3.6), (3.6, 12.2), (12.3, 16.6), (16.7, 25)
        ]
        for index, band in enumerate(bands):
            rotulo = f'rms_bands_{index}'
            self.features[rotulo] = self.rms_band(v, band[0], band[1])

    def max_amplitudes_harmonics(self):
        freq = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]
        harmonics = np.array(freq)*self.rps
        harmonics[harmonics > self.fs / 2] = 0.99 * self.fs / 2
        depv2 = self.DEPV2
        for i in range( len(freq) ):
            j = int(harmonics[i]/self.df_sinc)
            harm = np.sum( depv2[j-2:j+3] )
            self.features[f'max_harmonics_{freq[i]}'] = np.sqrt(harm)

    def max_amplitudes_harmonics_env(self, passa_alta = 500):
        sinal = np.copy(self.sinal)
        envelope = ds.timeparameters.envelope( sinal, fc = passa_alta, dt = self.dt ) 
        freq = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]
        harmonics = np.array([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8])*self.rps
        harmonics[harmonics>self.fs/2] = .99*self.fs/2
        dep_harm = ds.utilities.dep(envelope,self.dt,freqs=harmonics)
        for index, harm in enumerate(dep_harm):
            rotulo = 'max_harmonics_env' + str(freq[index])
            self.features[rotulo] = harm

    def kurtosis(self):
        self.features['kurtosis'] = ds.timeparameters.kurtosis(self.sinal)

    def fator_crista(self):
        self.features['crest_factor'] = ds.timeparameters.crest_factor(self.sinal)

    def fator_k(self):
        self.features['k_factor'] = ds.timeparameters.k_factor(self.sinal)

    def _preprocess_fft(self):
        x = np.copy( self.sinal_sinc ) 
        return x, 1/self.dt_sinc
    
    # precisa modificar os features para receber as freq. de rolamento
    # modificar as features para retornar valor maximo e nao soma         
    # colocar um IF caso nao exista a freq. de rolamento (BPFO, BPFI, BSF e FTF) ou passar valor 0
    def rol_1(self):
        if self.freq_rolamento is not None and np.any(np.isfinite(np.concatenate(self.freq_rolamento))):
            i_0 = int(self.rps / self.df_sinc)
            X = np.copy(self.DEP2)
            X[:int(750 / self.df_sinc)] = 0
            X[0:int(10/self.df_sinc)] = 0 
            for i in range(i_0, self.npto2, i_0):
                X[i - 2:i + 3] = 0
            threshold = np.mean(X) - np.std(X)
            self.features['rol_1'] = np.max(X[X > threshold])

    def rol_2(self):
        x = np.copy(self.sinal_sinc) 
        df = self.df_sinc
        env = ds.timeparameters.envelope(x, 500, dt=self.dt_sinc)
        X = np.abs(np.fft.fft(env / self.npto)[:self.npto2]) ** 2
        X = X[0:int(1000 / df)]
        X[0:int(3 / df)] = 0
        i_0 = int(self.rps / df)
        for i in range( i_0, self.npto2, i_0):
            X[i - 2:i + 3] = 0
        threshold = np.mean(X) - np.std(X)
        self.features['rol_2'] = np.max(X[X > threshold])

    def rol_3(self):
        if np.isnan(self.freq_rolamento[0]) == False:
            names = ['BPFO','BPFI','BSF','FTF']
            x = np.copy(self.sinal_sinc) 
            npto = self.npto
            df = self.df_sinc
            env = ds.timeparameters.envelope(x, 500, dt=self.dt)
            X = ds.utilities.fft_spectrum( env) ** 2
            X = X[:int(1000 / df)]
            X[:int(3 / df)] = 0
            i_1000= int(1000 / df)
            for k_ext in range( 4 ) :
                freq_rel = self.freq_rolamento[k_ext]
                freq_idx = int(freq_rel * self.rps / df)
                freq_max = 4*freq_idx
                ii=1
                for i in range(freq_idx, freq_max, freq_idx) : 
                    soma_harm = np.sum(X[i - ii*5:i + ii*6]) 
                    ii +=1
                self.features['rol3_%s'%(names[k_ext])] = soma_harm

    def rol_3_max(self):
        if np.isnan(self.freq_rolamento[0]) == False:
            names = ['BPFO','BPFI','BSF','FTF']
            x = np.copy(self.sinal_sinc) 
            npto = self.npto
            df = self.df_sinc
            env = ds.timeparameters.envelope(x, 500, dt=self.dt)
            X = ds.utilities.fft_spectrum( env , 'peak-peak' ) 
            X = X[:int(1000 / df)]
            X[:int(3 / df)] = 0
            i_1000= int(1000 / df)
            for k_ext in range( 4 ) :
                freq_rel = self.freq_rolamento[k_ext]
                freq_idx = int(freq_rel * self.rps / df)
                max = -1e12
                for i in range(freq_idx, i_1000, freq_idx) : 
                    max_local = np.max(X[i - 2:i + 3]) 
                    if max_local > max : 
                        max = max_local
                self.features['rol_%s_max'%(names[k_ext])] = max


    # criar o feature com o nivel de vibracao nas GMFs em velocidade
    def eng_1(self):
        X = np.copy(self.DEPV2)
        X[:int(400 / self.df_sinc)] = 0
        i_0 = int(self.rps / self.df_sinc)
        rms = sum(np.sum(X[i - 2:i + 3]) for i in range(i_0, self.npto2, i_0))
        self.features['eng_1'] = np.sqrt(rms) 

    def eng_2(self):
        X = np.copy(self.DEPV2)
        X[:int(750 / self.df_sinc)] = 0
        i_0 = int(self.rps / self.df_sinc)
        rms = sum(np.sum(X[i - 2:i + 3]) for i in range(i_0, self.npto2, i_0))
        self.features['eng_2'] = np.sqrt(rms)

    def eng_3(self):
        if np.isnan(self.num_dentes) == False:
            GMF = np.copy(self.num_dentes * self.rps)
            X =  np.copy(self.DEP2)
            X[:int(3 / self.df_sinc)] = 0
            i_GMF = int(GMF / self.df_sinc)
            rms = sum(np.sum(X[i - 2:i + 3]) for i in range(i_GMF, 4 * i_GMF, i_GMF))
            self.features['eng_3'] = np.sqrt(rms)

    def eng_3a_max(self):
        if np.isnan(self.num_dentes) == False:
            GMF = self.num_dentes * self.rps          
            X = np.copy(2*self.DEP)
            X[:int(3 / self.df_sinc)] = 0
            i_GMF = int(GMF / self.df_sinc) 
            i_max = int(4*GMF)+3 
            if i_max > self.npto2 : 
                i_max = self.npto2-3
            max = -1e12    
            for i in range(i_GMF, i_max, i_GMF) : 
                max_loc = np.max(X[i - 2:i + 3])
                if max_loc > max : 
                    max = max_loc 
            self.features['eng_3a_max'] = max 

    def eng_3v_max(self):
        if np.isnan(self.num_dentes) == False:
            GMF = self.num_dentes * self.rps          
            X = 2*np.copy(self.DEPV)  
            X[:int(3 / self.df_sinc)] = 0
            i_GMF = int(GMF / self.df_sinc) 
            i_max = int(4*GMF)+3 
            if i_max > self.npto2 : 
                i_max = self.npto2-3
            max = -1e12    
            for i in range(i_GMF, i_max, i_GMF) : 
                max_loc = np.max(X[i - 2:i + 3])
                if max_loc > max : 
                    max = max_loc 
            self.features['eng_3v_max'] = max 

    def eng_4(self):
        if np.isnan(self.num_dentes) == False:
            GMF = self.num_dentes * self.rps
            X = np.copy(self.DEP2)
            X[:int(3 / self.df_sinc)] = 0
            i_GMF = int(GMF / self.df_sinc)
            i_0 = int(self.rps / self.df_sinc)
            soma_harm = 0
            for i in range(i_GMF, 4 * i_GMF, i_GMF):
                for j in range(-3, 4):
                    soma_harm += np.sum(X[i + j * i_0:i + j * i_0 + 3])
            self.features['eng_4'] = np.sqrt(soma_harm)
        
    # mudar o lub_1, remove as harmonicas e as 10 maiores assicronas, depois faz o valor medio das 10 maiores freq. acima de 500 hz
    def lub_1(self):
        X = np.copy(self.DEP2)
        X[:int(500 / self.df_sinc)] = 0
        i_0 = int(self.rps/self.df_sinc)
        for i in range( i_0, self.npto2, i_0):
            X[i - 2:i + 3] = 0
        media = 0 
        for i in range(10):
            k = np.argmax(X)
            media += ( np.sum(X[k - 2:k + 3]) - media)/(i+1)
            X[k - 2:k + 3] = 0
        self.features['lub_1'] = media

    # mudar o lub_2 para média dos 100 menores valores no espectro de envelope até 500 hz
    def lub_2(self):
        npto = len(self.sinal)
        df = self.fs / npto
        env = ds.timeparameters.envelope(self.sinal, fc=500, dt=self.dt)
        X = np.fft.fft(env / npto)
        X = np.abs(X[:self.npto2]) * np.sqrt(2)
        X = X[int(500 / df):]
        min_vals = np.zeros(100)
        for i in range(100):
            k = np.argmin(X)
            min_vals[i] = X[k]
            X[k - 1:k + 2] = 1e8
        self.features['lub_2'] = np.mean(min_vals)

    # conferir os limites da cavitacao
    def cav_1(self):
        X = np.copy(self.DEP2)
        X = X[int(500 / self.df_sinc):int(3000 / self.df_sinc)]
        i_0 = int(self.rps / self.df_sinc)
        for i in range( i_0, self.npto2, i_0):
            X[i - 2:i + 3] = 0
        for _ in range(10):
            k = np.argmax(X)
            X[k - 2:k + 3] = 0
        self.features['cav_1'] = np.sqrt(np.sum(X))

    # conferir os limites da cavitacao
    def cav_2(self):
        X = np.copy(self.DEP2)
        X = X[int(500 / self.df_sinc):int(3000 / self.df_sinc)]
        min_vals = np.zeros(100)
        for i in range(100):
            k = np.argmin(X)
            min_vals[i] = X[k]
            X[k - 1:k + 2] = 1e8
        self.features['cav_2'] = np.mean(min_vals)

    #???? mudar o parametro para valor maximo de amplitude entre as harmonicas (melhor para comparar)
    def pas_1(self):
        if np.isnan(self.num_pas) == False:
            FP = self.num_pas * self.rps
            X = np.copy(self.DEP2)
            X = X[:int(1000 / self.df_sinc)]
            i_FP = int(FP / self.df_sinc)
            soma_harm = sum(np.sum(X[i - 3:i + 3]) for i in range(i_FP, 3 * i_FP, i_FP))
            self.features['pas_1'] = np.sqrt(soma_harm)

    def pas_max(self):
        if np.isnan(self.num_pas) == False:
            FP = self.num_pas * self.rps
            X = np.copy(self.DEP2)
            X = X[:int(1000 / self.df_sincf)]
            i_FP = int(FP / self.df_sincf)
            max = -1e6
            for i in range(i_FP, 3 * i_FP, i_FP) : 
                soma_harm = sum(np.sum(X[i - 3:i + 3]))
                if soma_harm > max: 
                    max = soma_harm
                                
            self.features['pas_1'] = np.sqrt(soma_harm)

    def oil_1(self):
        v = self.DEPV2
        self.features['oil_1'] = np.sqrt(self.rms_band(v, 0.42, 0.48))

    def turb_1(self):
        i_0 = int(self.rps / self.df_sinc)
        X = np.copy(self.DEP2)
        X[:4] = 1e8
        min_vals = np.zeros(i_0 - 2)
        for i in range(i_0 - 2):
            k = np.argmin(X)
            j = 0
            while X[k] < 1e-12 and j < 10:
                X[k] = 1e8
                k = np.argmin(X)
                j += 1
            min_vals[i] = X[k]
            X[k - 1:k + 2] = 1e8
        self.features['turb_1'] = np.sqrt(np.sum(min_vals))
