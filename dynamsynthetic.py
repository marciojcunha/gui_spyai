import numpy as np
import dynamsignal as dys
from scipy import interpolate
from dynamplot import plotar_graficos
import matplotlib.pyplot as plt
from dynamaugmentation import Augmentation
import time

class SyntheticSignal:
    def __init__(self, sinal, fri, rot_ang, fs=20480, freq_rotacao = 60, sinal_original =  None):
        sinal = np.array(sinal, dtype=float)
        npto = len(sinal) 
        if npto%2 != 0 : 
            sinal = np.concatenate( (sinal,[sinal[0]]), axis = 0 )
            if sinal_original is not None : 
                sinal_original = np.concatenate( (sinal_original,[sinal_original[0]]), axis = 0 )        
        self.sinal = np.copy(sinal)
        self.sinal_original = sinal_original
        self.fri = fri
        self.rot_ang = rot_ang
        self.fs = fs
        self.Freq_rotacao = freq_rotacao
        self.freq_rotacao = freq_rotacao
        self.npto =  len(self.sinal)
        self.npto2 = int(self.npto/2)
        self.dt = 1 / self.fs
        self.t = np.arange(0, self.npto) * self.dt
        self.pseudo_tf = self.pseudo_response()
        self.dt_sinc = self.dt
        self.sinais_base = self.sinal_base_augmentation( self.sinal, n=3)
        self.len_sinais = len(self.sinais_base)
        # Pré-calcula amplitude média do sinal original (evita recomputar)
        self.xAmp = 1 # np.mean(sinal[sinal > 0.9 * np.max(sinal)])

    def sinal_base_augmentation(self, sinal_base, n = 10 ) :         
        sinal = np.copy(sinal_base )
        a = Augmentation(sinal, 3, self.fs)
        sinais_aug = a.calcular_data_aug(np.std(sinal))
        sinal_augmentation = np.zeros( (len(sinais_aug)*n,len(sinal)) )  
        ii = 0       
        for i in range(len(sinais_aug)):
            for j in range( n ) : 
                sinal_augmentation[ii] = sinais_aug[i][j]
                ii += 1
        return sinal_augmentation 


    
    def rpm_sinc_data(self,data) : 
        f0 = self.freq_rotacao 
        npto_old = self.npto
        df_old = 1/(npto_old*self.dt)
        i = int(f0/df_old) 
        df = f0/i 
        dt_new = 1/(df*npto_old) 
        sinal_sinc = np.zeros((len(data)))
        sinal = np.copy(data)
        sinal_sinc = dys.utilities.changedeltat(sinal,self.dt,dt_new)[1] 
        self.dt_sinc = dt_new
        return sinal_sinc




    def add_fault(self, tipo_falha, **kwargs): 
        i = np.random.randint(0,self.len_sinais) 
        df = 1/(self.npto*self.dt_sinc)
        self.freq_rotacao = self.Freq_rotacao+np.random.uniform(-1*df,2*df)
        self.sinal = self.sinais_base[i]   
        self.X_sinal = np.abs(dys.utilities.fft_spectrum(self.sinal ))         

        fator_correcao = 1
        index = kwargs['index_show'] 
        if tipo_falha == 'Normal': falha = self.normal_signal(**kwargs)
        if tipo_falha == 'Rolamento_BPFI': falha = self.BPFI_bearing_fault(**kwargs)
        if tipo_falha == 'Rolamento_BPFO': falha = self.BPFO_bearing_fault(**kwargs)   
        if tipo_falha == 'Rolamento_BSF': falha = self.BSF_bearing_fault(**kwargs)  
        if tipo_falha == 'Rolamento_FTF': falha = self.FTF_bearing_fault(**kwargs)                      
        if tipo_falha == 'Desbalanceamento': 
            falha =self.unbalance_fault(**kwargs) + self.sinal*np.random.uniform(1,3)
        if tipo_falha == 'Polia': falha = self.belt_fault(**kwargs)

        if tipo_falha == 'Desalinhamento': 
            falha =self.misalignment_fault(**kwargs)
            i_X1 = int( self.freq_rotacao/df)
            X1 = 0
            for i in range( i_X1, 4*i_X1, i_X1 ) :
                X1 += np.sum(self.X_sinal[i-5:i+4]) 
            self.Falha = np.sum(np.abs(dys.utilities.fft_spectrum(falha))) 
            fator_correcao = X1/self.Falha 

        if tipo_falha == 'Engrenagem': falha =self.gear_fault(**kwargs)
        if tipo_falha == 'Folga': 
            falha =self.looseness_fault(**kwargs)
            fator_correcao = .5*dys.timeparameters.rms(self.sinal)/dys.timeparameters.rms(falha) 
        if tipo_falha == 'Cavitação': falha =self.cavitation_fault(**kwargs)
        if tipo_falha == 'Roçamento': falha =self.rubbing_fault(**kwargs)
        if tipo_falha == 'Origem Elétrica': 
            falha =self.motor_fault(**kwargs)
            i_60 = int(60/df)
            X1 = 0
            for i in range( i_60, 1000, i_60 ) :
                X1 += np.sum(self.X_sinal[i-5:i+4]) 
            self.Falha = np.sum(np.abs(dys.utilities.fft_spectrum(falha))) 
            fator_correcao = X1/self.Falha 
        if tipo_falha == 'Lubrificação': 
            falha =self.lubrification_fault(**kwargs)
            fator_correcao = .2*dys.timeparameters.rms(self.sinal)/dys.timeparameters.rms(falha) 
        if tipo_falha == 'Ressonância' : falha = self.ressonance_fault()
        if tipo_falha == 'Normal' or tipo_falha == 'Desbalanceamento' : 
            if index < 2 :
                #plotar_graficos(self.sinal_original, falha, self.dt_sinc, title = tipo_falha ) 
                pass
            bobo=1
            return falha
            
        x_conv = falha                  
        # Convolução da falha com resposta pseudo-dinâmica
        falha *=  fator_correcao*np.random.uniform(2,6)
        sinal =  (self.sinal + falha)*np.random.uniform(.8,2)     
        '''
        import matplotlib.pyplot as plt
        A  =np.abs(np.fft.fft(self.sinal/8192))
        B=np.abs(np.fft.fft(x_conv/8192))
        df=1/(Npto*self.dt)
        freq = np.linspace(0,(Npto-1)*df,Npto) 
        Npto2 = int(Npto/2)
        A = A[0:Npto2]
        B=B[0:Npto2]
        freq = freq[0:Npto2]
        # plt.plot(freq,A,freq,B, freq, A+B) 
        # plt.show()
        '''
        #sinal = self.sinal + x_conv
        if index < 2 :
            # plotar_graficos(self.sinal_original, sinal, self.dt_sinc, title = tipo_falha ) 
            # bobo=1
            pass
        return sinal # /(dys.timeparameters.rms(sinal)+1e-8)


    def pseudo_response(self):
        if not self.fri:
            Npto = self.npto
            Npto2 = int(Npto/2)
            npto = 256 
            while npto > int(Npto/2) :
                npto = int(npto/2) 
            #if self.sinal_original is None :     
            if 3>2 :
                sinal = dys.utilities.frequency_filter(self.sinal_original,1000,self.dt,type='highpass',
                                                       poles=6)             
                
                # ok = True 
                npto2 = int(Npto/2)
                for kext in range(2) :
                    S = np.fft.fft(sinal/Npto)                     
                    Sa = np.abs(S[0:npto2])
                    Sa[0:10] = 0  
                    for i in range(11,npto2-1) : 
                        if Sa[i-1]+Sa[i+1] < Sa[i] :
                            S[i] = .5*(S[i-1]+S[i+1]) 
                    S[Npto-1:Npto2+1:-1] = np.conj(S[1:Npto2-1])
                    sinal= np.real(np.fft.fft(np.conj(S)))           
                # ii = 0
                # Sam = np.zeros(npto2) 
                # for i in range(7,npto2-7) : 
                #     Sam[i] = np.mean(Sa[i-7:i+8])
                # while ok or ii<npto2: 
                #     index =  np.argmax(Sa)
                #     if Sa[index-1]+Sa[index+1] < Sa[index] : 
                #         Sa[index] = 0 
                #         S[index] = 0
                #     else: 
                #         ok = False 
                #     ii += 1         
                pseudo_tf = dys.utilities.autocorrelation(sinal,npto,self.dt,60*self.freq_rotacao)

            else : 
                sinal = dys.utilities.frequency_filter(self.sinal_original,1000,self.dt,type='highpass')
                pseudo_tf = dys.utilities.autocorrelation(sinal,npto,self.dt,60*self.freq_rotacao)                
            npto = len(pseudo_tf)
            pseudo_tf = pseudo_tf/np.exp(6*np.arange(0,npto,1)/npto)
        else:
            pseudo_tf = np.zeros( self.npto )
            pseudo_tf[:len(self.fri)] = self.fri
        return pseudo_tf/np.max(pseudo_tf)

    def _sin_component(self, freq, amplitude=1.0):
        ang_ran =  np.random.uniform(0, 2 * np.pi)
        return amplitude * self.xAmp * np.sin(2 * np.pi * freq * self.t + ang_ran)

    def BPFI_bearing_fault(self, **kwargs):
        BPFI = kwargs['bpfi']
        fb = self.freq_rotacao*BPFI   
        i_fb = int(self.fs/fb)      
        n_fault = np.random.random_integers(1,4) 
        i_T0 = np.zeros(n_fault, dtype=int) # extensão do defeito 
        Amp = np.zeros(n_fault) # amplitude do defeito
        Posic = np.zeros(n_fault, dtype=int) # Posição do defeito
        for k in range(n_fault) : 
            i_T0[k] = np.random.random_integers(2,6) 
            Amp[k] = np.random.uniform(.8,1.2 ) 
            Posic[k] = np.random.random_integers(2,i_fb-i_T0[k]-3)
            if Posic[k] < 0 : 
                Posic[k] = 0 
        npto = self.npto 
        falha = np.zeros(npto) 
        for k_ext in range(0,npto,i_fb) : 
            for j_fault in range(n_fault) :
                j_i = k_ext + Posic[j_fault] + np.random.random_integers(-2,3) 
                if j_i< 0 : 
                    j_i=0
                j_f = j_i+i_T0[j_fault] 
                if j_f >= npto : 
                    break 
                amp = Amp[j_fault] + np.random.uniform(-.1,.1)  
                falha[j_i:j_f] = amp*np.sin(np.pi*np.arange(0,i_T0[j_fault],1)/(i_T0[j_fault]-1) )  
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-npto:]  

        return x_conv    

    def BPFO_bearing_fault(self, **kwargs):
        BPFO = kwargs['bpfo']
        fb = self.freq_rotacao*BPFO   
        i_fb = int(self.fs/fb)      
        n_fault = np.random.random_integers(1,4) 
        i_T0 = np.zeros(n_fault, dtype=int) # extensão do defeito 
        Amp = np.zeros(n_fault) # amplitude do defeito
        Posic = np.zeros(n_fault, dtype=int) # Posição do defeito
        for k in range(n_fault) : 
            i_T0[k] = np.random.random_integers(2,6) 
            Amp[k] = np.random.uniform(.8,1.2 ) 
            Posic[k] = np.random.random_integers(2,i_fb-i_T0[k]-3)
            if Posic[k] < 0 : 
                Posic[k] = 0 
        npto = self.npto 
        falha = np.zeros(npto) 
        for k_ext in range(0,npto,i_fb) : 
            for j_fault in range(n_fault) :
                j_i = k_ext + Posic[j_fault] + np.random.random_integers(-2,3) 
                if j_i< 0 : 
                    j_i=0
                j_f = j_i+i_T0[j_fault] 
                if j_f >= npto : 
                    break 
                amp = Amp[j_fault] + np.random.uniform(-.1,.1)  
                falha[j_i:j_f] = amp*np.sin(np.pi*np.arange(0,i_T0[j_fault],1)/(i_T0[j_fault]-1) )                 
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-npto:]  

        return x_conv    

    def BSF_bearing_fault(self, **kwargs):
        BSF = kwargs['bsf']
        fb = self.freq_rotacao*BSF   
        i_fb = int(self.fs/fb)      
        n_fault = np.random.random_integers(1,4) 
        i_T0 = np.zeros(n_fault, dtype=int) # extensão do defeito 
        Amp = np.zeros(n_fault) # amplitude do defeito
        Posic = np.zeros(n_fault, dtype=int) # Posição do defeito
        for k in range(n_fault) : 
            i_T0[k] = np.random.random_integers(2,6) 
            Amp[k] = np.random.uniform(.8,1.2 ) 
            Posic[k] = 2+np.random.random_integers(2,i_fb-i_T0[k]-3)
            if Posic[k] < 0 : 
                Posic[k] = 0 
        npto = self.npto 
        falha = np.zeros(npto) 
        for k_ext in range(0,npto,i_fb) : 
            for j_fault in range(n_fault) :
                j_i = k_ext + Posic[j_fault] + np.random.random_integers(-2,3) 
                if j_i< 0 : 
                    j_i=0
                j_f = j_i+i_T0[j_fault] 
                if j_f >= npto : 
                    break 
                amp = Amp[j_fault] + np.random.uniform(-.1,.1)  
                falha[j_i:j_f] = amp*np.sin(np.pi*np.arange(0,i_T0[j_fault],1)/(i_T0[j_fault]-1) )                 
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-npto:]  

        return x_conv    

    def FTF_bearing_fault(self, **kwargs):
        FTF = kwargs['ftf']
        fb = self.freq_rotacao*FTF   
        i_fb = int(self.fs/fb)      
        n_fault = 1 + 0*np.random.random_integers(1,4) 
        i_T0 = np.zeros(n_fault, dtype=int) # extensão do defeito 
        Amp = np.zeros(n_fault) # amplitude do defeito
        Posic = np.zeros(n_fault, dtype=int) # Posição do defeito
        for k in range(n_fault) : 
            i_T0[k] = 2 + 2 +0*np.random.random_integers(2,6) 
            Amp[k] = np.random.uniform(.8,1.2 ) 
            Posic[k] = 2 + np.random.random_integers(2,i_fb-i_T0[k]-3)
            if Posic[k] < 0 : 
                Posic[k] = 0 
        npto = self.npto 
        falha = np.zeros(npto) 
        for k_ext in range(0,npto,i_fb) : 
            for j_fault in range(n_fault) :
                j_i = k_ext + Posic[j_fault] + np.random.random_integers(-2,3) 
                if j_i< 0 : 
                    j_i=0
                j_f = j_i+i_T0[j_fault] 
                if j_f >= npto : 
                    break 
                amp = Amp[j_fault] + np.random.uniform(-.1,.1)  
                falha[j_i:j_f] = amp*np.sin(np.pi*np.arange(0,i_T0[j_fault],1)/(i_T0[j_fault]-1) ) 
        falha = dys.utilities.frequency_filter(falha,5*fb,self.dt)                        
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-npto:]  

        return x_conv    

    def multi_harmonic_fault(self, freqs, amps=None):
        if amps is None:
            amps = [np.random.uniform(1, 1.3) for _ in range(len(freqs))]
        a = sum(self._sin_component(freq, amp) for freq, amp in zip(freqs, amps))
        return a

    def unbalance_fault(self,**kwargs):
        Amp = np.random.uniform(4,11)*np.pi*2*self.freq_rotacao/10000
        return self._sin_component(self.freq_rotacao,amplitude=Amp)

    def misalignment_fault(self,**kwargs):

        freqs = [self.freq_rotacao, 2 * self.freq_rotacao, 3 * self.freq_rotacao, 
                 4*self.freq_rotacao, 5*self.freq_rotacao ]
        amps = [np.random.uniform(.8, 1.5),
                np.random.uniform(2.4, 4.5),
                np.random.uniform(3.5, 6.5),
                np.random.uniform(5, 10),
                np.random.uniform(8, 15),
                ]
        
        return self.multi_harmonic_fault(freqs, amps)

    
    def belt_fault( self, **kwargs): 
        BPF = kwargs['BPF'] 
        if BPF is None : 
            return None
        freq_polia_movida = self.freq_rotacao*kwargs['BReduc']
        x_fault = np.sin(2*np.pi*BPF*self.t+np.random.normal) 
        x_fault += np.random.uniform(2,4)*np.sin(4*np.pi*BPF*self.t+np.random.normal) 
        x_fault += np.random.uniform(1.5,2.)*np.sin(2*np.pi*self.freq_rotacao*self.t+np.random.normal) 
        x_fault += np.random.uniform(1.5,2.)*np.sin(2*np.pi*freq_polia_movida*self.t+np.random.normal) 
        return x_fault
    
    def rubbing_fault(self,**kwargs):
        t_impact = np.arange(0, 5e-3, self.dt)
        xImpact = np.sin(2 * np.pi * 2000 * t_impact) * np.hanning(len(t_impact))

        xComb = np.zeros( self.npto)
        indices = np.arange(0, self.npto, int(self.fs / (0.5 * self.freq_rotacao)))
        xComb[indices] = 1
        xRubbing = np.convolve(xComb, xImpact[::-1], mode='same')

        mod_harm = self.multi_harmonic_fault([self.freq_rotacao, 2 * self.freq_rotacao, 3 * self.freq_rotacao])
        falha = xRubbing + 0.5*mod_harm
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-self.npto:]  

        return x_conv
    
    def gear_fault( self,**kwargs) :
        npto =  self.npto
        dt = self.dt
        fs = 1/dt
        n1 = kwargs['num_dentes_pinhao'] 
        if n1 is None : 
            return None
        tempo = np.linspace(0,(npto-1)*dt, npto)  
        n2 = kwargs['num_dentes_coroa'] 
        freq_rotac = kwargs['freq_rotac_pinhao'] 
        if freq_rotac is None : 
            freq_rotac = kwargs['freq_rotac_coroa']*n1/n2  
        A = 1
        e = A*np.random.uniform(.8,1.5) 
        GMF = freq_rotac*n1 
        n = 5 
        if n*GMF > .35*fs : 
            n = int(.35*fs/GMF)
        m = 5 
        x = np.zeros(npto) 
        for i in range(n) : 
            Ar = np.random.uniform(.8,1.2)*A/(i+1)**4    
            for j in range(m) : 
                er = e*np.random.uniform(.2,.8)*Ar 
                x += (Ar+er*np.sin(2*np.pi*j*freq_rotac*tempo+np.random.normal()) )*np.sin(2*np.pi*i*GMF*tempo + np.random.normal() )
        a = np.zeros(npto)
        for i in range(1,npto-1) : 
            a[i] = (x[i+1]+x[i-1]-2*x[i])/dt**2
        X = np.fft.fft(a/npto)
        npto2 = int(npto/2)
        X = np.abs(X[0:npto2])*np.sqrt(2) 
        i_max = np.argmax(X) 
        max_geral =  0.26*GMF
        if max_geral > 15 : 
            max_geral = 15
        ganho = np.random.uniform(.2,1)*max_geral/X[i_max] 
        x = x*ganho
        a = a*ganho 
        X = X*ganho
        # Baixas Frequências 
        GP_freq = freq_rotac
        GC_freq = freq_rotac*n1/n2   
        HTF_freq = GMF/np.lcm(n1,n2) 
        GAPF_freq = GMF/3
        # frequencia do pinhão
        max = np.max(X) 
        GP_A =  np.random.uniform(.5,2.)*max
        a += GP_A*np.sin(2*np.pi*GP_freq*tempo + np.random.normal())
        a +=  np.random.uniform(0,.4)*GP_A*np.sin(4*np.pi*GP_freq*tempo + np.random.normal())
        # frequencia da coroa
        GC_A =  np.random.uniform(.5,2)*max
        a += GC_A*np.sin(2*np.pi*GC_freq*tempo + np.random.normal())
        a +=  np.random.uniform(0,.4)*GC_A*np.sin(4*np.pi*GC_freq*tempo + np.random.normal())
        # HTF
        HTF_A =  np.random.uniform(.5,3)*max
        a += 0*HTF_A*np.sin(2*np.pi*HTF_freq*tempo + np.random.normal())
        a +=  0*np.random.uniform(0,.4)*HTF_A*np.sin(4*np.pi*HTF_freq*tempo + np.random.normal())
        # GAPF 
        GAPF_A =  np.random.uniform(.5,1.8)*max
        a += 0*GAPF_A*np.sin(2*np.pi*GAPF_freq*tempo + np.random.normal())
        a +=  0*np.random.uniform(0,.4)*GAPF_A*np.sin(4*np.pi*GAPF_freq*tempo + np.random.normal())
        A_impact = np.random.uniform(.5,2.5)*max_geral
        impact = self.impact_(self.freq_rotacao,amp=A_impact)
        falha = a + impact 
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-npto:]  
        return x_conv
    
    def impact_(self,freq,amp=.1) : 
        npto = self.npto
        imp = np.zeros(npto)
        impac_interval = int(1/(freq*self.dt)) 
        T_imp = 5 
        impact = np.zeros(T_imp) 
        for j in range(T_imp) : 
            impact[j] = amp*np.sin( j*np.pi/(T_imp-1)) 
        for i in range(0,npto-T_imp,impac_interval) : 
            j = i+np.random.randint(0,int(impac_interval/8))
            if j + T_imp > npto-1 : 
                j = npto-1-T_imp
            imp[j:j+T_imp] = np.random.uniform(.8,1)*impact 
        return imp
    
    def looseness_fault(self,**kwargs):
        freqs = [0.5 * self.freq_rotacao,  1.5 * self.freq_rotacao,
            2 * self.freq_rotacao, 2.5 * self.freq_rotacao, 3 * self.freq_rotacao,
            4 * self.freq_rotacao, 5 * self.freq_rotacao, 6 * self.freq_rotacao, 7 * self.freq_rotacao]
        amps = [np.random.uniform(0.3, 0.6), np.random.uniform(.3, 0.6),
        np.random.uniform(1.2, 2.0), np.random.uniform(1, 1.5),
        np.random.uniform(0.6, 1.4), np.random.uniform(2.5, 5.5),
        np.random.uniform(0.6, 1.), np.random.uniform(.4, .8),
        np.random.uniform(.2, .4), np.random.uniform(.2, .4)]
        harm = self.multi_harmonic_fault(freqs, amps)
        impact = self.impact_(self.freq_rotacao,amp=10)
        falha =  harm+impact
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-self.npto:]  
        return x_conv


    def cavitation_fault(self,**kwargs):
        num_pas = kwargs['num_pas']
        Fa = 1/self.dt
        nfft = self.npto
        nfft2 = int(nfft/2)
        df = Fa/nfft
        i_0 = int(self.freq_rotacao/df) 
        i_vp = num_pas*i_0 
        f_ini = 1000 
        f_fim = 5000 
        if f_fim > Fa/2 : 
            f_fim = Fa/2 
        f_meio = f_ini+.3*(f_fim-f_ini)
        f_meio2 = f_ini+.8*(f_fim-f_ini)
        i_ini = int(f_ini/df)
        i_meio = int(f_meio/df)
        i_meio2 = int(f_meio2/df)
        i_fim = int(f_fim/df)
        x = [i_ini, i_meio, i_meio2, i_fim] 
        y = [0, np.random.uniform(.4,1.8), np.random.uniform(.9,1.8), 0 ] 
        tck  = interpolate.splrep(x,y, s=0) 
        z = np.arange(i_ini, i_fim, 1, dtype=int) 
        pz = interpolate.splev(z, tck, der=0)
        ruido_r = np.random.uniform(0.4, .8, size= len(pz))
        ruido_i = np.random.uniform(0.4, .8, size= len(pz))
        for i in range(1,len(ruido_i)) : 
            ruido_r[i] = ruido_r[i-1] + (ruido_r[i]-ruido_r[i-1])/50
            ruido_i[i] = ruido_r[i-1] + (ruido_i[i]-ruido_i[i-1])/50

        x = np.zeros(nfft).astype('complex') 
        x[i_ini:i_fim] = 10*pz*ruido_r + 10*pz*ruido_i*1j
        sinal= np.sign(np.random.normal(size=nfft2)) + np.sign(np.random.normal(size=nfft2))*1j
        x[0:nfft2] *= sinal
        x[i_0] = np.random.uniform(1.9,2.3) + np.random.uniform(1.9,2.3)*1j
        x[i_0-1] = np.random.uniform(.2,.6)*x[i_0]
        x[i_0+1] = np.random.uniform(.2,.6)*x[i_0]
        x[i_vp] = np.random.uniform(1.8,2.2) + np.random.uniform(1.8,2.2)*1j
        x[i_vp-1] = np.random.uniform(.3,.7)*x[i_vp]
        x[i_vp+1] = np.random.uniform(.3,.7)*x[i_vp]
        x[nfft-1:nfft2+1:-1] = np.conj(x[1:nfft2-1])
        x= np.real(np.fft.fft(np.conj(x)))
        return x

    def motor_fault(self,**kwargs ):
        Fr = self.freq_rotacao
        Fl = 60
        P = kwargs['num_polos'] 
        NRB = kwargs['num_barras_rotor']
        if np.isnan(NRB) : 
            if NRB < 1 :
                Fp = 60 
                freqs = [] 
                amps = [] 
                for i in range(Fp,1000,Fp) : 
                    freqs.append(i) 
                    amps.append(np.random.uniform(0.8,1.2)) 
                return self.multi_harmonic_fault(freqs, amps)            
        Fb = NRB*Fr 
        Fs = Fl-self.freq_rotacao
        Fp = Fs*P 
        Fl2 =2*Fl 
        Fl4 = 4*Fl
        Fp2 = 2*Fp
        Fr2 = 2*Fr
        Fr3 = 3*Fr
        Fb2 = 2*Fb
        index = np.random.random_integers(4)
        if index == 0 : # Stator eccentricity, shorted laminations and loose iron
            freqs = [Fr, 2*Fr, 2*Fl] 
            amps = [np.random.uniform(0.8,1.2),np.random.uniform(0.3,.6), 
                    np.random.uniform(1.6,2.2) ]
            return self.multi_harmonic_fault(freqs, amps)
        if index == 1 : # Eccentric Rotor
            dados = [Fp, np.random.uniform(0.3,0.6),
                    Fr-Fp,np.random.uniform(0.1,.4),
                    Fr,np.random.uniform(0.8,1.2),
                    Fr+Fp, np.random.uniform(0.1,.4),
                    Fl2 - Fp2, np.random.uniform(0.1,.4),
                    Fl2 - Fp, np.random.uniform(0.2,.8),
                    Fl2, np.random.uniform(1.6,2.4),
                    Fl2 + Fp, np.random.uniform(0.2,.8),
                    Fl2 + Fp, np.random.uniform(0.1,.4),]
            freqs = dados[0:-1:2] 
            amps = dados[1:-1:2]
            return self.multi_harmonic_fault(freqs, amps)
        if index == 2 : # Rotor Problems
            dados = [Fr-Fp, np.random.uniform(0.3,0.6),
                    Fr,np.random.uniform(1.4,2.2),
                    Fr+Fp,np.random.uniform(0.3,0.6),
                    Fr2-Fp, np.random.uniform(0.1,.4),
                    Fr2, np.random.uniform(0.8,1.4),
                    Fr2 + Fp, np.random.uniform(0.1,.4),
                    Fr3 -Fp, np.random.uniform(0.1,0.4),
                    Fr3, np.random.uniform(0.7,1.3),
                    Fr3 + Fp, np.random.uniform(0.1,.4),]
            freqs = dados[0:-1:2] 
            amps = dados[1:-1:2]
            return self.multi_harmonic_fault(freqs, amps)
        if index == 3 : # Rotor Problems
            dados = [Fr, np.random.uniform(0.8,1.2),
                    Fr2,np.random.uniform(0.2,.6),
                    Fb-Fl4,np.random.uniform(0.1,0.4),
                    Fb-Fl2, np.random.uniform(0.6,1.2),
                    Fb, np.random.uniform(1.6,2.1),
                    Fb + Fl2, np.random.uniform(0.6,1.2),
                    Fb + Fl4, np.random.uniform(0.1,.4)
                     ]

            freqs = dados[0:-1:2] 
            amps = dados[1:-1:2]
            return self.multi_harmonic_fault(freqs, amps)
        if index == 4 : # Rotor Problems
            dados = [Fr, np.random.uniform(0.8,1.2),
                    Fr2, np.random.uniform(0.2,0.6),
                    Fb-Fl4,np.random.uniform(0.2,0.4),
                    Fb-Fl2, np.random.uniform(0.5,.8),
                    Fb, np.random.uniform(1.2,1.8),
                    Fb + Fl2, np.random.uniform(0.5,.8),
                    Fb + Fl4, np.random.uniform(0.2,0.4),
                    Fb2-Fl4,np.random.uniform(0.2,0.8),
                    Fb2-Fl2, np.random.uniform(0.8,1.4),
                    Fb, np.random.uniform(1.2,1.8),
                    Fb + Fl2, np.random.uniform(0.5,.8),
                    Fb + Fl4, np.random.uniform(0.2,0.4),

                    ]
            freqs = dados[0:-1:2] 
            amps = dados[1:-1:2]
            return self.multi_harmonic_fault(freqs, amps)  
                
    def lubrification_fault(self, **kwargs):
        dc_carpet = 0*np.random.uniform(.5,2.5)
        Fa = 1/self.dt
        nfft = self.npto
        nfft2 = int(nfft/2)
        df = Fa/nfft
        f_ini = 1000 
        f_fim = 3000 
        if f_fim > Fa/2 : 
            f_fim = Fa/2 
        f_meio = 1500 + np.random.uniform(-100,100)
        f_meio2 = 2500 + np.random.uniform(-100,100)
        i_ini = int(f_ini/df)
        i_meio = int(f_meio/df)
        i_meio2 = int(f_meio2/df)
        i_fim = int(f_fim/df)
        x = [i_ini, i_meio, i_meio2, i_fim] 
        y = [0, np.random.uniform(1.8,2.6), np.random.uniform(1.8,2.6), 0 ] 
        tck  = interpolate.splrep(x,y, s=0) 
        z = np.arange(i_ini, i_fim, 1, dtype=int) 
        pz = interpolate.splev(z, tck, der=0)
        ruido = np.random.uniform(-.2, .2, size= len(pz))
        f_ini = 3000 
        f_fim = 6000 
        if f_fim > Fa/2 : 
            f_fim = Fa/2 
        f_meio = 3500 + np.random.uniform(-100,100)
        f_meio2 = 5000 + np.random.uniform(-100,100)
        f_meio3 = 5500 + np.random.uniform(-100,100)
        i_ini_1 = int(f_ini/df)
        i_meio = int(f_meio/df)
        i_meio2 = int(f_meio2/df)
        i_meio3 = int(f_meio3/df)
        i_fim_1 = int(f_fim/df)
        x = [i_ini_1, i_meio, i_meio2, i_meio3, i_fim_1] 
        y = [0, np.random.uniform(4,6), np.random.uniform(3,4), np.random.uniform(2,3), 0 ] 
        tck  = interpolate.splrep(x,y, s=0) 
        z = np.arange(i_ini_1, i_fim_1, 1, dtype=int) 
        pz1 = interpolate.splev(z, tck, der=0)
        ruido1 = np.random.uniform(-.6, .6, size= len(pz1))
        X = np.zeros(nfft).astype('complex') 
        X[1:nfft2] = dc_carpet*1/np.exp((np.arange(1,nfft2,1)/nfft2)) + dc_carpet*1/np.exp((np.arange(1,nfft2,1)/nfft2))*1j
        X[i_ini:i_fim] += .5*(pz*np.random.uniform(-.9,1.1) + ruido) + .5*(pz*np.random.uniform(-.9,1.1) + ruido)*1j 
        X[i_ini_1:i_fim_1] += .5*(pz1*np.random.uniform(-.7,1.4) + ruido1) + .5*(pz1*np.random.uniform(-.7,1.4) + ruido1)*1j 
        sinal= np.sign(np.random.normal(size=nfft2)) + np.sign(np.random.normal(size=nfft2))*1j
        X[0:nfft2] *= sinal
        X[nfft-1:nfft2+1:-1] = np.conj(X[1:nfft2-1])
        x= np.real(np.fft.fft(np.conj(X))/nfft)
        falha = np.concatenate((x,x,x))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-self.npto:]  
        return x_conv        

    

    def ressonance_fault(self, **kwargs):
        i_f0 = int(self.fs/self.freq_rotacao) 
        npto = self.npto 
        falha = np.zeros(npto) 
        i_ini = np.random.random_integers(0,i_f0) 
        falha[i_ini] = 1 
        falha = dys.utilities.frequency_filter(falha, 500, dt = self.dt, type='highpass')
        falha = np.concatenate((falha,falha,falha))
        x_conv = falha
        x_conv = np.convolve(falha, self.pseudo_tf, mode='same')
        x_conv = x_conv[-npto:]  
        return x_conv    
    
    def normal_signal(self, **kwargs) :
        data = np.copy(self.sinal )
        dt = self.dt
        G = 7
        ISO = 4.5
        # data: machine base signal vibration signal (m/s²)
        # dt = aquisition time (s)
        # G = grau de desbalanceamento permitido para a máquina 
        f0 = self.freq_rotacao 
        x = self.rpm_sinc_data(data) 
        npto = self.npto
        npto2 = int(npto/2)
        dt_new = self.dt_sinc
        df = 1/(npto*dt_new)
        i_0 = int(f0/df)  
        rms_v = self.velocity_10_1000(self.sinal, dt_new)   
        x = x*ISO*np.random.uniform(.5,1.1)/rms_v    
        # Aumentar harmonicos
        X = np.fft.fft(x/npto)
        X_clean = np.copy(X[0:npto2]) 
        i_1000 = int(1000/df)  
        if i_1000 > npto2 : 
            i_1000 = npto2-3
        Amp_rpm = G*(2*np.pi*f0)/10000
        for i in range( i_0,5*i_0,i_0) :
            X_clean[i-2:i+3] = Amp_rpm*np.random.uniform(.1,1)/ X_clean[i-2:i+3]
        x = np.zeros(npto).astype('complex')
        x[0:npto2] = X_clean
        x[npto-1:npto2+1:-1] = np.conj(X_clean[1:npto2-1])
        x= np.real(np.fft.fft(np.conj(x)))
        x = dys.utilities.changedeltat(x,dt_new,dt)[1] 
        x = x[0:npto]*np.random.uniform(.7,2)
        self.sinal = x
        return x
    
    def velocity_10_1000(self,data, dt) : 
        # velocidade 10:1000
        acel  = np.copy( data )               
        acel = dys.utilities.detrend(acel,dt, deg = 2)
        acel_filt = dys.utilities.frequency_filter(acel,[10,1000], dt,type='bandpass')
        v = dys.utilities.time_integration(acel_filt,dt)
        rms = 10000*dys.timeparameters.rms(v)
        return rms



