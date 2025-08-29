import numpy as np

class Augmentation:
    def __init__( self , data, quantidade_sinais = 100, fs = 20480 ) : 
        self.data = np.array(data) # sample time interval [s]
        self.quantidade_sinais = quantidade_sinais # quantidade de sinais a serem gerados
        self.fs = fs

        
    def add_noise ( self, noise_factor ) :
        noise = np.random.randn(len(self.data))
        augmented_data = self.data + noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(self.data[0]))
        
        return augmented_data

    def masking_noise ( self, porcentagem_zeros ) :
        quantity = int(porcentagem_zeros*len(self.data)/100)
        noise = np.random.choice(len(self.data),size=quantity,replace=False) 
        augmented_data = self.data.copy()
        augmented_data[noise] = 0
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(self.data[0]))
        
        return augmented_data

    def frequency_add ( self ,qta_freq_final) :
        
        dt = 1/self.fs
        t_total = len(self.data)/self.fs
        t = np.arange(0,t_total,dt)

        qta_freq = int(np.random.uniform(0,qta_freq_final))
        xAmp = np.mean(self.data[np.where(self.data > 0.9*np.max(self.data))]) # Amplitude do sinal para gerar sinal teórico compatível
        augmented_data = np.copy(self.data)

        for i in range (0, qta_freq) :
            ampl_freq = (np.random.uniform(0,.1*xAmp))
            freq = (np.random.uniform(10,self.fs/4)) # frequencias
            xFrq = ampl_freq*np.sin(2*np.pi*freq*t)
            augmented_data += xFrq

        return augmented_data

    def calcular_data_aug ( self,std) :
        noise_factor = np.random.uniform(0,.3*std,self.quantidade_sinais)
        porcentagem_zeros = np.random.uniform(0,30,self.quantidade_sinais)
        rates = np.random.randint(10,30,self.quantidade_sinais)
        rr = []
        mm = []
        ff = []

        for i in range(0,self.quantidade_sinais):
            ruido_fator = noise_factor[i]
            #porc_noise = int(porcentagem_zeros[i])
            rate = rates[i]

            r = self.add_noise(ruido_fator)
            #m = self.masking_noise(porc_noise)
            f = self.frequency_add(rate)

            rr.append(r)
            mm.append( f )
            ff.append(.5*(r+f))

        return rr, mm, ff
