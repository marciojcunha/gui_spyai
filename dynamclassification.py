import dynamsignal as ds
from dynamsynthetic import SyntheticSignal
from dynamaugmentation import Augmentation
from dynamfeatures import Features_classification
from dynambase import Quality_Signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.saving import save_model, load_model
import joblib
import datetime
from dynamplot import plotar_graficos
nivel_alarme = [] 

def date_to_month(date) : 
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
        return seconds/2628288 # transforma para mês

class Classification:
    def __init__(self, path_sinal_base, path, path_v, dados_sinal, dados_cadastro, treinamento, data, idx_dados, sinal_teste = None):

        # Dados de entrada
        self.npz_path_sinal_base = path_sinal_base
        self.path = path
        self.data = data
        self.I =[]
        self.dados_sinal = dados_sinal
        self.dados_cadastro = dados_cadastro
        caminho = self.dados_sinal[0][0]
        partes = caminho.split( os.path.sep)
        self.tag = partes[-2]
        self.ponto = partes[-1]
        self.features_test = None
        self.fs = int(self.dados_sinal[0][5]/self.dados_sinal[0][6])
        self.valor_trip = dados_cadastro['valor_trip'].values
        self.bpfi = dados_cadastro['bpfi'].values
        self.bpfo = dados_cadastro['bpfo'].values
        self.bsf = dados_cadastro['bsf'].values
        self.ftf = dados_cadastro['ftf'].values
        self.estagio = dados_cadastro['estagio'].values
        self.z_motora = dados_cadastro['z_motora'].values
        self.z_motriz = dados_cadastro['z_motriz'].values
        self.rpm_entrada = dados_cadastro['rpm_entrada'].values
        self.rpm_saida = dados_cadastro['rpm_saida'].values
        self.bpf = dados_cadastro['bpf'].values
        self.num_pas = dados_cadastro['num_pas'].values
        self.num_polos = dados_cadastro['num_polos'].values
        self.num_barras = dados_cadastro['num_barras'].values
        rpm = self.dados_cadastro['rpm'].iloc[0]
        num_amostras = self.dados_sinal[0][5] 
        self.sinal = np.array(dados_sinal[0,idx_dados:num_amostras+idx_dados],dtype=float)
        self.sinal = self.sinal[~np.isnan(self.sinal)]
        self.dt = 1/self.fs
        bool_fs = True
        while self.fs < 4000: 
            npto = len(self.sinal) 
            sinal = np.zeros(2*npto)
            ii = 0 
            for i in range(npto-1) : 
                sinal[ii] = self.sinal[i] 
                sinal[ii+1] = .5*(self.sinal[i] + self.sinal[i+1] ) 
                ii +=2 
            self.fs *=2 
            self.dt = 1/self.fs
            self.sinal = sinal  
            bool_fs = False   
        self.rpm = ds.utilities.rpm_estimation(self.sinal, self.dt,[rpm-300, rpm], estimator='convolution')
        self.rps = self.rpm/60
        if path_v != None:
            files = os.listdir(path_v)
            found = next((f for f in files if 'Export_Time' in f and f.endswith('.npz')), None)
            if found is None:
                raise FileNotFoundError("Arquivo Export_Time*.npz não encontrado")
            dados_numpy = np.load(os.path.join(path_v, found), allow_pickle=True)
        self.dados_numpy = dados_numpy['data'][:]
        # tipos_falha = ['Normal','Rolamento_BPFI', 'Rolamento_BPFO', 'Rolamento_BSF', 'Rolamento_FTF', 'Desbalanceamento', 'Desalinhamento', 'Folga', 'Lubrificação']
        a=[2,6,9,12,16,21-34,36]
        I_low = [] 
        tipos_falha_low =[]
        # Seleção das features
        if np.isnan(self.bpfi) == False: # Tem rolamento
            self.I += [0, 1, 2, 3, 4, 5, 6,9,17,18,19,29,30 ]
            self.tipos_falha = ['Normal','Rolamento', 'Desalinhamento', 'Folga', 'Lubrificação']
            I_low += [0, 1, 3, 4, 5 ]
            tipos_falha_low += ['Normal','Rolamento', 'Desalinhamento', 'Folga']
        if np.isnan(self.z_motora) == False: # Tem engrenagem
            self.I += [0,1,2,3,4,21,22,23,24,25,26,29,30]
            self.tipos_falha = ['Normal','Engrenagem', 'Desalinhamento', 'Folga', 'Lubrificação']
            I_low += [0, 1, 3, 4, 5,21,22 ]
            tipos_falha_low += ['Normal','Engrenagem','Desalinhamento', 'Folga']
        if np.isnan(self.bpf) == False: # Tem correia
            self.I += [0,1,2,3,4,38,39,40]
            self.tipos_falha = ['Normal','Rolamento', 'Desalinhamento', 'Folga', 'Lubrificação', 'Polia']

        if np.isnan(self.num_polos) == False: # É motor
            self.I += [0,1,3,4,7,29,30]
            self.tipos_falha = ['Normal','Origem Elétrica','Desbalanceamento', 'Desalinhamento','Lubrificação', 'Folga']
            I_low += [0, 1, 3, 4, 5, 7]
            tipos_falha_low += ['Normal','Origem Elétrica','Desalinhamento', 'Folga']
        if np.isnan(self.bpfi) == False: # Não tem rolamento, é mancal de escorregamento
            self.I += [0,1,2,3,4,38,39,40]
            self.tipos_falha = ['Normal','Desbalanceamento', 'Desalinhamento', 'Folga','Lubrificação']

        if np.isnan(self.num_pas) == False: # É ventilador
            self.I += [0,1,3,4,6,13,29,30]
            self.tipos_falha = ['Normal','Desbalanceamento', 'Desalinhamento', 'Folga','Lubrificação']
            I_low += [0, 1, 3, 4, 5, 13]
            tipos_falha_low += ['Normal','Desalinhamento', 'Folga']

        if len(self.I) == 0:
            self.I += [0,1,2,3,4,7,38,39,40]
            self.tipos_falha = ['Normal','Desbalanceamento', 'Desalinhamento', 'Folga']
        else : 
            if not bool_fs : # Frequencia de aquisição baixa 
                self.I = I_low
                self.tipos_falha = tipos_falha_low


        # Remove duplicatas da lista
        self.I = list(dict.fromkeys(self.I))
        #treinamento = True
        # Treinar a rede
        if treinamento != False:
            self.falhas = []
            self.faults(sinal_teste=sinal_teste)
            self.signal_features()
            modelo_rf, pred_rf = self.random_classifier()
            self.signal()
            modelo_cnn, pred_cnn = self.cnn_classifier()
            modelo_bp, pred_bp = self.backpropagation_classifier()
            self.df_pred(modelo_rf, pred_rf, modelo_bp, pred_bp, modelo_cnn, pred_cnn)

        # Carregar modelo e classificar
        else:
            self.signal_features_test()
            self.signal_dado()

            caminho = self.dados_sinal[0][0]
            partes = caminho.split("\\")

#            print("\n=== CARREGANDO MODELOS ===")
            rf_model = joblib.load( os.path.join(self.path, f'modelo_rf_{partes[-2]}_{partes[-1]}.joblib'))
            bp_model = load_model(os.path.join(self.path, f'modelo_bp_{partes[-2]}_{partes[-1]}.keras'))
            cnn_model = load_model( os.path.join(self.path,f'modelo_cnn_{partes[-2]}_{partes[-1]}.keras'))
            self.backpropagation_scaler = np.load(os.path.join(self.path, 'backpropagation_scaler.npy'))

#            print("Modelos carregados com sucesso.\n")

#            print(">> Random Forest")
            pred_rf = self.randomforest_run(rf_model)
#            print(">> MLP (Backpropagation)")
            pred_bp = self.backpropagation_run(bp_model)
#            print(">> CNN")
            pred_cnn = self.cnn_run(cnn_model)      

            # Salvar no CSV final
            self.df_pred(rf_model, pred_rf, bp_model, pred_bp, cnn_model, pred_cnn)
    
    def features_sinal_base(self):
        feat_sinal_base = Features_classification( self.fs, self.rpm,
                        rot_ang= np.random.uniform(0,2*np.pi),
                        num_dentes_pinhao=self.z_motora,
                        num_dentes_coroa=self.z_motriz,
                        freq_rotac_pinhao=self.rpm / 60,
                        num_pas=self.num_pas,
                        num_polos=self.num_polos,
                        num_barras_rotor=self.num_barras,
                        bpfi=self.bpfi,
                        bpfo=self.bpfo,
                        bsf=self.bsf,
                        ftf=self.ftf,
                        bpf=self.bpf)                                             
        feat_sinal_base.calcular_features(np.array(self.sinal_base, dtype=float))
        feats = feat_sinal_base.features
        
        registro = {
            'TAG': self.tag,
            'Ponto': self.ponto,
            'DATA': '28/03/1960',
            'dt': self.dt,
            'Fs': self.fs,
            'RPM': self.rpm,
            **feats}
        registro = [registro]
        registro = pd.DataFrame(registro)
        self.npz_path_sinal_base = self.path + os.path.sep + 'features_sinal_base.npz'
        np.savez(self.npz_path_sinal_base, data=registro.values, columns=registro.columns)


    def faults(self, sinal_teste = None):
        fs = self.fs
        dt = 1/fs
        rpm = self.rpm
        num_sinteticos = 100
        fri = []
        sinal_teste = self.sinal
        # sinal_base = self.sinal
        # plotar_graficos_lucas(self.sinal,self.sinal,self.dt, title='Candidato a Sinal Base', subtitle=self.data)
        # choice = input('Quer limpar o Sinal Base? Sim: entre com 1; Não: qualquer outro valor')
        # if choice == '1' : 
        #     sinal_base = Quality_Signal.baseline_signal(self.sinal, dt, rpm)
        # else: 
        #     sinal_base = np.copy(self.sinal)    
        # self.sinal_base = sinal_base
        # np.save('sinal_base.npy', sinal_base) 
        
        self.sinal_base = Quality_Signal.baseline_signal(self.sinal, dt, rpm)
        self.features_sinal_base()
        #plotar_graficos(sinal_teste,self.sinal_base,self.dt, title='Candidato a Sinal Base', subtitle=self.data)
        rot_ang = np.random.normal()
        sintetizador = SyntheticSignal(self.sinal_base, fri,  rot_ang, fs=fs, freq_rotacao=rpm/60, sinal_original=sinal_teste)

        for tipo_falha in self.tipos_falha:
            n_s = num_sinteticos
            if tipo_falha == 'Normal' : 
                n_s = int(num_sinteticos/2)
            if tipo_falha == 'Desbalanceamento' : 
                n_s = int(num_sinteticos/2)
            if tipo_falha.find('Rolamento') > -1 :
                n_s = int(num_sinteticos/4)

            for i in range(1, n_s + 1):
                sinal_falha = sintetizador.add_fault(tipo_falha,
                            rot_and=rot_ang,
                            num_dentes_pinhao=self.z_motora,
                            num_dentes_coroa=self.z_motriz,
                            freq_rotac_pinhao=self.rpm / 60,
                            num_pas=self.num_pas,
                            num_polos=self.num_polos,
                            num_barras_rotor=self.num_barras,
                            bpfi=self.bpfi,
                            bpfo=self.bpfo,
                            bsf=self.bsf,
                            ftf=self.ftf,
                            bpf=self.bpf,
                            index_show = i)
                a = Augmentation(np.copy(sinal_falha), 2, fs)
                st = np.std(np.abs(sinal_falha))
                sinais_a = a.calcular_data_aug(st)
                # Lista para armazenar DataFrames parciais
                falha = tipo_falha 
                if tipo_falha == 'Desbalanceamento' : 
                    falha = 'Normal'
                if tipo_falha.find('Rolamento') > -1 :
                    falha = 'Rolamento'
    
                for a in range(len(sinais_a)):
                    for b in range(len(sinais_a[0])):
                        self.falhas.append([sinais_a[a][b],falha])
        i = 1            

    def signal_features_test( self ): 
        #plotar_graficos(self.sinal,self.sinal,self.dt)
        feat_class = Features_classification( self.fs, self.rpm,
                                rot_ang= np.random.uniform(0,2*np.pi),
                                num_dentes_pinhao=self.z_motora,
                                num_dentes_coroa=self.z_motriz,
                                freq_rotac_pinhao=self.rpm / 60,
                                num_pas=self.num_pas,
                                num_polos=self.num_polos,
                                num_barras_rotor=self.num_barras,
                                bpfi=self.bpfi,
                                bpfo=self.bpfo,
                                bsf=self.bsf,
                                ftf=self.ftf,
                                bpf=self.bpf)                                             
        feat_class.calcular_features(np.array(self.sinal, dtype=float))
        self.features_test = feat_class.features 

    def signal_features(self):
        self.df_features = []
        target1 = []
        features1 = [] 
        self.sinal_features = []
        feat_class = Features_classification( self.fs, self.rpm,
                                rot_ang= np.random.uniform(0,2*np.pi),
                                num_dentes_pinhao=self.z_motora,
                                num_dentes_coroa=self.z_motriz,
                                freq_rotac_pinhao=self.rpm / 60,
                                num_pas=self.num_pas,
                                num_polos=self.num_polos,
                                num_barras_rotor=self.num_barras,
                                bpfi=self.bpfi,
                                bpfo=self.bpfo,
                                bsf=self.bsf,
                                ftf=self.ftf,
                                bpf=self.bpf)                                             
        for i in range(len(self.falhas)):
            feat_class.calcular_features(np.array(self.falhas[i][0], dtype=float))
            features = feat_class.features
            target = {'target':self.falhas[i][1]}
            self.df_features.append([features,target])
            if i%100 == 0:
                print(f'{i}/{len(self.falhas)}')

        feat_class.calcular_features(np.array(self.sinal, dtype=float))
        self.features_test = feat_class.features 

    def reduc_freq(self,  X, lag  ) : 
        lag = int(lag)
        npto = len(X) 
        X2 = np.copy(X)**2
        X_r = np.zeros(int(npto/lag)) 
        ii = 0 
        for i in range(0,npto,lag) : 
            X_r[ii] = np.max(X2[i:i+lag]) 
            ii+=1
        return np.sqrt(X_r)    
    
    def signal(self, sinal_run = None):       
        self.sinal_falha_features = []
        self.sinal_falha_target = [] 
        npto =  int(len(self.sinal))
        df = 1/(self.dt*npto)
        i_10 = int(10 / df)
        i_1000 = int(1000 / df )  
        df_a = (.5/self.dt-1000)/(i_1000-i_10)
        nfft = int(1/(df_a*self.dt))
        i_a = i_1000-i_10   
      
        
        # Sinal com falha
        for i in range(len(self.falhas)):
            # Aceleração
            x = np.array(self.falhas[i][0], dtype = float)/(ds.timeparameters.rms(self.sinal)+1e-8)
            try : 
                dep_x = ds.utilities.dep_welch(x, self.dt, nfft )[1]
            except : 
                kk =1    
            # dep_x = self.reduc_freq(dep_x)
            dep_x/=10
            # Velocidade
            v = 1000 * ds.utilities.time_integration(x, self.dt)
            dep_v =  ds.utilities.fft_spectrum(v)/10
            # Envelope
            aux_env = ds.timeparameters.envelope(x, fc=1000, dt=self.dt)           
            dep_env =  ds.utilities.fft_spectrum(aux_env)/10
            dep_v = dep_v[i_10:i_1000] 
            dep_env = dep_env[i_10 : i_1000] 
            dep_x = dep_x[-i_a:]
            self.sinal_falha_features.append([dep_x,dep_v,dep_env])
            self.sinal_falha_target.append(self.falhas[i][1])

            if i%100 ==0 : 
                print(i)

    def signal_dado(self):        
        # Sinal dado 
        self.sinal_dado = []
            # Aceleração
        x = np.array(self.sinal, dtype = float)/(ds.timeparameters.rms(self.sinal)+1e-8)
        npto =  int(len(x))
        df = 1/(self.dt*npto)
        i_10 = int(10 / df)
        i_1000 = int(1000 / df )  
        df_a = (.5/self.dt-1000)/(i_1000-i_10)
        nfft = int(1/(df_a*self.dt))
        i_a = i_1000-i_10   

        dep_x = ds.utilities.dep_welch(x, self.dt, nfft )[1]
        # dep_x = self.reduc_freq(dep_x)
        dep_x/=10
        # Velocidade
        v = 1000 * ds.utilities.time_integration(x, self.dt)
        dep_v =  ds.utilities.fft_spectrum(v)/10
        # Envelope
        aux_env = ds.timeparameters.envelope(x, fc=1000, dt=self.dt)           
        dep_env =  ds.utilities.fft_spectrum(aux_env)/10
        dep_v = dep_v[i_10:i_1000] 
        dep_env = dep_env[i_10 : i_1000] 
        dep_x = dep_x[-i_a:]
        self.sinal_dado.append([dep_x,dep_v,dep_env])

    def boxplot(self):
        """
        Para cada feature já selecionada em self.selected_features,
        gera um boxplot comparando a distribuição dessa feature entre
        as diferentes classes (targets).
        """
        # 1) Monta um DataFrame com todas as features + coluna 'target'
        df = pd.DataFrame(
            [feat for feat, tgt in self.df_features]
        )
        df['target'] = [t['target'] for _, t in self.df_features]

        # 2) Garante que self.select_features() já tenha sido chamado
        if not hasattr(self, 'selected_features'):
            self.select_features()

        # 3) Cria um subplot por feature
        n = len(self.selected_features)
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        axes = axes.flatten()

        for i, feat in enumerate(self.selected_features):
            ax = axes[i]
            # boxplot da feature por classe
            df.boxplot(column=feat, by='target', ax=ax)
            ax.set_title(feat)
            ax.set_xlabel('Classe')
            ax.set_ylabel('Valor')
            # remove título gerado pelo pandas
            ax.get_figure().suptitle('')

        # 4) Remove eixos extras
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.suptitle('Distribuição das Features por Classe', y=1.02)
        plt.show()


    def select_features(self, I = None):
        if I is None : 
            indices = np.arange(1,len(self.df_features[0][0])).tolist()
        else : 
            indices = I    
        all_keys = list(self.df_features[0][0].keys())
        self.selected_features = [all_keys[i] for i in indices if 0 <= i < len(all_keys)]

    def randomforest_run( self, clf):
        # Codifica os rótulos
        le = LabelEncoder()
        le.fit_transform(self.tipos_falha)
        # Predição com o sinal 
        feat_class = Features_classification( self.fs, self.rpm,
                                rot_ang= np.random.uniform(0,2*np.pi),
                                num_dentes_pinhao=self.z_motora,
                                num_dentes_coroa=self.z_motriz,
                                freq_rotac_pinhao=self.rpm / 60,
                                num_pas=self.num_pas,
                                num_polos=self.num_polos,
                                num_barras_rotor=self.num_barras,
                                bpfi=self.bpfi,
                                bpfo=self.bpfo,
                                bsf=self.bsf,
                                ftf=self.ftf,
                                bpf=self.bpf)                                             
        feat_class.calcular_features(np.array(self.sinal, dtype=float))
        self.features_test = feat_class.features 
        features = self.features_test
        # Seleção de mesmas features usadas no treino (indices pré-definidos)
        all_keys = list(features.keys())
        indices = self.I
        selected = [all_keys[i] for i in indices if i < len(all_keys)]
        X_test = np.array([[features[f] for f in selected]])
        pred = clf.predict(X_test)[0]
        classe_predita = le.inverse_transform([pred])[0]
#        print(clf.predict_proba(X_test))
#        print('rf = %s'%(classe_predita))
        return pred


    def random_classifier(self):
        # Prepara os dados de entrada X e os rótulos y
        I = self.I
        self.select_features(I)
        X = np.array([[row[0][feat] for feat in self.selected_features] for row in self.df_features])
        y = [row[1]['target'] for row in self.df_features]

        # Codifica os rótulos
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Treina o modelo
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y_encoded)

        # Predição com o sinal real
        features = self.features_test
        # Seleção de mesmas features usadas no treino (indices pré-definidos)
        all_keys = list(features.keys())
        indices = I
        selected = [all_keys[i] for i in indices if i < len(all_keys)]
        X_test = np.array([[features[f] for f in selected]])
        pred = clf.predict(X_test)[0]
#        print(clf.predict_proba(X_test))
        classe_predita = le.inverse_transform([pred])[0]
        a = max(clf.predict_proba(X_test)[0])
        prob = a*100
        classe_predita = '%s (%.2f %%)'%(classe_predita, prob) 
        #plotar_graficos(self.sinal_base, self.sinal, self.dt)
        
#        print(f'RANDOM FOREST: {classe_predita}')

        return clf, pred


    def cnn_run(self, model) : 
        le = LabelEncoder()
        le.fit_transform( self.tipos_falha)        
        # Previsão para o sinal dado (teste)
        self.signal_dado()
        sinais = np.array( self.sinal_dado, dtype=float)
        X_test = np.array([np.stack(s, axis=-1) for s in sinais])        
        pred_proba = model.predict(X_test, verbose=0)
#        print(pred_proba )
        pred_class = np.argmax(pred_proba, axis=1)
        pred_label = le.inverse_transform(pred_class)[0]

#        print(f'CNN: {pred_label}')

        return pred_proba

    
    def cnn_classifier(self):
        # Extrair sinais e labels para treino
        sinais = np.array( self.sinal_falha_features, dtype=float)
        labels = self.sinal_falha_target

        # Converter para numpy arrays (stack nas features para ter shape [samples, timesteps, features])
        X = np.array([np.stack(s, axis=-1) for s in sinais])
        y = np.array(labels)

        # Label encoding e one-hot
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        y_cat = to_categorical(y_enc)

        # Dividir em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.1, random_state=42)

        # Criar modelo CNN 1D
        model = Sequential([
            Conv1D(32, 5, activation='relu', input_shape=(X.shape[1], X.shape[2])),
            MaxPooling1D(2),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(y_cat.shape[1], activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Treinar com EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)


        # Previsão para o sinal dado (teste)
        self.signal_dado()
        sinais = np.array( self.sinal_dado, dtype=float)
        X_test = np.array([np.stack(s, axis=-1) for s in sinais])
        
        pred_proba = model.predict(X_test, verbose = 0)
        pred_class = np.argmax(pred_proba, axis=1)
        pred_label = le.inverse_transform(pred_class)[0]

        a = max(pred_proba[0])
        prob = a*100
        pred_label = '%s (%.2f %%)'%(pred_label, prob)

#        print(f'CNN: {pred_label}')

        return model, pred_proba
    
    def df_alarme(self):
        rms_v = self.features_test['v_10_1000']*ds.timeparameters.rms(self.sinal) 
        if rms_v <= 2.8:
            alerta = 0 
        elif rms_v <= 4.5:
            alerta  = 1 
        elif rms_v <= 7.1:
            alerta = 2
        else:
            alerta = 3 

        features_base = np.load(self.npz_path_sinal_base, allow_pickle=True)
        aux = np.array(features_base['data'][0,6:],dtype=float)
        features_base = aux[self.I]
        features_test = np.array(list(self.features_test.values()))[self.I]
        alerta_aux = np.zeros(len(features_base))
        for i in range(len(features_base)) : 
            ratio = 20*np.log10(features_test[i]/(features_base[i]+1e-12)) 
            alerta_aux[i] = 0 
            if ratio > 6 : 
                alerta_aux[i] = 1 
            if ratio > 10 : 
                alerta_aux[i] = 2 
            if ratio > 20 : 
                alerta_aux[i] =3              
        alerta_aux = np.max(alerta_aux)   
        
        if alerta_aux > alerta : 
            alerta = alerta_aux             
            ratio= 20*np.log10(features_test/(features_base+1e-12))
            print('Alerta = %d, %.1f'%(np.argmax(ratio),np.max(ratio)))
        else : 
            print('Alerta = rms ' )   

        return alerta, rms_v                     





    def df_pred(self, modelo_rf, pred_rf, modelo_bp, pred_bp, modelo_cnn, pred_cnn):
        # Definindo nomes dos arquivos dos modelos
        # extrai o caminho uma vez
        caminho = self.dados_sinal[0][0]

        # separa os componentes (aqui você realmente usa "\\" como separador)
        partes = caminho.split( "\\")

        # monta o nome do arquivo usando f-string só para interpolar variáveis sem backslashes nas chaves
        nome_modelo_rf =  os.path.join(self.path, f'modelo_rf_{partes[-2]}_{partes[-1]}.joblib')
        nome_modelo_bp = os.path.join(self.path, f'modelo_bp_{partes[-2]}_{partes[-1]}.keras')
        nome_modelo_cnn = os.path.join(self.path,f'modelo_cnn_{partes[-2]}_{partes[-1]}.keras')
        nome_scaler =os.path.join(self.path,'backpropagation_scaler.npy')

        # Salvando modelos
        joblib.dump(modelo_rf, nome_modelo_rf)
        #joblib.dump(modelo_bp, nome_modelo_bp)
        #joblib.dump(modelo_cnn, nome_modelo_cnn)
        save_model(modelo_bp, nome_modelo_bp)
        save_model(modelo_cnn, nome_modelo_cnn)
        np.save(nome_scaler, self.backpropagation_scaler)

        data = self.dados_sinal[0][1]

        # Montar DataFrame com as informações
        tag = partes[-2]
        ponto = partes[-1]
        desbalanceamento = self.features_test['harm1']*ds.timeparameters.rms(self.sinal)
        pred_desb = 'Não'
        pred = '----' # concertar
        data_mes1 = date_to_month(self.data)
        data_mes2 = np.zeros(len(self.dados_numpy)) 
        for i in range(len(data_mes2)) : 
            data_mes2[i] = date_to_month( self.dados_numpy[i,1] )
        i = np.argmin(np.abs(data_mes2-data_mes1))     
        if np.abs(data_mes1-data_mes2[i]) < 1 :             
            npto = self.dados_numpy[i,5]
            T = self.dados_numpy[i,6] 
            dt = T/npto  
            acel = np.array(self.dados_numpy[i,10:10+npto], dtype=float) 
            Acel = ds.utilities.fft_spectrum(acel)**2 
            rps = self.rps
            df = 1/(npto*dt) 
            i_0 = int(rps/df) 
            rms_v = 10000*np.sqrt( np.sum(Acel[i_0-2:i_0+2]))/(2*np.pi*rps) 
            ratio = (rms_v+1e-8)/desbalanceamento 
            if ratio < 1 :
                ratio = 1/ratio
                ratio = 'V/H = %.1f'%(ratio)
            else :
                ratio = 'H/V = %.1f'%(ratio)
        else : 
            ratio = 'H/V = %.1f'%(1.0) 

        if desbalanceamento >= 3.5:
            pred_desb = '%.1f [mm/s]'%(desbalanceamento) 

        le = LabelEncoder()
        le.fit_transform( self.tipos_falha)
        falhas = le.inverse_transform(np.arange(0,len(pred_cnn[0])))
        pred = (np.array(pred_rf) + np.array(pred_bp) + np.array(pred_cnn))[0]/3

        for i in range(len(pred)) : 
            if pred[i] < .01 : 
                pred[i] = .01
        
        alerta, rms_v = self.df_alarme()

        # Porcentagem de normalidade                             
        i_max = np.argmax(pred) 
        i_normal =  np.strings.find( falhas, 'Normal') 
        if np.max(i_normal) > -1 : 
            falha_normal = True 
            i_normal =  np.argmax(i_normal)
        else : 
            falha_normal = False    
        if alerta > 0 : 
            if falha_normal : 
                pred[i_normal] = 0 
                sum = np.sum(pred) 
                pred = pred/sum                            
        else : 
            val_normal = .3*(2.8-rms_v)/2.8+.5 
            if falha_normal : 
                pred[i_normal] = 0 
                sum = np.sum(pred) 
                pred *= (1-val_normal)/sum 
                pred[i_normal] = val_normal
            else : 
                i_pos = np.argmin(pred) 
                falhas[i_pos] = 'Normal' 
                pred[i_pos] = 0 
                sum = np.sum(pred) 
                pred *= (1-val_normal)/sum 
                pred[i_pos] = val_normal                                   
        if desbalanceamento >= 3.5 :
            val_desb = .3*(10-desbalanceamento)/10+.5 
            if falha_normal :
                falhas[i_normal] = 'Desbalanceamento'  
                pred[i_normal] = 0 
                sum = np.sum(pred) 
                pred *= (1-val_desb)/sum 
                pred[i_normal] = val_desb
            else : 
                i_pos = np.argmin(pred) 
                falhas[i_pos] = 'Desbalancemanento' 
                pred[i_pos] = 0 
                sum = np.sum(pred) 
                pred *= (1-val_desb)/sum 
                pred[i_pos] = val_desb                  
        index = np.argsort(pred) 
        falhas = falhas[index] 
        pred = pred[index] 
        sum = 0 
        for i in range(1,4) : 
            sum += pred[-i] 
        for i in range(1,4) : 
            pred[-i] /= sum            
        pred_1 ='%s (%.1f %%)'%(falhas[-1],100*pred[-1])
        pred_2 ='%s (%.1f %%)'%(falhas[-2],100*pred[-2])
        pred_3 ='%s (%.1f %%)'%(falhas[-3],100*pred[-3]) 
        erro_sinal = False
        if ds.utilities.signal_defect_detector(self.sinal, self.dt) :  
            erro_sinal = True  
            # plotar_graficos(self.sinal, self.sinal, self.dt, title='Erro de Sinal', subtitle = self.data)     
            # a = input('concorda, entre com 1')
            # if a == '1' :
            #     erro_sinal = True
        #plotar_graficos_lucas(self.sinal, self.sinal, self.dt, title=txt, subtitle = self.data)
        if erro_sinal :
            df_novo = pd.DataFrame({'TAG': [tag],'PONTO': [ponto],'DATA': [data],'Alerta':[ '--' ],'TIPO FALHA - PROB. 1': ['Sinal'],'TIPO FALHA - PROB. 2': [ '--'],'TIPO FALHA - PROB. 3': ['--'], 'DESB.': ['--'], 'Razão 1X ': ['--'], 'MODELO_RF_PATH': [nome_modelo_rf],'MODELO_BP_PATH': [nome_modelo_bp],'MODELO CNN_PATH': [nome_modelo_cnn]})
        else :
            df_novo = pd.DataFrame({'TAG': [tag],'PONTO': [ponto],'DATA': [data],'Alerta':[alerta],'TIPO FALHA - PROB. 1': [pred_1],'TIPO FALHA - PROB. 2': [pred_2],'TIPO FALHA - PROB. 3': [pred_3], 'DESB.': [pred_desb], 'Razão 1X ': [ratio], 'MODELO_RF_PATH': [nome_modelo_rf],'MODELO_BP_PATH': [nome_modelo_bp],'MODELO CNN_PATH': [nome_modelo_cnn]})

        caminho_csv = os.path.join(self.path,'SPYAI_pred.csv')
        caminho_npz = os.path.join(self.path, 'SPYAI_pred.npz')

        if os.path.exists(caminho_csv):
            df_existente = pd.read_csv(caminho_csv)
            df_final = pd.concat([df_existente, df_novo], ignore_index=True)
        else:
            df_final = df_novo

        df_final.to_csv(caminho_csv, index=False)
        np.savez(caminho_npz, data=df_final.values, columns=df_final.columns)
#        print(caminho_csv)


    def norma_z(self,X_orig) :
        X = np.copy(X_orig) 
        n2 = int(len(self.backpropagation_scaler)/2) 
        Xb = self.backpropagation_scaler[0:n2] 
        St = self.backpropagation_scaler[n2:]  
        n_col = X.shape[1] 
        for i in range(n_col) : 
            aux = X[:,i] 
            aux = (aux-Xb[i])/(St[i]+1e-4) 
            aux[aux>3] = 3 
            aux[aux<-3] = -3
            X[:,i] = aux
        return X     

    def backpropagation_run(self,model) :
        # 6. Prever o x_input
        features = self.features_test
        # Codifica os rótulos
        le = LabelEncoder()
        le.fit_transform(self.tipos_falha)        
        # Seleção de mesmas features usadas no treino (indices pré-definidos)
        all_keys = list(features.keys())
        indices = self.I
        selected = [all_keys[i] for i in indices if i < len(all_keys)]
        x_input = np.array([[features[f] for f in selected]])
        x_input = self.norma_z(x_input)
        pred = model.predict(x_input, verbose = 0)
        pred_idx = np.argmax(pred, axis=1)[0]
        classe_predita = le.inverse_transform([pred_idx])[0]
#        print( pred )
#        print(f'MLP: {classe_predita}')
        return pred


    def backpropagation_classifier(self):

        # 1. Separar features e alvo
        self.select_features( self.I )
        X = np.array([[row[0][feat] for feat in self.selected_features] for row in self.df_features])
        y = [row[1]['target'] for row in self.df_features]

        # 2. Codificar o target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        from sklearn.preprocessing import MinMaxScaler

        # 3. Dividir em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.3, random_state=42)

        Xb = np.mean(X_train, axis=0)
        St = np.std(X_train, axis=0)
        self.backpropagation_scaler = np.concatenate((Xb,St), axis=0) 
        X_train = self.norma_z(X_train)
        X_val = self.norma_z(X_val)
        X_test = self.norma_z(X_test) 
        # 4. Criar o modelo MLP
        model = Sequential()
        model.add(Dense(120, input_shape=(X.shape[1],),activation='relu', kernel_initializer= initializers.GlorotNormal()  , bias_initializer= initializers.Zeros()))
        model.add(Dropout(.4))
        model.add(Dense(60,activation='relu', kernel_initializer= initializers.GlorotNormal()  , bias_initializer= initializers.Zeros()))
        model.add(Dropout(.4))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(.4))
        model.add(Dense(y_categorical.shape[1], activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 5. Treinar com early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=16, callbacks=[early_stopping], verbose=1)
        # 
        pred = model.predict(X_test, verbose=0)
        pred_idx = np.argmax(pred, axis=1)[0]
        classe_predita = le.inverse_transform([pred_idx])[0]
        # Predição para treino e validação para a matriz de confusão
        X_test = self.norma_z(X)
        y_test = y_categorical
        y_train_pred = np.argmax(model.predict(X_test,verbose=0), axis=1)
        y_train_true = np.argmax(y_test, axis=1)
        # cm = confusion_matrix(y_train_true, y_train_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.tipos_falha).plot()        
        # #disp.plot(cmap=plt.cm.Blues)
        # plt.title("Matriz de Confusão - Treino MLP")
        # plt.show()
 
        # 6. Prever o x_input
        features = self.features_test
        # Seleção de mesmas features usadas no treino (indices pré-definidos)
        all_keys = list(features.keys())
        indices = self.I
        selected = [all_keys[i] for i in indices if i < len(all_keys)]
        x_input = np.array([[features[f] for f in selected]])
        x_input = self.norma_z(x_input)
        pred = model.predict(x_input,verbose=0)
        pred_idx = np.argmax(pred, axis=1)[0]
        classe_predita = le.inverse_transform([pred_idx])[0]
        a = max(pred[0])
        prob = a*100
        classe_predita = '%s (%.2f %%)'%(classe_predita, prob)

#        print(f'MLP: {classe_predita}')

        return model, pred
