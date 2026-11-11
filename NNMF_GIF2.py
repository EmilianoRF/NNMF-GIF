import numpy as np
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import Funciones as Funciones

class NNMF_GIF:
    def __init__(self, speech, acelerometro, dEGG,tiempos,mascara_tiempos,fs,
                 ventana='Rectangular',
                 duracion=5.5,
                 hop=1,
                 norma='frobenius',
                 pre_iteraciones=0,
                 iteraciones=250,
                 T = 10,
                 delta = 10e-4,
                 p_filtro_speech=12,
                 alfa_speech=0.96,
                 p_filtro_acc=12,
                 alfa_acc=0.96,
                 beta_speech = 0.55,
                 beta_acc    = 0.55,
                 pre_enfasis_speech = True,
                 pre_enfasis_acc    = True,
                 doble_pre_enfasis_acc = False,
                 guardar_proceso = True,
                 normalizar_espectrogramas = True,
                 nombre_archivo = ''):

        # ============ Parámetros por defecto
        self.speech = speech
        self.speech_pre_enfasis = -1
        self.acc  = acelerometro
        self.acc_pre_enfasis = -1
        self.dEGG = dEGG
        self.fs     = fs
        self.tiempos = tiempos
        self.mascara_tiempos = mascara_tiempos
        self.duracion_ventana = duracion # En ms
        self.long_ventana_espect = int(np.round(fs * duracion / 1000))
        self.desplazamiento   = hop    
        self.ventana = ventana
        self.beta_speech = beta_speech
        self.alfa_speech = alfa_speech
        self.orden_filtro_tracto_speech = p_filtro_speech
        self.beta_acc = beta_acc
        self.alfa_acc = alfa_acc
        self.orden_filtro_tracto_acc = p_filtro_acc
        self.norma         = norma
        self.n_iteraciones = iteraciones
        self.n_iteraciones_realizadas = -1
        self.n_pre_itereraciones = pre_iteraciones
        self.T = T
        self.delta = delta
        self.pre_enfasis_speech = pre_enfasis_speech
        self.pre_enfasis_acc = pre_enfasis_acc
        self.doble_pre_enfasis_acc = doble_pre_enfasis_acc       
        self.guardar_proceso = guardar_proceso
        self.normalizar_espectrogramas = normalizar_espectrogramas
        self.lag = None
        self.fo_espectrograma = 0
        self.ff_espectrograma = fs / 2
        self.nombre_archivo = nombre_archivo
        #-----------------------------------------------------------------------------------
        self.tiempos_sinc = -1

        self.pre_itereraciones = -1
        self.itereraciones = -1

        self.error_NNMF = -1
        self.matriz_STFT = -1
        self.espectrograma = -1
        self.espectrograma_plot = -1
        self.espectrograma_aprox_plot = -1
        self.espectrograma_speech = -1
        self.espectrograma_acc = -1
        self.to_espec = -1
        self.tf_espec = -1


        self.tiempos_espec = -1
        self.tiempos_eH_speech = -1
        self.tiempos_eH_acc = -1
        self.fo_espec = -1
        self.ff_espec = -1
        self.espectrograma_aprox = -1
        self.energia_speech = -1

        self.H_i =  -1 
        self.W_i =  -1
        self.W1_speech = -1
        self.W2_speech = -1
        self.W1_acc = -1
        self.W2_acc = -1
        self.H1 = -1
        self.H2 = -1

        self.a_W1_speech = -1
        self.a_W2_speech = -1
        self.polos_W1_speech = -1
        self.polos_W1_speech_originales = -1
        self.ceros_W1_speech = -1
        self.polos_W2_speech = -1
        self.polos_W2_speech_originales = -1
        self.ceros_W2_speech = -1

        
        self.a_W1_acc = -1
        self.a_W2_acc = -1
        self.polos_W1_acc = -1
        self.polos_W1_acc_originales = -1
        self.ceros_W1_acc = -1
        self.polos_W2_acc = -1
        self.polos_W1_acc_originales = -1
        self.ceros_W2_acc = -1

        self.flujo1_speech = -1
        self.flujo2_speech = -1
        self.dflujo1_speech = -1
        self.dflujo2_speech = -1

        self.flujo1_acc = -1
        self.flujo2_acc = -1
        self.dflujo1_acc = -1
        self.dflujo2_acc = -1

        self.error_predic1_speech = -1
        self.error_predic2_speech = -1
        self.e1H1_speech =-1
        self.e2H2_speech =-1
        self.sum_e1H1_speech = -1
        self.sum_e2H2_speech = -1

        self.error_predic1_acc = -1
        self.error_predic2_acc = -1
        self.e1H1_acc =-1
        self.e2H2_acc =-1
        self.sum_e1H1_acc = -1
        self.sum_e2H2_acc = -1

        self.dif_error_X = -1

        self.dif_error_W1_speech = -1
        self.dif_error_W2_speech = -1
        self.dif_error_W1_acc = -1
        self.dif_error_W2_acc = -1



        #-----------------------------------------------------------------------------------
        self.proceso_espectrograma = []
        self.proceso_espectrograma_aprox = []
        self.proceso_espectrograma_aprox_plot = []
        self.proceso_dif_error_X = []
        self.proceso_H1 = []
        self.proceso_H2 = []
        self.proceso_dif_error_H1  = []
        self.proceso_dif_error_H2   = []


        self.proceso_W1_speech = []
        self.proceso_W2_speech = []

        self.proceso_a_W1_speech = []
        self.proceso_a_W2_speech = []
        self.proceso_polos_W1_speech = []
        self.proceso_polos_W1_speech_originales = []
        self.proceso_ceros_W1_speech = []
        self.proceso_polos_W2_speech = []
        self.proceso_polos_W2_speech_originales = []
        self.proceso_ceros_W2_speech = []

        self.proceso_flujo1_speech = []
        self.proceso_flujo2_speech = []
        self.proceso_dflujo1_speech = []
        self.proceso_dflujo2_speech = []

        self.proceso_error_predic1_speech = []
        self.proceso_error_predic2_speech = []
        self.proceso_e1H1_speech = []
        self.proceso_e2H2_speech = []
        self.proceso_sum_e1H1_speech = []
        self.proceso_sum_e2H2_speech = []

        self.proceso_dif_error_W1_speech = []
        self.proceso_dif_error_W2_speech = []


        #---
        self.proceso_W1_acc = []
        self.proceso_W2_acc = []


        self.proceso_a_W1_acc = []
        self.proceso_a_W2_acc = []
        self.proceso_polos_W1_acc = []
        self.proceso_polos_W1_acc_originales = []
        self.proceso_ceros_W1_acc = []
        self.proceso_polos_W2_acc = []
        self.proceso_polos_W2_acc_originales = []
        self.proceso_ceros_W2_acc = []

        self.proceso_flujo1_acc = []
        self.proceso_flujo2_acc = []
        self.proceso_dflujo1_acc = []
        self.proceso_dflujo2_acc = []

        self.proceso_error_predic1_acc = []
        self.proceso_error_predic2_acc = []
        self.proceso_e1H1_acc = []
        self.proceso_e2H2_acc = []
        self.proceso_sum_e1H1_acc = []
        self.proceso_sum_e2H2_acc = []

        self.proceso_dif_error_W1_acc = []
        self.proceso_dif_error_W2_acc = []


        # ====================================
        self.Run()


#======================================================  Funciones auxiliares 

    def _actualizar_WH(self, W, H, X, X_aprox, norma):
        if norma == 'frobenius':
            W = W * (X @ H.T) / (W @ (H @ H.T))
            H = H * (W.T @ X) / ((W.T @ W) @ H)
        elif norma == 'itakura-saito':
            W = W * ((X_aprox**(-2) * X) @ H.T) / (X_aprox**(-1) @ H.T)
            H = H * (W.T @ (X * X_aprox**(-2))) / (W.T @ X_aprox**(-1))
        H = H / H.sum(axis=0)
        return W, H

    def _calcular_error(self, X, X_aprox, norma):
        if norma == 'frobenius':
            return np.linalg.norm(X - X_aprox, ord='fro')
        elif norma == 'itakura-saito':
            return np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))

    def _iterar(self, W, H, X, norma, num_iter, error,
                dif_error_X, dif_error_W1_speech, dif_error_W2_speech,dif_error_W1_acc, dif_error_W2_acc, dif_error_H1, dif_error_H2):
        X_aprox = W @ H


        for t in range(num_iter):
            W_old, H_old = W.copy(), H.copy()
            W, H = self._actualizar_WH(W, H, X, X_aprox, norma)
            X_aprox = W @ H
            e = self._calcular_error(X, X_aprox, norma)
            error.append(e)


            N = W.shape[0]
            mid = N // 2  
            dif_error_W1_speech.append(
                np.linalg.norm(W_old[:mid,0] - W[:mid,0]) / np.linalg.norm(W_old[:mid,0])
            )
            dif_error_W2_speech.append(
                np.linalg.norm(W_old[:mid,1] - W[:mid,1]) / np.linalg.norm(W_old[:mid,1])
            )

            dif_error_W1_acc.append(
                np.linalg.norm(W_old[mid:,0] - W[mid:,0]) / np.linalg.norm(W_old[mid:,0])
            )
            dif_error_W2_acc.append(
                np.linalg.norm(W_old[mid:,1] - W[mid:,1]) / np.linalg.norm(W_old[mid:,1])
            )

            dif_error_H1.append(np.linalg.norm(H_old[0,:]-H[0,:])/np.linalg.norm(H_old[0,:]))
            dif_error_H2.append(np.linalg.norm(H_old[1,:]-H[1,:])/np.linalg.norm(H_old[1,:]))

            if len(error) > 1:
                dif_error_X.append(abs((error[-1]-error[-2])/error[-1]))

            if self.guardar_proceso:
                self.proceso_espectrograma_aprox.append(X_aprox)
                self.proceso_espectrograma_aprox_plot.append(np.vstack((X_aprox[mid:,:], X_aprox[:mid,:])))
                self.proceso_W1_speech.append(W[:mid,0])
                self.proceso_W2_speech.append(W[:mid,1])
                self.proceso_W1_acc.append(W[mid:,0])
                self.proceso_W2_acc.append(W[mid:,1])
                self.proceso_H1.append(H[0,:])
                self.proceso_H2.append(H[1,:])
            if t >=self.T and t>=100:
                if abs(error[t-self.T] - error[t]) <= self.delta*error[t]:
                    self.n_iteraciones = t
                    break
        return W, H, X_aprox

    def _resolver_toeplitz(self, W, orden_filtro):
        W_  = np.concatenate((W, np.flip(W)[1:-1]))
        r_W = np.fft.ifft(W_).real
        a_  = solve_toeplitz(
            (r_W[0:orden_filtro], r_W[0:orden_filtro]),
            r_W[1:orden_filtro+1]
        )
        a_W_originales = np.concatenate(([1.0], -a_))
        num = np.zeros(len(a_W_originales))
        num[0] = 1
        ceros_W, polos_W, _ = sig.tf2zpk(num, a_W_originales)

        polos_originales = polos_W
        polos_ = [p for p in polos_W if not (((p.real > 0.0) and (p.imag == 0.0)) or ((p.real > 0.0) and (np.abs(p) < 0.8)))]

        b, a = sig.zpk2tf(np.zeros(len(polos_)), polos_, k=1)
        a_W = a

        ganancia_dc = np.sum(a_W)   # A(z=1)
        if np.abs(ganancia_dc) > 1e-8:
            a_W = a_W / ganancia_dc

        polos_W = polos_
        return a_W, polos_W, ceros_W, a_W_originales,polos_originales

    def _normalizar(self, señal):
        maximo = np.max(np.abs(señal))
        return señal / maximo if maximo > 1 else señal

    def _calcular_flujo(self, a_W, señal,alfa):
        flujo = sig.lfilter(a_W, [1, -alfa], señal)
        return self._normalizar(flujo)


    def _filtrado_inverso(self, a_W,señal,alfa):
        dflujo = sig.lfilter(a_W, [1], señal)
        flujo = sig.lfilter([1], [1, -alfa], dflujo)     
        #dflujo = dflujo[self.mascara_tiempos]
        dflujo = dflujo/abs(min(dflujo))
        #flujo = flujo[self.mascara_tiempos]
        flujo = Funciones.Normalizar(flujo)
        return dflujo,flujo

    def _calcular_error_prediccion(self, a_W, señal):
        ep = sig.lfilter(b=a_W, a=[1], x=señal)
        return ep

    def _seleccionar_candidato(self, H1, H2,
                               a_W1_speech, a_W2_speech,
                               polos_W1_speech,polos_W2_speech,
                               polos_W1_originales_speech,polos_W2_originales_speech,
                               ceros_W1_speech,ceros_W2_speech,
                               flujo_w1_speech, flujo_w2_speech,
                               dflujo_w1_speech, dflujo_w2_speech,
                               ep1_speech, ep2_speech,
                               a_W1_acc, a_W2_acc,
                               polos_W1_acc,polos_W2_acc,
                               polos_W1_originales_acc,polos_W2_originales_acc,
                               ceros_W1_acc,ceros_W2_acc,
                               flujo_w1_acc, flujo_w2_acc,
                               dflujo_w1_acc, dflujo_w2_acc,
                               ep1_acc, ep2_acc, i=None):


        ep1_speech  = ep1_speech[self.long_ventana_espect-1:]
        ep1_speech  =  ep1_speech/max(abs(ep1_speech))
        e1H1_speech = H1*np.abs(ep1_speech)
        ep2_speech = ep2_speech[self.long_ventana_espect-1:]
        ep2_speech  =  ep2_speech/max(abs(ep2_speech))
        e2H2_speech = H2*np.abs(ep2_speech)

        ep1_acc  = ep1_acc[self.long_ventana_espect-1:]
        ep1_acc  =  ep1_acc/max(abs(ep1_acc))
        e1H1_acc = H1*np.abs(ep1_acc)
        ep2_acc  = ep2_acc[self.long_ventana_espect-1:]
        ep2_acc  =  ep2_acc/max(abs(ep2_acc))
        e2H2_acc = H2*np.abs(ep2_acc)
        
        parametro_e1H1 = np.sum(H1*np.abs(ep1_speech) + H1*ep1_acc)/np.sum(H1)
        parametro_e2H2 = np.sum(H2*np.abs(ep2_speech)+ H1*ep2_acc)/np.sum(H2)  
        self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:]

        if parametro_e1H1 < parametro_e2H2:
            self.a_W1_speech = a_W1_speech
            self.a_W2_speech = a_W2_speech
            self.polos_W1_speech = polos_W1_speech
            self.polos_W2_speech = polos_W2_speech
            self.polos_W1_speech_originales = polos_W1_originales_speech
            self.polos_W2_speech_originales = polos_W2_originales_speech
            self.ceros_W1_speech = ceros_W1_speech
            self.ceros_W2_speech = ceros_W2_speech
            self.flujo1_speech = flujo_w1_speech
            self.flujo2_speech = flujo_w2_speech
            self.dflujo1_speech = dflujo_w1_speech
            self.dflujo2_speech = dflujo_w2_speech
            self.error_predic1_speech = ep1_speech
            self.error_predic2_speech = ep2_speech
            self.e1H1_speech = e1H1_speech
            self.e2H2_speech = e2H2_speech
            self.a_W1_acc = a_W1_acc
            self.a_W2_acc = a_W2_acc
            self.polos_W1_acc = polos_W1_acc
            self.polos_W2_acc = polos_W2_acc
            self.polos_W1_acc_originales = polos_W1_originales_acc
            self.polos_W2_acc_originales = polos_W2_originales_acc
            self.ceros_W1_acc = ceros_W1_acc
            self.ceros_W2_acc = ceros_W2_acc
            self.flujo1_acc = flujo_w1_acc
            self.flujo2_acc = flujo_w2_acc
            self.dflujo1_acc = dflujo_w1_acc
            self.dflujo2_acc = dflujo_w2_acc
            self.error_predic1_acc = ep1_acc
            self.error_predic2_acc = ep2_acc
            self.e1H1_acc = e1H1_acc
            self.e2H2_acc = e2H2_acc


            self.suma_e1H1 = parametro_e1H1
            self.suma_e2H2 = parametro_e2H2
        else:
            W2 = self.W2_speech
            W1 = self.W1_speech
            self.W1_speech = W2
            self.W2_speech = W1
            self.H1 = H2
            self.H2 = H1
            self.a_W1_speech = a_W2_speech
            self.a_W2_speech = a_W1_speech
            self.polos_W1_speech = polos_W2_speech
            self.polos_W2_speech = polos_W1_speech
            self.polos_W2_speech_originales = polos_W1_originales_speech
            self.polos_W1_speech_originales = polos_W2_originales_speech
            self.ceros_W1_speech = ceros_W2_speech
            self.ceros_W2_speech = ceros_W1_speech
            self.flujo1_speech  = flujo_w2_speech
            self.flujo2_speech  = flujo_w1_speech
            self.dflujo1_speech = dflujo_w2_speech
            self.dflujo2_speech =  dflujo_w1_speech
            self.error_predic1_speech = ep2_speech
            self.error_predic2_speech = ep1_speech
            self.e1H1_speech = e2H2_speech
            self.e2H2_speech = e1H1_speech

            W2 = self.W2_acc
            W1 = self.W1_acc
            self.W1_acc = W2
            self.W2_acc = W1
            self.H1 = H2
            self.H2 = H1
            self.a_W1_acc = a_W2_acc 
            self.a_W2_acc = a_W1_acc
            self.polos_W1_acc = polos_W2_acc
            self.polos_W2_acc = polos_W1_acc
            self.polos_W2_acc_originales = polos_W1_originales_acc
            self.polos_W1_acc_originales = polos_W2_originales_acc
            self.ceros_W1_acc = ceros_W2_acc
            self.ceros_W2_acc = ceros_W1_acc
            self.flujo1_acc = flujo_w2_acc
            self.flujo2_acc = flujo_w1_acc
            self.dflujo1_acc = dflujo_w2_acc
            self.dflujo2_acc = dflujo_w1_acc
            self.error_predic1_acc = ep2_acc
            self.error_predic2_acc = ep1_acc
            self.e1H1_acc = e2H2_acc
            self.e2H2_acc = e1H1_acc

            self.suma_e1H1 = parametro_e2H2
            self.suma_e2H2 = parametro_e1H1
            
        if self.guardar_proceso:
            #self.proceso_W1_speech[i] = self.W1_speech
            #self.proceso_W2_speech[i] = self.W2_speech
            #self.proceso_H1[i] = self.H1
            #self.proceso_H2[i] = self.H2
            self.proceso_a_W1_speech.append(self.a_W1_speech)
            self.proceso_a_W2_speech.append(self.a_W2_speech)
            self.proceso_polos_W1_speech.append(self.polos_W1_speech)
            self.proceso_polos_W1_speech_originales.append(self.polos_W1_speech_originales)
            self.proceso_polos_W2_speech.append(self.polos_W2_speech)
            self.proceso_polos_W2_speech_originales.append(self.polos_W2_speech_originales)
            self.proceso_ceros_W1_speech.append(self.ceros_W1_speech)
            self.proceso_ceros_W2_speech.append(self.ceros_W2_speech)                                
            self.proceso_flujo1_speech.append(self.flujo1_speech)
            self.proceso_flujo2_speech.append(self.flujo2_speech)
            self.proceso_dflujo1_speech.append(self.dflujo1_speech)
            self.proceso_dflujo2_speech.append(self.dflujo2_speech)
            self.proceso_error_predic1_speech.append(self.error_predic1_speech)
            self.proceso_error_predic2_speech.append(self.error_predic2_speech)

            self.proceso_a_W1_acc.append(self.a_W1_acc)
            self.proceso_a_W2_acc.append(self.a_W2_acc)
            self.proceso_polos_W1_acc.append(self.polos_W1_acc)
            self.proceso_polos_W1_acc_originales.append(self.polos_W1_acc_originales)
            self.proceso_polos_W2_acc.append(self.polos_W2_acc)
            self.proceso_polos_W2_acc_originales.append(self.polos_W2_acc_originales)
            self.proceso_ceros_W1_acc.append(self.ceros_W1_acc)
            self.proceso_ceros_W2_acc.append(self.ceros_W2_acc)     
            self.proceso_flujo1_acc.append(self.flujo1_acc)
            self.proceso_flujo2_acc.append(self.flujo2_acc)
            self.proceso_dflujo1_acc.append(self.dflujo1_acc)
            self.proceso_dflujo2_acc.append(self.dflujo2_acc)
            self.proceso_error_predic1_acc.append(self.error_predic1_acc)
            self.proceso_error_predic2_acc.append(self.error_predic2_acc)

    def _sincronizar_por_energia(self):

        self.energia_teager_speech_original,self.tiempos_energia_teager_original = Funciones.energiaTeager(self.speech,self.tiempos)
        self.energia_teager_acc_original,_ = Funciones.energiaTeager(self.acc,self.tiempos)
        
        
        corr = sig.correlate(self.energia_teager_speech_original, self.energia_teager_acc_original, mode='full')
        lags = sig.correlation_lags(len(self.energia_teager_speech_original),len(self.energia_teager_acc_original),mode='full')

        lag = lags[np.argmax(np.abs(corr))]
        #lag = 0
        self.lag = lag
        if lag > 0:
            # speech adelantada
            self.speech = self.speech[lag:]
            self.acc    = self.acc[:len(self.speech)]
            self.tiempos = self.tiempos[lag:]
            self.dEGG = self.dEGG[lag:] 

        elif lag < 0:
            lag = abs(lag)
            self.acc    = self.acc[lag:]
            self.speech = self.speech[:len(self.acc)]
            self.tiempos = self.tiempos[:len(self.speech)]
            self.dEGG = self.dEGG[:len(self.speech)]
        elif lag == 0:
            self.speech = self.speech
            self.acc    = self.acc
            self.tiempos = self.tiempos
            self.dEGG = self.dEGG 

        self.energia_teager_speech_sincronizado,self.tiempos_energia_teager_sincronizado = Funciones.energiaTeager(self.speech,self.tiempos)
        self.energia_teager_acc_sincronizado,_ = Funciones.energiaTeager(self.acc,self.tiempos)


        self.tiempos_espectrograma_sincronizado = self.tiempos[self.long_ventana_espect - 1:]
        self.to_espectrograma_sincronizado = self.tiempos_espectrograma_sincronizado[0]
        self.tf_espectrograma_sincronizado = self.tiempos_espectrograma_sincronizado[-1]

        self.tiempos_eH_speech = self.tiempos[self.long_ventana_espect-1:]
        self.tiempos_eH_acc = self.tiempos[self.long_ventana_espect-1:]

#======================================================  Funciones Principales 


    def NNMF(self):
        X = self.espectrograma_sincronizado
        N, M = X.shape
        error, dif_error_X, dif_error_W1_speech, dif_error_W2_speech,dif_error_W1_acc, dif_error_W2_acc, dif_error_H1, dif_error_H2 = [], [], [], [], [], [],[],[]

        # Inicialización de W y H
        energia_espectrograma  = [np.sum(np.abs(X[:, i]) ** 2) for i in range(X.shape[1])]
        idx_max = np.argmax(energia_espectrograma)
        idx_min = np.argmin(energia_espectrograma)
        W = np.vstack([X[:, idx_max], X[:, idx_min]]).T
        fila1 = np.random.rand(M)
        fila2 = 1-fila1
        H = np.array([fila1,fila2])
        
        #H = np.vstack([np.random.rand(M), 1 - np.random.rand(M)])

        # Aproximación inicial
        X_aprox = W @ H
        error.append(self._calcular_error(X, X_aprox, self.norma))

        # Pre-iteraciones
        if self.n_pre_itereraciones > 0:
            W, H, X_aprox = self._iterar(W, H, X, self.norma, self.n_pre_itereraciones,
                                        error, dif_error_X, dif_error_W1_speech, dif_error_W2_speech,dif_error_W1_acc, dif_error_W2_acc, dif_error_H1, dif_error_H2)

        # Iteraciones principales
        W, H, X_aprox = self._iterar(W, H, X, self.norma, self.n_iteraciones,
                                    error, dif_error_X, dif_error_W1_speech, dif_error_W2_speech,dif_error_W1_acc, dif_error_W2_acc, dif_error_H1, dif_error_H2)

        # Guardo los resultados
        mid = N // 2 
        self.W1_speech = W[:mid,0]
        self.W2_speech = W[:mid,1]
        self.W1_acc = W[mid:,0]
        self.W2_acc = W[mid:,1]
        self.H1 = H[0,:]
        self.H2 = H[1,:]

        self.espectrograma_aprox = X_aprox
        self.espectrograma_aprox_plot = np.vstack((X_aprox[mid:,:], X_aprox[:mid,:]))
        self.error_NNMF = np.array(error)
        self.dif_error_X = np.array(dif_error_X)
        self.dif_error_W1_speech = np.array(dif_error_W1_speech)
        self.dif_error_W2_speech = np.array(dif_error_W2_speech)
        self.dif_error_W1_acc = np.array(dif_error_W1_acc)
        self.dif_error_W2_acc = np.array(dif_error_W2_acc)
        self.dif_error_H1 = np.array(dif_error_H1)
        self.dif_error_H2 = np.array(dif_error_H2)

    def Espectrograma(self):

        # Primero calculos los espectrogramas originales
        matriz_speech, espec_speech = Funciones.calcular_espectrograma(self.speech,
                                                                       fs=self.fs,
                                                                       ventana=self.ventana,
                                                                       duracion_ventana=self.duracion_ventana,
                                                                       desplazamiento=self.desplazamiento,
                                                                       pre_enfasis=self.pre_enfasis_speech,
                                                                       p=self.beta_speech)
        matriz_acc, espec_acc = Funciones.calcular_espectrograma(self.acc,
                                                                       fs=self.fs,
                                                                       ventana=self.ventana,
                                                                       duracion_ventana=self.duracion_ventana,
                                                                       desplazamiento=self.desplazamiento,
                                                                       pre_enfasis=self.pre_enfasis_acc,
                                                                       doble_pre_enfasis=self.doble_pre_enfasis_acc,
                                                                       p=self.beta_acc)

        self.matriz_STFT_speech_original = matriz_speech
        self.matriz_STFT_acc_original   = matriz_acc

        if self.normalizar_espectrogramas:
            espec_speech = espec_speech/espec_speech.max()
            espec_acc    = espec_acc /espec_acc.max()         

        self.espectrograma_original = np.vstack([espec_speech, espec_acc])
        # Por la forma en que matplotlib representa las imagenes como matrices, tengo que invertir el orden de los espectrogramas solo para graficar
        self.espectrograma_plot_original = np.vstack([espec_acc, espec_speech])

        self.espectrograma_speech_original = espec_speech
        self.espectrograma_acc_original = espec_acc       

        self.tiempos_espectrograma_original = self.tiempos[self.long_ventana_espect - 1:]
        self.dEGG_original = self.dEGG[self.long_ventana_espect - 1:]
        self.to_espectrograma_original = self.tiempos[0]+self.duracion_ventana
        self.tf_espectrograma_original = self.tiempos[-1]

        # Ahora sincronizo las señales y calculo los espectrogramas con los que voy a laburar
        self._sincronizar_por_energia()
        matriz_speech, espec_speech = Funciones.calcular_espectrograma(self.speech,
                                                                       fs=self.fs,
                                                                       ventana=self.ventana,
                                                                       duracion_ventana=self.duracion_ventana,
                                                                       desplazamiento=self.desplazamiento,
                                                                       pre_enfasis=self.pre_enfasis_speech,
                                                                       p=self.beta_speech)
        matriz_acc, espec_acc = Funciones.calcular_espectrograma(self.acc,
                                                                       fs=self.fs,
                                                                       ventana=self.ventana,
                                                                       duracion_ventana=self.duracion_ventana,
                                                                       desplazamiento=self.desplazamiento,
                                                                       pre_enfasis=self.pre_enfasis_acc,
                                                                       doble_pre_enfasis=self.doble_pre_enfasis_acc,
                                                                       p=self.beta_acc)

        self.matriz_STFT_speech_sincronizado = matriz_speech
        self.matriz_STFT_acc_sincronizado   = matriz_acc

        if self.normalizar_espectrogramas:
            espec_speech = espec_speech/espec_speech.max()
            espec_acc    = espec_acc /espec_acc.max()         

        self.espectrograma_sincronizado = np.vstack([espec_speech, espec_acc])
        self.espectrograma_plot_sincronizado = np.vstack([espec_acc, espec_speech])

        self.espectrograma_speech_sincronizado = espec_speech
        self.espectrograma_acc_sincronizado = espec_acc       




    def GIF(self):
        if self.pre_enfasis_speech:
            señal_speech = sig.lfilter([1, -1], [1], self.speech)
            self.speech_pre_enfasis = señal_speech
        else:
            señal_speech = self.speech - np.mean(self.speech)
        if self.doble_pre_enfasis_acc:
            señal_acc = sig.lfilter([1, -2, 1], [1], self.acc)
            self.acc_pre_enfasis = señal_acc
        elif self.pre_enfasis_acc:
            señal_acc = sig.lfilter([1, -1], [1], self.acc)
            self.acc_pre_enfasis = señal_acc
        else:
            señal_acc = self.acc - np.mean(self.acc)

        if self.guardar_proceso:
            for i in range(self.n_iteraciones):
                H1, H2 = self.proceso_H1[i], self.proceso_H2[i]

                a_W1_speech,polos_W1_speech,ceros_W1_speech, a_W1_speech_originales,polos_W1_speech_originales = self._resolver_toeplitz(self.proceso_W1_speech[i]**(2/self.beta_speech),self.orden_filtro_tracto_speech)
                a_W2_speech,polos_W2_speech,ceros_W2_speech, a_W2_speech_originales,polos_W2_speech_originales = self._resolver_toeplitz(self.proceso_W2_speech[i]**(2/self.beta_speech),self.orden_filtro_tracto_speech)

                dflujo_w1_speech,flujo_w1_speech = self._filtrado_inverso(a_W1_speech, self.speech,self.alfa_speech)
                dflujo_w2_speech,flujo_w2_speech = self._filtrado_inverso(a_W2_speech, self.speech,self.alfa_speech)

                ep1_speech = self._calcular_error_prediccion(a_W1_speech_originales, señal_speech)
                ep2_speech = self._calcular_error_prediccion(a_W2_speech_originales, señal_speech)

                
                a_W1_acc,polos_W1_acc,ceros_W1_acc,a_W1_acc_originales, polos_W1_acc_originales = self._resolver_toeplitz(self.proceso_W1_acc[i]**(2/self.beta_acc),self.orden_filtro_tracto_acc)
                a_W2_acc,polos_W2_acc,ceros_W2_acc,a_W2_acc_originales, polos_W2_acc_originales = self._resolver_toeplitz(self.proceso_W2_acc[i]**(2/self.beta_acc),self.orden_filtro_tracto_acc)

                dflujo_w1_acc,flujo_w1_acc = self._filtrado_inverso(a_W1_acc, self.acc,self.alfa_acc)
                dflujo_w2_acc,flujo_w2_acc = self._filtrado_inverso(a_W2_acc, self.acc,self.alfa_acc)


                ep1_acc = self._calcular_error_prediccion(a_W1_acc_originales, señal_acc)
                ep2_acc = self._calcular_error_prediccion(a_W2_acc_originales, señal_acc)

                self._seleccionar_candidato(H1,H2,
                                            a_W1_speech,a_W2_speech,
                                            polos_W1_speech,polos_W2_speech,
                                            polos_W1_speech_originales,polos_W2_speech_originales,
                                            ceros_W1_speech,ceros_W2_speech,
                                            flujo_w1_speech,flujo_w2_speech,
                                            dflujo_w1_speech,dflujo_w2_speech,
                                            ep1_speech,ep2_speech,
                                            a_W1_acc,a_W2_acc,
                                            polos_W1_acc,polos_W2_acc,
                                            polos_W1_acc_originales,polos_W2_acc_originales,
                                            ceros_W1_acc,ceros_W2_acc,
                                            flujo_w1_acc,flujo_w2_acc,
                                            dflujo_w1_acc,dflujo_w2_acc,
                                            ep1_acc,ep2_acc,i)

        else:
            H1, H2 = self.H1, self.H2

            a_W1_speech,polos_W1_speech,ceros_W1_speech,a_W1_speech_originales, polos_W1_speech_originales   = self._resolver_toeplitz(self.W1_speech**(2/self.beta_speech),self.orden_filtro_tracto_speech)
            a_W2_speech,polos_W2_speech,ceros_W2_speech,a_W2_speech_originales, polos_W2_speech_originales   = self._resolver_toeplitz(self.W2_speech**(2/self.beta_speech),self.orden_filtro_tracto_speech)


            dflujo_w1_speech,flujo_w1_speech = self._filtrado_inverso(a_W1_speech, self.speech,self.alfa_speech)
            dflujo_w2_speech,flujo_w2_speech = self._filtrado_inverso(a_W2_speech, self.speech,self.alfa_speech)

            ep1_speech = self._calcular_error_prediccion(a_W1_speech_originales, señal_speech)
            ep2_speech = self._calcular_error_prediccion(a_W2_speech_originales, señal_speech)


            a_W1_acc,polos_W1_acc,ceros_W1_acc,a_W1_acc_originales,polos_W1_acc_originales = self._resolver_toeplitz(self.W1_acc**(2/self.beta_acc),self.orden_filtro_tracto_acc)
            a_W2_acc,polos_W2_acc,ceros_W2_acc,a_W2_acc_originales,polos_W2_acc_originales = self._resolver_toeplitz(self.W2_acc**(2/self.beta_acc),self.orden_filtro_tracto_acc)

            dflujo_w1_acc,flujo_w1_acc = self._filtrado_inverso(a_W1_acc, self.acc,self.alfa_acc)
            dflujo_w2_acc,flujo_w2_acc = self._filtrado_inverso(a_W2_acc, self.acc,self.alfa_acc)

            ep1_acc= self._calcular_error_prediccion(a_W1_acc_originales, señal_acc)
            ep2_acc = self._calcular_error_prediccion(a_W2_acc_originales, señal_acc)

            self._seleccionar_candidato(H1,H2,
                                        a_W1_speech,a_W2_speech,
                                        polos_W1_speech,polos_W2_speech,
                                        polos_W1_speech_originales,polos_W2_speech_originales,
                                        ceros_W1_speech,ceros_W2_speech,
                                        flujo_w1_speech,flujo_w2_speech,
                                        dflujo_w1_speech,dflujo_w2_speech,
                                        ep1_speech,ep2_speech,
                                        a_W1_acc,a_W2_acc,
                                        polos_W1_acc,polos_W2_acc,
                                        polos_W1_acc_originales,polos_W2_acc_originales,
                                        ceros_W1_acc,ceros_W2_acc,
                                        flujo_w1_acc,flujo_w2_acc,
                                        dflujo_w1_acc,dflujo_w2_acc,
                                        ep1_acc,ep2_acc)


    def Run(self):
        self.Espectrograma()
        self.NNMF()
        self.GIF()