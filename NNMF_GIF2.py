import numpy as np
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import Funciones as Funciones

class NNMF_GIF:
    def __init__(self, speech, acelerometro, dEGG,tiempos, fs,
                 ventana='boxcar',
                 duracion=5.5,
                 hop=1,
                 norma='frobenius',
                 pre_iteraciones=0,
                 iteraciones=50,
                 margen_error = 1,
                 p_filtro=12,
                 alfa=0.99,
                 beta_speech = 0.55,
                 beta_acc    = 0.55,
                 pre_enfasis_speech = True,
                 pre_enfasis_acc    = True,
                 guardar_proceso = False,
                 normalizar_espectrogramas = False):

        # ============ Parámetros por defecto
        self.speech = speech
        self.acc  = acelerometro
        self.dEGG = dEGG
        self.fs     = fs
        self.tiempos = tiempos
        self.to = -1
        self.tf = -1
        self.duracion_ventana = duracion # En ms
        self.long_ventana_espect = int(np.round(fs * duracion / 1000))
        self.desplazamiento   = hop    
        self.ventana = ventana
        self.beta_speech = beta_speech
        self.beta_acc = beta_acc
        self.alfa = alfa
        self.norma         = norma
        self.n_iteraciones = iteraciones
        self.n_pre_itereraciones = pre_iteraciones
        self.margen_error = margen_error/100
        self.orden_filtro_tracto = p_filtro
        self.tiempos_gve = -1
        self.pre_enfasis_speech = pre_enfasis_speech
        self.pre_enfasis_acc = pre_enfasis_acc       
        self.guardar_proceso = guardar_proceso
        self.normalizar_espectrogramas = normalizar_espectrogramas
        self.lag = None
        #-----------------------------------------------------------------------------------
        self.speech_sinc = -1
        self.acc_sinc = -1
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
        self.tiempos_eH = -1
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
        self.ceros_W1_speech = -1
        self.polos_W2_speech = -1
        self.ceros_W2_speech = -1

        
        self.a_W1_acc = -1
        self.a_W2_acc = -1
        self.polos_W1_acc = -1
        self.ceros_W1_acc = -1
        self.polos_W2_acc = -1
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
        self.dif_error_H1_speech = -1
        self.dif_error_H2_speech = -1

        self.dif_error_W1_acc = -1
        self.dif_error_W2_acc = -1
        self.dif_error_H1_acc = -1
        self.dif_error_H2_acc = -1



        #-----------------------------------------------------------------------------------
        self.proceso_espectrograma = []
        self.proceso_espectrograma_aprox = []
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
        self.proceso_ceros_W1_speech = []
        self.proceso_polos_W2_speech = []
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
        self.proceso_ceros_W1_acc = []
        self.proceso_polos_W2_acc = []
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
        for _ in range(num_iter):
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
                self.proceso_W1_speech.append(W[:mid,0])
                self.proceso_W2_speech.append(W[:mid,1])
                self.proceso_W1_acc.append(W[mid:,0])
                self.proceso_W2_acc.append(W[mid:,1])
                self.proceso_H1.append(H[0,:])
                self.proceso_H2.append(H[1,:])

        return W, H, X_aprox

    def _resolver_toeplitz(self, W):
        W_  = np.concatenate((W, np.flip(W)[1:-1]))
        r_W = np.fft.ifft(W_).real
        a_  = solve_toeplitz(
            (r_W[0:self.orden_filtro_tracto], r_W[0:self.orden_filtro_tracto]),
            r_W[1:self.orden_filtro_tracto+1]
        )
        a_W= np.concatenate(([1.0], -a_))
        num      = np.zeros(len(a_W))
        num[0]   = 1
        ceros_W,polos_W,_ = sig.tf2zpk(num,a_W)
        return a_W,polos_W,ceros_W

    def _normalizar(self, señal):
        maximo = np.max(np.abs(señal))
        return señal / maximo if maximo > 1 else señal

    def _calcular_flujo(self, a_W, señal):
        flujo = sig.lfilter(a_W, [1, -self.alfa], señal)
        return self._normalizar(flujo)

    def _calcular_derivada(self, flujo):
        dflujo = np.gradient(flujo, self.tiempos)
        return self._normalizar(dflujo)

    def _calcular_error_prediccion(self, a_W, señal):
        ep = sig.lfilter(b=a_W, a=[1], x=señal)
        return self._normalizar(ep)

    def _seleccionar_candidato(self, señal, H1, H2,
                               a_W1, a_W2,
                               polos_W1,polos_W2,
                               ceros_W1,ceros_W2,
                               flujo_w1, flujo_w2,
                               dflujo_w1, dflujo_w2,
                               ep1, ep2, i=None):

        parametro_e1H1 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*
                                np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
        parametro_e2H2 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*
                                np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1])  
        
        if señal == 'speech':
            if parametro_e1H1 < parametro_e2H2:
                self.a_W1_speech = a_W1
                self.a_W2_speech = a_W2
                self.polos_W1_speech = polos_W1
                self.polos_W2_speech = polos_W2
                self.ceros_W1_speech = ceros_W1
                self.ceros_W2_speech = ceros_W2
                self.flujo1_speech = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                self.flujo2_speech = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                self.dflujo1_speech = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                self.dflujo2_speech = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                self.error_predic1_speech = ep1[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]
                self.error_predic2_speech = ep2[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
            else:
                W2 = self.W2_speech
                W1 = self.W1_speech
                self.W1 = W1
                self.W2 = W2
                self.H1 = H2
                self.H2 = H1
                self.a_W1_speech = a_W2
                self.a_W2_speech = a_W1
                self.polos_W1_speech = polos_W2
                self.polos_W2_speech = polos_W1
                self.ceros_W1_speech = ceros_W2
                self.ceros_W2_speech = ceros_W1
                self.flujo1_speech  = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                self.flujo2_speech  = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                self.dflujo1_speech = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                self.dflujo2_speech =  dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                self.error_predic1_speech = ep2[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
                self.error_predic2_speech = ep1[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]

            if self.guardar_proceso:
                self.proceso_W1_speech[i] = self.W1_speech
                self.proceso_W2_speech[i] = self.W2_speech
                self.proceso_H1[i] = self.H1
                self.proceso_H2[i] = self.H2
                self.proceso_a_W1_speech.append(self.a_W1_speech)
                self.proceso_a_W2_speech.append(self.a_W2_speech)
                self.proceso_polos_W1_speech.append(self.polos_W1_speech)
                self.proceso_polos_W2_speech.append(self.polos_W2_speech)
                self.proceso_ceros_W1_speech.append(self.ceros_W1_speech)
                self.proceso_ceros_W2_speech.append(self.ceros_W2_speech)                                
                self.proceso_flujo1_speech.append(self.flujo1_speech)
                self.proceso_flujo2_speech.append(self.flujo2_speech)
                self.proceso_dflujo1_speech.append(self.dflujo1_speech)
                self.proceso_dflujo2_speech.append(self.dflujo2_speech)
                self.proceso_error_predic1_speech.append(self.error_predic1_speech)
                self.proceso_error_predic2_speech.append(self.error_predic2_speech)
        else:
            if parametro_e1H1 < parametro_e2H2:
                self.a_W1_acc = a_W1
                self.a_W2_acc = a_W2
                self.polos_W1_acc = polos_W1
                self.polos_W2_acc = polos_W2
                self.ceros_W1_acc = ceros_W1
                self.ceros_W2_acc = ceros_W2
                self.flujo1_acc = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                self.flujo2_acc = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                self.dflujo1_acc = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                self.dflujo2_acc = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                self.error_predic1_acc = ep1[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]
                self.error_predic2_acc = ep2[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
            else:
                W2 = self.W2_acc
                W1 = self.W1_acc
                self.W1 = W2
                self.W2 = W1
                self.H1 = H2
                self.H2 = H1
                self.a_W1_acc = a_W2 
                self.a_W2_acc = a_W1
                self.polos_W1_acc = polos_W2
                self.polos_W2_acc = polos_W1
                self.ceros_W1_acc = ceros_W2
                self.ceros_W2_acc = ceros_W1
                self.flujo1_acc = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                self.flujo2_acc = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                self.dflujo1_acc = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                self.dflujo2_acc = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                self.error_predic1_acc = ep2[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
                self.error_predic2_acc = ep1[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]

            if self.guardar_proceso:
                self.proceso_W1_acc[i] = self.W1_acc
                self.proceso_W2_acc[i] = self.W2_acc
                self.proceso_H1[i] = self.H1
                self.proceso_H2[i] = self.H2
                self.proceso_a_W1_acc.append(self.a_W1_acc)
                self.proceso_a_W2_acc.append(self.a_W2_acc)
                self.proceso_polos_W1_acc.append(self.polos_W1_acc)
                self.proceso_polos_W2_acc.append(self.polos_W2_acc)
                self.proceso_ceros_W1_acc.append(self.ceros_W1_acc)
                self.proceso_ceros_W2_acc.append(self.ceros_W2_acc)     
                self.proceso_flujo1_acc.append(self.flujo1_acc)
                self.proceso_flujo2_acc.append(self.flujo2_acc)
                self.proceso_dflujo1_acc.append(self.dflujo1_acc)
                self.proceso_dflujo2_acc.append(self.dflujo2_acc)
                self.proceso_error_predic1_acc.append(self.error_predic1_acc)
                self.proceso_error_predic2_acc.append(self.error_predic2_acc)

    def Sincronizar_por_energia(self):
        matriz_speech, _ = Funciones.calcular_espectrograma(
            self.speech,
            fs=self.fs,
            ventana=self.ventana,
            duracion_ventana=self.duracion_ventana,
            desplazamiento=self.desplazamiento,
            pre_enfasis=self.pre_enfasis_speech,
            p=self.beta_speech
        )
        matriz_acc, _ = Funciones.calcular_espectrograma(
            self.acc,
            fs=self.fs,
            ventana=self.ventana,
            duracion_ventana=self.duracion_ventana,
            desplazamiento=self.desplazamiento,
            pre_enfasis=self.pre_enfasis_acc,
            p=self.beta_acc
        )

        energia_speech = np.sum(np.abs(matriz_speech)**2, axis=0)
        energia_acc    = np.sum(np.abs(matriz_acc)**2, axis=0)

        energia_speech = energia_speech / np.max(energia_speech)
        energia_acc    = energia_acc / np.max(energia_acc)

        energia_speech = energia_speech -  np.mean(energia_speech)
        energia_acc    = energia_acc - np.mean(energia_acc)

        corr = sig.correlate(energia_speech, energia_acc, mode='full')
        lags = sig.correlation_lags(len(energia_speech),len(energia_acc),mode='full')

        lag = lags[np.argmax(np.abs(corr))]
        self.lag = lag
        if lag > 0:
            # speech adelantada
            self.speech_ = self.speech[lag:]
            self.acc    = self.acc[:len(self.speech)]
            self.tiempos = self.tiempos[lag:]
            self.dEGG = self.dEGG[lag:] 

        elif lag < 0:
            lag = abs(lag)
            self.acc    = self.acc[lag:]
            self.speech = self.speech[:len(self.acc)]
            self.tiempos = self.tiempos[:len(self.speech)]
            self.dEGG = self.dEGG[:len(self.speech)]

        self.to = self.tiempos[0]
        self.tf = self.tiempos[-1]
        self.tiempos_espec = self.tiempos[self.long_ventana_espect - 1:]
        self.dEGG = self.dEGG[self.long_ventana_espect - 1:]
        self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:len(self.tiempos)-self.orden_filtro_tracto-1]
        self.tiempos_gve = self.tiempos[self.orden_filtro_tracto-1:len(self.tiempos)-self.orden_filtro_tracto-1]

#======================================================  Funciones Principales 


    def NNMF(self):
        X = self.espectrograma
        N, M = X.shape
        error, dif_error_X, dif_error_W1_speech, dif_error_W2_speech,dif_error_W1_acc, dif_error_W2_acc, dif_error_H1, dif_error_H2 = [], [], [], [], [], [],[],[]

        # Inicialización de W y H
        idx_max = np.argmax(self.energia_speech)
        idx_min = np.argmin(self.energia_speech)
        W = np.vstack([self.espectrograma[:, idx_max], self.espectrograma[:, idx_min]]).T
        H = np.vstack([np.random.rand(M), 1 - np.random.rand(M)])

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

        # Guardar resultados finales
        mid = N // 2  # punto de corte
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
        self.Sincronizar_por_energia()
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
                                                                       p=self.beta_acc)

        self.matriz_STFT_speech = matriz_speech
        self.matriz_STFT_acc   = matriz_acc

        if self.normalizar_espectrogramas:
            espec_speech = espec_speech/espec_speech.max()
            espec_acc    = espec_acc /espec_acc.max()         

        self.espectrograma = np.vstack([espec_speech, espec_acc])
        self.espectrograma_plot = np.vstack([espec_acc, espec_speech])

        self.espectrograma_speech = espec_speech
        self.espectrograma_acc = espec_acc       


        self.fo_espec = 0
        self.ff_espec = self.fs / 2
        self.to_espec = self.to+self.duracion_ventana
        self.tf_espec = self.tf

        energia_speech = [np.sum(np.abs(matriz_speech[:, i]) ** 2) for i in range(matriz_speech.shape[1])]
        energia_acc    = [np.sum(np.abs(matriz_acc[:, i]) ** 2) for i in range(matriz_acc.shape[1])]
        max_speech     = max(energia_speech)
        max_acc        = max(energia_acc)
        self.energia_speech = [val / max_speech for val in energia_speech]
        self.energia_acc = [val / max_acc for val in energia_acc]

    def GIF(self):
        if self.pre_enfasis_speech:
            señal_speech = sig.lfilter([1, -1], [1], self.speech)      
        else:
            señal_speech = self.speech - np.mean(self.speech)
        if self.pre_enfasis_acc:
            señal_acc    = sig.lfilter([1, -1], [1], self.acc)
        else:
            señal_acc    = self.acc - np.mean(self.acc)

        if self.guardar_proceso:
            for i in range(self.n_iteraciones):
                H1, H2 = self.proceso_H1[i], self.proceso_H2[i]

                a_W1_speech,polos_W1_speech,ceros_W1_speech = self._resolver_toeplitz(self.proceso_W1_speech[i]**(2/self.beta))
                a_W2_speech,polos_W2_speech,ceros_W2_speech = self._resolver_toeplitz(self.proceso_W2_speech[i]**(2/self.beta))

                flujo_w1_speech = self._calcular_flujo(a_W1_speech, self.speech)
                flujo_w2_speech = self._calcular_flujo(a_W2_speech, self.speech)

                dflujo_w1_speech = self._calcular_derivada(flujo_w1_speech)
                dflujo_w2_speech = self._calcular_derivada(flujo_w2_speech)

                ep1_speech = self._calcular_error_prediccion(a_W1_speech, señal_speech)
                ep2_speech = self._calcular_error_prediccion(a_W2_speech, señal_speech)

                self._seleccionar_candidato('speech',H1,H2,
                                            a_W1_speech,a_W2_speech,
                                            polos_W1_speech,polos_W2_speech,
                                            ceros_W1_speech,ceros_W2_speech,
                                            flujo_w1_speech,flujo_w2_speech,
                                            dflujo_w1_speech,dflujo_w2_speech,
                                            ep1_speech,ep2_speech,i)
                
                a_W1_acc,polos_W1_acc,ceros_W1_acc = self._resolver_toeplitz(self.proceso_W1_acc[i]**(2/self.beta))
                a_W2_acc,polos_W2_acc,ceros_W2_acc = self._resolver_toeplitz(self.proceso_W2_acc[i]**(2/self.beta))

                flujo_w1_acc = self._calcular_flujo(a_W1_acc, self.acc)
                flujo_w2_acc = self._calcular_flujo(a_W2_acc, self.acc)

                dflujo_w1_acc = self._calcular_derivada(flujo_w1_acc)
                dflujo_w2_acc = self._calcular_derivada(flujo_w2_acc)

                ep1_acc = self._calcular_error_prediccion(a_W1_acc, señal_acc)
                ep2_acc = self._calcular_error_prediccion(a_W2_acc, señal_acc)

                self._seleccionar_candidato('acc',H1,H2,
                                            a_W1_acc,a_W2_acc,
                                            polos_W1_acc,polos_W2_acc,
                                            ceros_W1_acc,ceros_W2_acc,
                                            flujo_w1_acc,flujo_w2_acc,
                                            dflujo_w1_acc,dflujo_w2_acc,
                                            ep1_acc,ep2_acc,i)

        else:
            H1, H2 = self.H1, self.H2

            a_W1_speech,polos_W1_speech,ceros_W1_speech  = self._resolver_toeplitz(self.W1_speech**(2/self.beta_speech))
            a_W2_speech,polos_W2_speech,ceros_W2_speech  = self._resolver_toeplitz(self.W2_speech**(2/self.beta_speech))

            flujo_w1_speech = self._calcular_flujo(a_W1_speech, self.speech)
            flujo_w2_speech = self._calcular_flujo(a_W2_speech, self.speech)

            dflujo_w1_speech = self._calcular_derivada(flujo_w1_speech)
            dflujo_w2_speech = self._calcular_derivada(flujo_w2_speech)

            ep1_speech = self._calcular_error_prediccion(a_W1_speech, señal_speech)
            ep2_speech = self._calcular_error_prediccion(a_W2_speech, señal_speech)

            self._seleccionar_candidato('speech',H1,H2,
                                        a_W1_speech,a_W2_speech,
                                        polos_W1_speech,polos_W2_speech,
                                        ceros_W1_speech,ceros_W2_speech,
                                        flujo_w1_speech,flujo_w2_speech,
                                        dflujo_w1_speech,dflujo_w2_speech,
                                        ep1_speech,ep2_speech)
            

            a_W1_acc,polos_W1_acc,ceros_W1_acc = self._resolver_toeplitz(self.W1_acc**(2/self.beta_acc))
            a_W2_acc,polos_W2_acc,ceros_W2_acc = self._resolver_toeplitz(self.W2_acc**(2/self.beta_acc))

            flujo_w1_acc = self._calcular_flujo(a_W1_acc, self.acc)
            flujo_w2_acc = self._calcular_flujo(a_W2_acc, self.acc)

            dflujo_w1_acc= self._calcular_derivada(flujo_w1_acc)
            dflujo_w2_acc = self._calcular_derivada(flujo_w2_acc)

            ep1_acc= self._calcular_error_prediccion(a_W1_acc, señal_acc)
            ep2_acc = self._calcular_error_prediccion(a_W2_acc, señal_acc)

            self._seleccionar_candidato('acc',H1,H2,
                                        a_W1_acc,a_W2_acc,
                                        polos_W1_acc,polos_W2_acc,
                                        ceros_W1_acc,ceros_W2_acc,
                                        flujo_w1_acc,flujo_w2_acc,
                                        dflujo_w1_acc,dflujo_w2_acc,
                                        ep1_acc,ep2_acc)


    def Run(self):
        self.Espectrograma()
        self.NNMF()
        self.GIF()