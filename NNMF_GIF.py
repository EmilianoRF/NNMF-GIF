import numpy as np
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import Funciones as Funciones

class NNMF_GIF:
    def __init__(self, señal, dEGG,tiempos,mascara_tiempos, fs,
                 ventana='Rectangular',
                 duracion=5.5,
                 hop=1,
                 norma='frobenius',
                 pre_iteraciones=0,
                 iteraciones=250,
                 T = 10,
                 delta = 10e-4,
                 margen_error = 1,
                 p_filtro=12,
                 alfa=0.96,
                 beta = 0.55,
                 pre_enfasis = True,
                 doble_pre_enfasis = False,
                 guardar_proceso = True,
                 normalizar_espectrograma = True,
                 nombre_archivo = ''):

        # ============ Parámetros por defecto
        self.señal = señal
        self.señal_pre_enfasis = -1
        self.dEGG = dEGG
        self.fs     = fs
        self.tiempos = tiempos
        self.mascara_tiempos = mascara_tiempos
        self.duracion_ventana = duracion # En ms
        self.long_ventana_espect = int(np.round(fs * duracion / 1000))
        self.desplazamiento   = hop    
        self.ventana = ventana
        self.beta = beta
        self.alfa = alfa
        self.norma         = norma
        self.n_iteraciones = iteraciones
        self.T = T
        self.delta = delta
        self.n_pre_itereraciones = pre_iteraciones
        self.margen_error = margen_error/100
        self.orden_filtro_tracto = p_filtro
        self.pre_enfasis = pre_enfasis
        self.doble_pre_enfasis = doble_pre_enfasis   
        self.guardar_proceso = guardar_proceso
        self.normalizar_espectrograma = normalizar_espectrograma
        self.lag = None
        self.fo_espectrograma = 0
        self.ff_espectrograma = fs / 2
        self.tiempos_gve = self.tiempos[self.mascara_tiempos]
        self.tiempos_espectrograma = self.tiempos[self.long_ventana_espect - 1:]
        self.nombre_archivo = nombre_archivo
        #-----------------------------------------------------------------------------------
        self.speech_sinc = -1
        self.acc_sinc = -1
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
        self.tiempos_eH = -1
        self.fo_espec = -1
        self.ff_espec = -1
        self.espectrograma_aprox = -1
        self.energia_speech = -1

        self.H_i =  -1 
        self.W_i =  -1
        self.W1 = -1
        self.W2 = -1
        self.H1 = -1
        self.H2 = -1

        self.a_W1  = -1
        self.a_W2 = -1
        self.polos_W1 = -1
        self.polos_W1_originales = -1
        self.ceros_W1  = -1
        self.polos_W2 = -1
        self.polos_W2_originales = -1
        self.ceros_W2 = -1

        self.flujo1 = -1
        self.flujo2 = -1
        self.dflujo1  = -1
        self.dflujo2  = -1

        self.error_predic1 = -1
        self.error_predic2 = -1
        self.e1H1  =-1
        self.e2H2 =-1
        self.sum_e1H1 = -1
        self.sum_e2H2 = -1


        self.dif_error_X = -1

        self.dif_error_W1 = -1
        self.dif_error_W2 = -1
        self.dif_error_W1 = -1
        self.dif_error_W2 = -1



        #-----------------------------------------------------------------------------------
        self.proceso_espectrograma = []
        self.proceso_espectrograma_aprox = []
        self.proceso_espectrograma_aprox_plot = []
        self.proceso_dif_error_X = []
        self.proceso_H1 = []
        self.proceso_H2 = []
        self.proceso_dif_error_H1  = []
        self.proceso_dif_error_H2   = []


        self.proceso_W1  = []
        self.proceso_W2 = []

        self.proceso_a_W1 = []
        self.proceso_a_W2 = []
        self.proceso_polos_W1 = []
        self.proceso_polos_W1_originales = []
        self.proceso_ceros_W1 = []
        self.proceso_polos_W2 = []
        self.proceso_polos_W2_originales = []
        self.proceso_ceros_W2 = []

        self.proceso_flujo1 = []
        self.proceso_flujo2 = []
        self.proceso_dflujo1 = []
        self.proceso_dflujo2= []

        self.proceso_error_predic1 = []
        self.proceso_error_predic2 = []
        self.proceso_e1H1 = []
        self.proceso_e2H2 = []
        self.proceso_sum_e1H1 = []
        self.proceso_sum_e2H2 = []

        self.proceso_dif_error_W1 = []
        self.proceso_dif_error_W2 = []

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
                dif_error_X, dif_error_W1, dif_error_W2, dif_error_H1, dif_error_H2):
        X_aprox = W @ H
        for t in range(num_iter):
            W_old, H_old = W.copy(), H.copy()
            W, H = self._actualizar_WH(W, H, X, X_aprox, norma)
            X_aprox = W @ H
            e = self._calcular_error(X, X_aprox, norma)
            error.append(e)


            dif_error_W1.append(
                np.linalg.norm(W_old[:,0] - W[:,0]) / np.linalg.norm(W_old[:,0])
            )
            dif_error_W2.append(
                np.linalg.norm(W_old[:,1] - W[:,1]) / np.linalg.norm(W_old[:,1])
            )
            
            dif_error_H1.append(np.linalg.norm(H_old[0,:]-H[0,:])/np.linalg.norm(H_old[0,:]))
            dif_error_H2.append(np.linalg.norm(H_old[1,:]-H[1,:])/np.linalg.norm(H_old[1,:]))

            if len(error) > 1:
                dif_error_X.append(abs((error[-1]-error[-2])/error[-1]))

            if self.guardar_proceso:
                self.proceso_espectrograma_aprox.append(X_aprox)
                self.proceso_W1.append(W[:,0])
                self.proceso_W2.append(W[:,1])
                self.proceso_H1.append(H[0,:])
                self.proceso_H2.append(H[1,:])
            if t >=self.T:
                if abs(error[t-self.T] - error[t]) <= self.delta*error[t]:
                    self.n_iteraciones = t
                    break

        return W, H, X_aprox

    def _resolver_toeplitz(self, W):
        W_  = np.concatenate((W, np.flip(W)[1:-1]))
        r_W = np.fft.ifft(W_).real
        a_  = solve_toeplitz(
            (r_W[0:self.orden_filtro_tracto], r_W[0:self.orden_filtro_tracto]),
            r_W[1:self.orden_filtro_tracto+1]
        )
        a_W_originales= np.concatenate(([1.0], -a_))
        num      = np.zeros(len(a_W_originales))
        num[0]   = 1
        ceros_W,polos_W,_ = sig.tf2zpk(num,a_W_originales)
        polos_originales = polos_W
        polos_ = [p for p in polos_W if not ( ((p.real > 0.0) and(p.imag == 0.0))  or ((p.real > 0.0) and (np.abs(p) < 0.8)))]


        b,a = sig.zpk2tf(np.zeros(len(polos_)),polos_,k=1)
        a_W = a
        ganancia_dc = np.sum(a_W)   # A(z=1)
        if np.abs(ganancia_dc) > 1e-8:
            a_W = a_W / ganancia_dc
            
        polos_W = polos_
        return a_W, polos_W, ceros_W, a_W_originales, polos_originales

    def _normalizar(self, señal):
        maximo = np.max(np.abs(señal))
        return señal / maximo if maximo > 1 else señal

    def _calcular_flujo(self, señal):
        flujo = sig.lfilter([1], [1, -self.alfa], señal)
        flujo = flujo[self.mascara_tiempos]
        flujo = Funciones.Normalizar(flujo)
        return flujo

    def _filtrado_inverso(self, a_W,señal):
        dflujo = sig.lfilter(a_W, [1], señal)
        flujo = sig.lfilter([1], [1, -self.alfa], dflujo)     
        dflujo = dflujo[self.mascara_tiempos]
        dflujo = dflujo/abs(min(dflujo))
        flujo = flujo[self.mascara_tiempos]
        flujo = Funciones.Normalizar(flujo)
        return dflujo,flujo

    def _calcular_error_prediccion(self, a_W, señal):
        ep = sig.lfilter(b=a_W, a=[1], x=señal)
        #return self._normalizar(ep)
        return ep

    def _seleccionar_candidato(self, H1, H2,
                               a_W1, a_W2,
                               polos_W1,polos_W2,
                               polos_W1_originales,polos_W2_originales,
                               ceros_W1,ceros_W2,
                               flujo_w1, flujo_w2,
                               dflujo_w1, dflujo_w2,
                               ep1, ep2, i=None):

        ep1 = ep1[self.long_ventana_espect-1:]
        ep1 = ep1/max(abs(ep1))
        e1H1 = H1*np.abs(ep1)
        parametro_e1H1 = np.sum(e1H1)/np.sum(H1)

        ep2 = ep2[self.long_ventana_espect-1:]
        ep2 = ep2/max(abs(ep2))
        e2H2 = H1*np.abs(ep2)
        parametro_e2H2 = np.sum(e2H2)/np.sum(H2)
        
        self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:len(self.tiempos)]
        if parametro_e1H1 < parametro_e2H2:
            self.a_W1 = a_W1
            self.a_W2 = a_W2
            self.polos_W1 = polos_W1
            self.polos_W2 = polos_W2
            self.polos_W1_originales = polos_W1_originales
            self.polos_W2_originales = polos_W2_originales
            self.ceros_W1 = ceros_W1
            self.ceros_W2 = ceros_W2
            self.flujo1 = flujo_w1
            self.flujo2 = flujo_w2
            self.dflujo1 = dflujo_w1
            self.dflujo2 = dflujo_w2
            self.error_predic1 = ep1
            self.error_predic2 = ep2
            self.e1H1 = e1H1
            self.e2H2 = e2H2
            self.suma_e1H1 = parametro_e1H1
            self.suma_e2H2 = parametro_e2H2
        else:
            W2 = self.W2
            W1 = self.W1
            self.W1 = W2
            self.W2 = W1
            self.H1 = H2
            self.H2 = H1
            self.a_W1 = a_W2
            self.a_W2 = a_W1
            self.polos_W1 = polos_W2
            self.polos_W2 = polos_W1
            self.polos_W2_originales = polos_W1_originales
            self.polos_W1_originales = polos_W2_originales
            self.ceros_W1 = ceros_W2
            self.ceros_W2 = ceros_W1
            self.flujo1  = flujo_w2
            self.flujo2  = flujo_w1
            self.dflujo1 = dflujo_w2
            self.dflujo2 =  dflujo_w1
            self.error_predic1 = ep2
            self.error_predic2 = ep1
            self.e1H1 = e2H2
            self.e2H2 = e1H1
            self.suma_e1H1 = parametro_e2H2
            self.suma_e2H2 = parametro_e1H1
            
        if self.guardar_proceso:
            #self.proceso_W1[i] = self.W1
            #self.proceso_W2[i] = self.W2
            #self.proceso_H1[i] = self.H1
            #self.proceso_H2[i] = self.H2
            self.proceso_a_W1.append(self.a_W1)
            self.proceso_a_W2.append(self.a_W2)
            self.proceso_polos_W1.append(self.polos_W1)
            self.proceso_polos_W2.append(self.polos_W2)
            self.proceso_polos_W1_originales.append(self.polos_W1_originales)
            self.proceso_polos_W2_originales.append(self.polos_W2_originales)
            self.proceso_ceros_W1.append(self.ceros_W1)
            self.proceso_ceros_W2.append(self.ceros_W2)                                
            self.proceso_flujo1.append(self.flujo1)
            self.proceso_flujo2.append(self.flujo2)
            self.proceso_dflujo1.append(self.dflujo1)
            self.proceso_dflujo2.append(self.dflujo2)
            self.proceso_error_predic1.append(self.error_predic1)
            self.proceso_error_predic2.append(self.error_predic2)
            


#======================================================  Funciones Principales 


    def NNMF(self):
        X = self.espectrograma
        N, M = X.shape
        error, dif_error_X, dif_error_W1, dif_error_W2,dif_error_H1, dif_error_H2 = [], [], [], [], [], []

        idx_max = np.argmax(self.energia)
        idx_min = np.argmin(self.energia)
        W = np.vstack([self.espectrograma[:, idx_max], self.espectrograma[:, idx_min]]).T
        fila1 = np.random.rand(M)
        fila2 = 1-fila1
        H = np.array([fila1,fila2])
        #H = np.vstack([np.random.rand(M), 1 - np.random.rand(M)])

        X_aprox = W @ H
        error.append(self._calcular_error(X, X_aprox, self.norma))

        if self.n_pre_itereraciones > 0:
            W, H, X_aprox = self._iterar(W, H, X, self.norma, self.n_pre_itereraciones,
                                        error, dif_error_X, dif_error_W1, dif_error_W2, dif_error_H1, dif_error_H2)

        W, H, X_aprox = self._iterar(W, H, X, self.norma, self.n_iteraciones,
                                    error, dif_error_X, dif_error_W1, dif_error_W2, dif_error_H1, dif_error_H2)

        self.W1 = W[:,0]
        self.W2 = W[:,1]
        self.H1 = H[0,:]
        self.H2 = H[1,:]

        self.espectrograma_aprox = X_aprox
        self.error_NNMF = np.array(error)
        self.dif_error_X = np.array(dif_error_X)
        self.dif_error_W1 = np.array(dif_error_W1)
        self.dif_error_W2 = np.array(dif_error_W2)
        self.dif_error_H1 = np.array(dif_error_H1)
        self.dif_error_H2 = np.array(dif_error_H2)

    def Espectrograma(self):

        matriz_STFT, espectrograma = Funciones.calcular_espectrograma(self.señal,
                                                                       fs=self.fs,
                                                                       ventana=self.ventana,
                                                                       duracion_ventana=self.duracion_ventana,
                                                                       desplazamiento=self.desplazamiento,
                                                                       pre_enfasis=self.pre_enfasis,
                                                                       doble_pre_enfasis=self.doble_pre_enfasis,
                                                                       p=self.beta)

        if self.normalizar_espectrograma:
            espectrograma = espectrograma/espectrograma.max()

        self.espectrograma = espectrograma

        self.tiempos_espectrograma = self.tiempos[self.long_ventana_espect - 1:]
        self.dEGG_original = self.dEGG[self.long_ventana_espect - 1:]
        self.to_espectrograma = self.tiempos[0]+self.duracion_ventana
        self.tf_espectrograma = self.tiempos[-1]

        '''

        min_ = 0.1
        max_ = 1.0
        arreglo_energia = []
        for i in range(espectrograma.shape[1]):
            E = np.abs(espectrograma[:, i]) ** 2
            E_ = -E
            m = max(E_)
            M = min(E_)
            factor = (max_-min_)/(m-M)
            energia = factor*(E_-M)+min_
            arreglo_energia.append(sum(energia))
        '''
        energia = [sum(np.abs(espectrograma[:, i]) ** 2) for i in range(espectrograma.shape[1])]
        max_energia     = max(energia)
        self.energia = [val / max_energia for val in energia]
        energia_sin_compresion = [sum(np.abs(espectrograma[:, i]) ** (2/self.beta)) for i in range(espectrograma.shape[1])]
        self.energia_sin_compresion = [val / max(energia_sin_compresion) for val in energia_sin_compresion]      
        #self.energia = arreglo_energia
    def GIF(self):
        if self.doble_pre_enfasis:
            señal = sig.lfilter([1, -2, 1], [1], self.señal)
            self.señal_pre_enfasis = señal
        elif self.pre_enfasis:
            señal = sig.lfilter([1, -1], [1], self.señal)
            self.señal_pre_enfasis = señal
        else:
            señal = self.señal - np.mean(self.señal)

        if self.guardar_proceso:
            for i in range(self.n_iteraciones):
                H1, H2 = self.proceso_H1[i], self.proceso_H2[i]

                a_W1,polos_W1,ceros_W1,a_W1_originales,polos_W1_originales = self._resolver_toeplitz(self.proceso_W1[i]**(2/self.beta))
                a_W2,polos_W2,ceros_W2,a_W2_originales,polos_W2_originales = self._resolver_toeplitz(self.proceso_W2[i]**(2/self.beta))

                dflujo_w1,flujo_w1 = self._filtrado_inverso(a_W1,self.señal)
                dflujo_w2,flujo_w2 = self._filtrado_inverso(a_W2, self.señal)

                #flujo_w1 = self._calcular_flujo(dflujo_w1)
                #flujo_w2 = self._calcular_flujo(dflujo_w2)



                ep1 = self._calcular_error_prediccion(a_W1_originales, señal)
                ep2 = self._calcular_error_prediccion(a_W2_originales, señal)

                self._seleccionar_candidato(H1,H2,
                                            a_W1,a_W2,
                                            polos_W1,polos_W2,
                                            polos_W1_originales,polos_W2_originales,
                                            ceros_W1,ceros_W2,
                                            flujo_w1,flujo_w2,
                                            dflujo_w1,dflujo_w2,
                                            ep1,ep2,i)


        else:
            H1, H2 = self.H1, self.H2

            a_W1,polos_W1,ceros_W1,a_W1_originales,polos_W1_originales  = self._resolver_toeplitz(self.W1**(2/self.beta))
            a_W2,polos_W2,ceros_W2,a_W2_originales,polos_W2_originales  = self._resolver_toeplitz(self.W2**(2/self.beta))
            dflujo_w1,flujo_w1 = self._filtrado_inverso(a_W1,self.señal)
            dflujo_w2,flujo_w2 = self._filtrado_inverso(a_W2, self.señal)

            #flujo_w1 = self._calcular_flujo(dflujo_w1)
            #flujo_w2 = self._calcular_flujo(dflujo_w2)
            ep1 = self._calcular_error_prediccion(a_W1_originales, señal)
            ep2 = self._calcular_error_prediccion(a_W2_originales, señal)

            self._seleccionar_candidato(H1,H2,
                                        a_W1,a_W2,
                                        polos_W1,polos_W2,
                                        polos_W1_originales,polos_W2_originales,
                                        ceros_W1,ceros_W2,
                                        flujo_w1,flujo_w2,
                                        dflujo_w1,dflujo_w2,
                                        ep1,ep2)

    def Run(self):
        self.Espectrograma()
        self.NNMF()
        self.GIF()