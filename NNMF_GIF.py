import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt


class NNMF_GIF:
    def __init__(self, señal,tiempos, fs,
                 ventana='boxcar',
                 duracion=5.5,
                 hop=1,
                 norma='frobenius',
                 pre_iteraciones=0,
                 iteraciones=50,
                 margen_error = 1,
                 p_filtro=12,
                 alfa=0.99,
                 p = 0.55,
                 pre_enfasis =True,
                 guardar_proceso = False):

        # ============ Parámetros por defecto
        self.speech = señal
        self.fs     = fs
        self.tiempos = tiempos
        self.to = tiempos[0]
        self.tf = tiempos[-1]
        self.duracion_ventana = duracion # En ms
        self.desplazamiento   = hop    
        self.ventana = ventana
        self.p = p
        self.alfa = alfa
        self.norma         = norma
        self.n_iteraciones = iteraciones
        self.n_pre_itereraciones = pre_iteraciones
        self.margen_error = margen_error/100
        self.orden_filtro_tracto = p_filtro
        self.tiempos_gve = tiempos[p_filtro-1:len(tiempos)-p_filtro-1]
        self.pre_enfasis = pre_enfasis
        self.guardar_proceso = guardar_proceso
        #-----------------------------------------------------------------------------------

        self.pre_itereraciones = -1
        self.itereraciones = -1

        self.error_NNMF = -1
        self.matriz_STFT = -1
        self.espectrograma = -1
        self.to_espec = -1
        self.tf_espec = -1
        self.long_ventana_espect = -1
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

        self.a_W1 = -1
        self.a_W2 = -1
        self.polos_W1 = -1
        self.ceros_W1 = -1
        self.polos_W2 = -1
        self.ceros_W2 = -1

        self.flujo1 = -1
        self.flujo2 = -1
        self.dflujo1 = -1
        self.dflujo2 = -1

        self.error_predic1 = -1
        self.error_predic2 = -1
        self.e1H1 =-1
        self.e2H2 =-1
        self.sum_e1H1 = -1
        self.sum_e2H2 = -1
        self.dif_error_X = -1
        self.dif_error_W1 = -1
        self.dif_error_W2 = -1
        self.dif_error_H1 = -1
        self.dif_error_H2 = -1
        #-----------------------------------------------------------------------------------
        self.proceso_espectrograma = []
        self.proceso_espectrograma_aprox = []

        self.proceso_W1 = []
        self.proceso_W2 = []
        self.proceso_H1 = []
        self.proceso_H2 = []


        self.proceso_a_W1 = []
        self.proceso_a_W2 = []
        self.proceso_polos_W1 = []
        self.proceso_ceros_W1 = []
        self.proceso_polos_W2 = []
        self.proceso_ceros_W2 = []

        self.proceso_flujo1 = []
        self.proceso_flujo2 = []
        self.proceso_dflujo1 = []
        self.proceso_dflujo2 = []

        self.proceso_error_predic1 = []
        self.proceso_error_predic2 = []
        self.proceso_e1H1 = []
        self.proceso_e2H2 = []
        self.proceso_sum_e1H1 = []
        self.proceso_sum_e2H2 = []
        self.proceso_dif_error_X = []
        self.proceso_dif_error_W1 = []
        self.proceso_dif_error_W2 = []
        self.proceso_dif_error_H1 = []
        self.proceso_dif_error_H2 = []
        # ====================================
        self.Run()

    def NNMF(self):
        X = self.espectrograma
        N = X.shape[0]
        M = X.shape[1]
        R = 2
        i = 0
        debajo_error = False
        epsilon = np.finfo(np.float32).eps
        error= []
        dif_error_X = []
        dif_error_W1 = []
        dif_error_W2 = []
        dif_error_H1 = []
        dif_error_H2 = []

        pre_iteraciones = [i]
        iteraciones = []
        error_ = 0

         #============================= Defino W y H   
        indice_maximo = np.where(self.energia_speech == max(self.energia_speech))[0][0]
        indice_minimo = np.where(self.energia_speech == min(self.energia_speech))[0][0]
        W_i = np.array([np.array(self.espectrograma[:,indice_maximo]),np.array(self.espectrograma[:,indice_minimo])]).transpose()

        fila1 = np.random.rand(M)
        fila2 = 1-fila1
        H_i = np.array([fila1,fila2])


        # Calculo la aproximación inicial y después calculo el error
        X_aprox = W_i @ H_i

        if self.guardar_proceso:
            self.proceso_espectrograma_aprox.append(X_aprox)
            self.proceso_W1.append(W_i[:,0])
            self.proceso_W2.append(W_i[:,1])
            self.proceso_H1.append(H_i[0,:])
            self.proceso_H2.append(H_i[1,:])

            if self.norma == 'frobenius':
                error_ = np.linalg.norm(X-X_aprox, ord='fro')
                error.append(error_)
                if self.n_pre_itereraciones >0:
                #============================= Pre-iteraciones
                    while i <= self.n_pre_itereraciones:
                        # Guardo los valores de H1 y H2 previos
                        H_1_0 = H_i[0,:]
                        H_2_0 = H_i[1,:]
                        W_1_0 = W_i[:,0]
                        W_2_0 = W_i[:,1]
                        W_1_1 = W_i[:,0]
                        W_2_1 = W_i[:,1] 

                        H_i = H_i * (W_i.T @ X) / ((W_i.T @ W_i)@ H_i) 

                        suma_col = H_i.sum(axis=0)
                        H_i = H_i / suma_col
                        # Guardo los valores de H1 y H2 actuales
                        H_1_1 = H_i[0,:]
                        H_2_1 = H_i[1,:]            
                        # Calculo el error para cada H
                        dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                        dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                        dif_error_W1.append(np.linalg.norm(W_1_0-W_1_1)/np.linalg.norm(W_1_0))
                        dif_error_W2.append(np.linalg.norm(W_2_0-W_2_1)/np.linalg.norm(W_2_0))
                        # Calculo la nueva matriz X y su error
                        X_aprox =  W_i @ H_i
                        error_ = np.linalg.norm(X-X_aprox, ord='fro')
                        error.append(error_)
                        self.proceso_espectrograma_aprox.append(X_aprox)
                        self.proceso_W1.append(W_i[:,0])
                        self.proceso_W2.append(W_i[:,1])
                        self.proceso_H1.append(H_i[0,:])
                        self.proceso_H2.append(H_i[1,:])
                        i+=1
                #============================= Resto de las iteraciones
                while not debajo_error and i <= self.n_iteraciones:
                    # Guardo los valores de W1,W2, H1 yH2 previos
                    W_1_0 = W_i[:,0]
                    W_2_0 = W_i[:,1]
                    H_1_0 = H_i[0,:]
                    H_2_0 = H_i[1,:]

                    W_i = W_i * (X @ H_i.T)/(W_i @ (H_i @ H_i.T))
                    H_i = H_i * (W_i.T @ X) / ((W_i.T @ W_i)@ H_i)
                    suma_col = H_i.sum(axis=0)
                    H_i = H_i / suma_col                

                    # Guardo los valores de W1,W2, H1 yH2 actuales
                    W_1_1 = W_i[:,0]
                    W_2_1 = W_i[:,1]
                    H_1_1 = H_i[0,:]
                    H_2_1 = H_i[1,:]
                    # Calculo los errores
                    dif_error_W1.append(np.linalg.norm(W_1_0-W_1_1)/np.linalg.norm(W_1_0))
                    dif_error_W2.append(np.linalg.norm(W_2_0-W_2_1)/np.linalg.norm(W_2_0))
                    dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                    dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                    #dif_error_H1.append(np.sum(((H_1_1-H_1_0)**2)/H_1_0**2))
                    #dif_error_H2.append(np.sum(((H_2_1-H_2_0)**2)/H_2_0**2))               
                    # Calculo la nueva matriz X y su error
                    X_aprox =  W_i @ H_i
                    error_ = np.linalg.norm(X-X_aprox, ord='fro')
                    error.append(error_)
                    iteraciones.append(i)
                    dif_error_X.append(abs((error[len(error)-1]-error[len(error)-2])/(error[len(error)-1])))
                    self.proceso_espectrograma_aprox.append(X_aprox)
                    self.proceso_W1.append(W_i[:,0])
                    self.proceso_W2.append(W_i[:,1])
                    self.proceso_H1.append(H_i[0,:])
                    self.proceso_H2.append(H_i[1,:])
                    i += 1
            if self.norma == 'itakura-saito':

                error_ = np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))
                error.append(error_)
                if self.n_pre_itereraciones >0:
                    #============================= Pre-iteraciones
                    while i <= self.n_pre_itereraciones:
                        # Guardo los valores de H1 y H2 previos
                        H_1_0 = H_i[0,:]
                        H_2_0 = H_i[1,:]
                        W_1_0 = W_i[:,0]
                        W_2_0 = W_i[:,1]

                        H_i = H_i * (W_i.T @ (X*X_aprox**(-2))) / (W_i.T @ X_aprox**(-1))
                        H_sum = H_i.sum(axis=0)
                        H_i = H_i / H_sum
                        # Guardo los valores de H1 y H2 actuales
                        H_1_1 = H_i[0,:]
                        H_2_1 = H_i[1,:]
                        W_1_1 = W_i[:,0]
                        W_2_1 = W_i[:,1]
                        # Calculo el error para cada H                
                        dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                        dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                        dif_error_W1.append(np.linalg.norm(W_1_0-W_1_1)/np.linalg.norm(W_1_0))
                        dif_error_W2.append(np.linalg.norm(W_2_0-W_2_1)/np.linalg.norm(W_2_0))
                        # Calculo la nueva matriz X y su error
                        X_aprox =  W_i @ H_i
                        error_ = np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))
                        error.append(error_)
                        self.proceso_espectrograma_aprox.append(X_aprox)
                        self.proceso_W1.append(W_i[:,0])
                        self.proceso_W2.append(W_i[:,1])
                        self.proceso_H1.append(H_i[0,:])
                        self.proceso_H2.append(H_i[1,:])
                        i+=1


                #============================= Resto de las iteraciones
                while not debajo_error and i <= self.n_iteraciones:
                    # Guardo los valores de W1,W2, H1 yH2 previos
                    W_1_0 = W_i[:,0]
                    W_2_0 = W_i[:,1]
                    H_1_0 = H_i[0,:]
                    H_2_0 = H_i[1,:]

                    W_i = W_i * ( (X_aprox**(-2)*X) @ H_i.T) / (X_aprox**(-1) @ H_i.T)
                    H_i = H_i * (W_i.T @ (X*X_aprox**(-2))) / (W_i.T @ X_aprox**(-1))
                    H_sum = H_i.sum(axis=0)
                    H_i = H_i / H_sum

                    # Guardo los valores de W1,W2, H1 yH2 actuales
                    W_1_1 = W_i[:,0]
                    W_2_1 = W_i[:,1]
                    H_1_1 = H_i[0,:]
                    H_2_1 = H_i[1,:] 
                    # Calculo los errores
                    dif_error_W1.append(np.linalg.norm(W_1_0-W_1_1)/np.linalg.norm(W_1_0))
                    dif_error_W2.append(np.linalg.norm(W_2_0-W_2_1)/np.linalg.norm(W_2_0))
                    dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                    dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                    # Calculo la nueva matriz X y su error
                    X_aprox =  W_i @ H_i
                    error_ = np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))
                    error.append(error_)
                    #iteraciones.append(i)
                    dif_error_X.append(abs((error[len(error)-1]-error[len(error)-2])/(error[len(error)-1])))
                    self.proceso_espectrograma_aprox.append(X_aprox)
                    self.proceso_W1.append(W_i[:,0])
                    self.proceso_W2.append(W_i[:,1])
                    self.proceso_H1.append(H_i[0,:])
                    self.proceso_H2.append(H_i[1,:])
                    i += 1

        else:
            if self.norma == 'frobenius':
                error_ = np.linalg.norm(X-X_aprox, ord='fro')
                error.append(error_)
                if self.n_pre_itereraciones >0:
                #============================= Pre-iteraciones
                    while i <= self.n_pre_itereraciones:
                        # Guardo los valores de H1 y H2 previos
                        H_1_0 = H_i[0,:]
                        H_2_0 = H_i[1,:]
                        
                        H_i = H_i * (W_i.T @ X) / ((W_i.T @ W_i)@ H_i) 

                        suma_col = H_i.sum(axis=0)
                        H_i = H_i / suma_col
                        # Guardo los valores de H1 y H2 actuales
                        H_1_1 = H_i[0,:]
                        H_2_1 = H_i[1,:]            
                        # Calculo el error para cada H
                        dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                        dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                        #dif_error_H1.append(np.sum(((H_1_1-H_1_0)**2)/H_1_0**2))
                        #dif_error_H2.append(np.sum(((H_2_1-H_2_0)**2)/H_2_0**2))
                        # Calculo la nueva matriz X y su error
                        X_aprox =  W_i @ H_i
                        error_ = np.linalg.norm(X-X_aprox, ord='fro')
                        error.append(error_)
                        i+=1
                        pre_iteraciones.append(i)
                #============================= Resto de las iteraciones
                while not debajo_error and i <= self.n_iteraciones:
                    # Guardo los valores de W1,W2, H1 yH2 previos
                    W_1_0 = W_i[:,0]
                    W_2_0 = W_i[:,1]
                    H_1_0 = H_i[0,:]
                    H_2_0 = H_i[1,:]

                    W_i = W_i * (X @ H_i.T)/(W_i @ (H_i @ H_i.T))
                    H_i = H_i * (W_i.T @ X) / ((W_i.T @ W_i)@ H_i)
                    suma_col = H_i.sum(axis=0)
                    H_i = H_i / suma_col                

                    # Guardo los valores de W1,W2, H1 yH2 actuales
                    W_1_1 = W_i[:,0]
                    W_2_1 = W_i[:,1]
                    H_1_1 = H_i[0,:]
                    H_2_1 = H_i[1,:]
                    # Calculo los errores
                    dif_error_W1.append(np.linalg.norm(W_1_0-W_1_1)/np.linalg.norm(W_1_0))
                    dif_error_W2.append(np.linalg.norm(W_2_0-W_2_1)/np.linalg.norm(W_2_0))
                    dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                    dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                    #dif_error_H1.append(np.sum(((H_1_1-H_1_0)**2)/H_1_0**2))
                    #dif_error_H2.append(np.sum(((H_2_1-H_2_0)**2)/H_2_0**2))               
                    # Calculo la nueva matriz X y su error
                    X_aprox =  W_i @ H_i
                    error_ = np.linalg.norm(X-X_aprox, ord='fro')
                    error.append(error_)
                    i += 1
                    iteraciones.append(i)
                    dif_error_X.append(abs((error[len(error)-1]-error[len(error)-2])/(error[len(error)-1])))


            if self.norma == 'itakura-saito':

                error_ = np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))
                error.append(error_)
                if self.n_pre_itereraciones >0:
                    #============================= Pre-iteraciones
                    while i <= self.n_pre_itereraciones:
                        # Guardo los valores de H1 y H2 previos
                        H_1_0 = H_i[0,:]
                        H_2_0 = H_i[1,:]

                        H_i = H_i * (W_i.T @ (X*X_aprox**(-2))) / (W_i.T @ X_aprox**(-1))
                        H_sum = H_i.sum(axis=0)
                        H_i = H_i / H_sum
                        # Guardo los valores de H1 y H2 actuales
                        H_1_1 = H_i[0,:]
                        H_2_1 = H_i[1,:]
                        # Calculo el error para cada H                
                        dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                        dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                        #dif_error_H1.append(np.sum(((H_1_1-H_1_0)**2)/H_1_0**2))
                        #dif_error_H2.append(np.sum(((H_2_1-H_2_0)**2)/H_2_0**2))
                        # Calculo la nueva matriz X y su error
                        X_aprox =  W_i @ H_i
                        error_ = np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))
                        error.append(error_)
                        i+=1
                        pre_iteraciones.append(i)


                #============================= Resto de las iteraciones
                while not debajo_error and i <= self.n_iteraciones:
                    # Guardo los valores de W1,W2, H1 yH2 previos
                    W_1_0 = W_i[:,0]
                    W_2_0 = W_i[:,1]
                    H_1_0 = H_i[0,:]
                    H_2_0 = H_i[1,:]

                    W_i = W_i * ( (X_aprox**(-2)*X) @ H_i.T) / (X_aprox**(-1) @ H_i.T)
                    H_i = H_i * (W_i.T @ (X*X_aprox**(-2))) / (W_i.T @ X_aprox**(-1))
                    H_sum = H_i.sum(axis=0)
                    H_i = H_i / H_sum

                    # Guardo los valores de W1,W2, H1 yH2 actuales
                    W_1_1 = W_i[:,0]
                    W_2_1 = W_i[:,1]
                    H_1_1 = H_i[0,:]
                    H_2_1 = H_i[1,:] 
                    # Calculo los errores
                    dif_error_W1.append(np.linalg.norm(W_1_0-W_1_1)/np.linalg.norm(W_1_0))
                    dif_error_W2.append(np.linalg.norm(W_2_0-W_2_1)/np.linalg.norm(W_2_0))
                    dif_error_H1.append(np.linalg.norm(H_1_0-H_1_1)/np.linalg.norm(H_1_0))
                    dif_error_H2.append(np.linalg.norm(H_2_0-H_2_1)/np.linalg.norm(H_2_0))
                    #dif_error_H1.append(np.sum(((H_1_1-H_1_0)**2)/H_1_0**2))
                    #dif_error_H2.append(np.sum(((H_2_1-H_2_0)**2)/H_2_0**2))
                    # Calculo la nueva matriz X y su error
                    X_aprox =  W_i @ H_i
                    error_ = np.sum(X/X_aprox) - np.prod(X.shape) - np.sum(np.log(X/X_aprox))
                    error.append(error_)
                    i += 1
                    iteraciones.append(i)
                    dif_error_X.append(abs((error[len(error)-1]-error[len(error)-2])/(error[len(error)-1])))

        self.pre_itereraciones = np.array(pre_iteraciones)
        self.iteraciones = np.array(iteraciones)
        self.dif_error_X = np.array(dif_error_X)
        self.dif_error_W1= np.array(dif_error_W1)
        self.dif_error_W2 = np.array(dif_error_W2)
        self.espectrograma_aprox = X_aprox
        self.W_i = W_i
        self.H_i = H_i
        self.error_NNMF = np.array(error)
        self.dif_error_H1 = np.array(dif_error_H1)
        self.dif_error_H2 = np.array(dif_error_H2)

    def Espectrograma(self):
        if self.pre_enfasis:
            b = [1,-1]
            a = [1]
            señal = sig.lfilter(b,a,self.speech)
        else:
            señal = self.speech - np.mean(self.speech)
        longitud_ventana = int(np.round(self.fs*self.duracion_ventana/1000))
        self.long_ventana_espect = longitud_ventana
        if self.ventana == 'boxcar':
            ventana  = sig.windows.boxcar(longitud_ventana)
        '''
        Si 
            Ns = cantidad de elementos de la señal.
            Nv = cantidad de elementos de la ventana.
            d  = cantidad de elementos que se desplaza la ventana.

        La cantidad de ventanas que voy a tener va a ser igual a 
            Cv = 1 + (NS-Nv)/d

        '''
        cantidad_ventanas = int( 1 + np.floor(( len(señal) - longitud_ventana) / self.desplazamiento))

        matriz_STFT     = np.zeros([longitud_ventana,cantidad_ventanas],dtype=complex)

        for i in range(cantidad_ventanas):
            # Esto me da la posición inicial en la señal a partir de la cual voy a hacer el ventaneo
            inicio   = i*self.desplazamiento
            # Esto me da una lista de la forma [0, 1, 2, ...,longitud_ventana - 1 ]
            desp_rel = np.arange(longitud_ventana)
            '''
            La suma me da un resultado de la forma 

            [inicio + 0,inicio + 1,inicio + 2, ..., inicio + longitud_ventana - 1]

            '''
            posicion_ventana   = inicio + desp_rel 

            señal_ventaneada = np.multiply(ventana, señal[posicion_ventana])
            matriz_STFT[:,i] = np.fft.fft(señal_ventaneada)
        
        self.matriz_STFT   = matriz_STFT[0:matriz_STFT.shape[0]//2, :]
        self.espectrograma =np.abs(self.matriz_STFT)**self.p

        self.to_espec = self.to+self.duracion_ventana
        self.tf_espec = self.tf
        self.tiempos_espec = self.tiempos[self.long_ventana_espect-1:]
        self.fo_espec = 0
        self.ff_espec = self.fs/2


        energia = []
        for i in range(0,self.matriz_STFT.shape[1]):
            energia.append(np.sum(np.abs(self.matriz_STFT[:,i])**2))
        maximo = max(energia)
        energia = [valor/maximo for valor in energia]
        self.energia_speech = energia
    
    def GIF(self):
        if self.guardar_proceso:
            for i in range(0,self.n_iteraciones+1):
                H1 = self.proceso_H1[i]
                H2 = self.proceso_H2[i]
                W1 = self.proceso_W1[i]
                W2 = self.proceso_W2[i]
                W1 = W1**(2/self.p)
                W2 = W2**(2/self.p)
                #==================================== Cálculo de los flujos ==============================================================#

                W1_  = np.concatenate((W1,np.flip(W1)[1:-1]))
                W2_  = np.concatenate((W2,np.flip(W2)[1:-1]))
                r_W1 = np.fft.ifft(W1_).real
                r_W2 = np.fft.ifft(W2_).real
                a_   = solve_toeplitz((r_W1[0:self.orden_filtro_tracto], r_W1[0:self.orden_filtro_tracto]), r_W1[1:self.orden_filtro_tracto+1])
                a_W1 = np.concatenate(([1.0], -a_))
                a_   = solve_toeplitz((r_W2[0:self.orden_filtro_tracto], r_W2[0:self.orden_filtro_tracto]), r_W2[1:self.orden_filtro_tracto+1])
                a_W2 = np.concatenate(([1.0], -a_))
                ''' 
                if self.pre_enfasis:
                #====== Cálculo del flujo 1
                    flujo_w1 = sig.lfilter(a_W1,[1,-self.alfa],self.speech)
                    maximo   = max(abs(flujo_w1))
                    if maximo > 1:
                        flujo_w1 = np.array([val/maximo for val in flujo_w1])

                    #====== Cálculo del flujo 2
                    flujo_w2 = sig.lfilter(a_W2,[1,-self.alfa],self.speech)
                    maximo   = max(abs(flujo_w2))
                    if maximo > 1:
                        flujo_w2 = np.array([val/maximo for val in flujo_w2])
                    num = np.zeros(len(a_W2))
                else:
                    #====== Cálculo del flujo 1
                    flujo_w1 = sig.lfilter(a_W1,[1,-alfa],self.speech-np.mean(self.speech))
                    maximo   = max(abs(flujo_w1))
                    if maximo > 1:
                        flujo_w1 = np.array([val/maximo for val in flujo_w1])
                    #====== Cálculo del flujo 2
                    flujo_w2 = sig.lfilter(a_W2,[1,-alfa],self.speech-np.mean(self.speech))
                    maximo   = max(abs(flujo_w2))
                    if maximo > 1:
                        flujo_w2 = np.array([val/maximo for val in flujo_w2])
                    num = np.zeros(len(a_W2))
                ''' 
                #====== Cálculo del flujo 1
                flujo_w1 = sig.lfilter(a_W1,[1,-self.alfa],self.speech)
                maximo   = max(abs(flujo_w1))
                if maximo > 1:
                    flujo_w1 = np.array([val/maximo for val in flujo_w1])

                #====== Cálculo del flujo 2
                flujo_w2 = sig.lfilter(a_W2,[1,-self.alfa],self.speech)
                maximo   = max(abs(flujo_w2))
                if maximo > 1:
                    flujo_w2 = np.array([val/maximo for val in flujo_w2])
                num = np.zeros(len(a_W2))
                #==================================== Cálculo de las funciones glóticas ==============================================================#
                dflujo_w1 = np.gradient(flujo_w1,self.tiempos)
                maximo    = max(abs(dflujo_w1))
                if maximo > 1:
                    dflujo_w1 = np.array([val/maximo for val in dflujo_w1])

                dflujo_w2 = np.gradient(flujo_w2,self.tiempos)
                maximo    = max(abs(dflujo_w2))
                if maximo > 1:
                    dflujo_w2= np.array([val/maximo for val in dflujo_w2])
                   
                #==================================== Cálculo de los polos y ceros ==============================================================#

                num      = np.zeros(len(a_W1))
                num[0]   = 1
                ceros_W1,polos_W1,_ = sig.tf2zpk(num,a_W1)
                num[0]   = 1
                ceros_W2,polos_W2,_ = sig.tf2zpk(num,a_W2)

                #==================================== Cálculo del error de predición ==============================================================#

                # Como para obtener los coeficientes del filtro se trabajó con la señal con pre-énfasis, para calcular el error de predición
                # también tengo que considerar la señal modificada.
                if self.pre_enfasis:
                    b = [1,-1]
                    a = [1]
                    señal = sig.lfilter(b,a,self.speech)
                else:
                    señal = self.speech - np.mean(self.speech)
                #====== Error de predición 1
                ep1   = sig.lfilter(b=a_W1,a=[1],x=señal)
                maximo   = max(abs(np.array(ep1)))
                if maximo > 1:
                    ep1 = np.array([val/maximo for val in ep1])
                #====== Error de predición 2
                ep2   = sig.lfilter(b=a_W2,a=[1],x=señal)
                maximo   = max(abs(np.array(ep2)))
                if maximo > 1:
                    ep2 = np.array([val/maximo for val in ep2])

                #==================================== Selección de mejor candidato ==============================================================#

                parametro_e1H1 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
                parametro_e2H2 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1])  
                
                if  parametro_e1H1< parametro_e2H2:
                    self.e1H1 = H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1])
                    self.e2H2 = H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1])             
                    self.sum_e1H1 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
                    self.sum_e2H2 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1])                     
                    self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:len(self.tiempos)-self.orden_filtro_tracto-1]
                    self.error_predic1 = ep1[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]
                    self.error_predic2 = ep2[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
                    self.W1 = W1
                    self.proceso_W1[i]=self.W1
                    self.W2 = W2
                    self.proceso_W2[i]=self.W2
                    self.H1 = H1
                    self.proceso_H1[i]=self.H1
                    self.H2 = H2
                    self.proceso_H2[i]=self.H2
                    self.a_W1 = a_W1
                    self.a_W2 = a_W2
                    self.ceros_W1 = ceros_W1
                    self.ceros_W2 = ceros_W2
                    self.polos_W1 = polos_W1
                    self.polos_W2 = polos_W2
                    self.flujo1 = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                    self.flujo2 = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                    self.dflujo1 = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                    self.dflujo2 = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                else:
                    self.e1H1 = H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1])
                    self.e2H2 = H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1])
                    self.sum_e1H1 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1])
                    self.sum_e2H2 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
                    self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:len(self.tiempos)-self.orden_filtro_tracto-1]
                    self.error_predic1 = ep2[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]
                    self.error_predic2 = ep1[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
                    self.W1 = W2
                    self.W2 = W1
                    self.H1 = H2
                    self.H2 = H1
                    self.a_W1 = a_W2
                    self.a_W2 = a_W1
                    self.ceros_W1 = ceros_W2
                    self.ceros_W2 = ceros_W1
                    self.polos_W1 = polos_W2
                    self.polos_W2 = polos_W1       
                    self.flujo1 = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                    self.flujo2 = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                    self.dflujo1 = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                    self.dflujo2 = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                self.proceso_e1H1.append(self.e1H1)
                self.proceso_e2H2.append(self.e2H2)
                self.proceso_sum_e1H1.append(self.sum_e1H1)
                self.proceso_sum_e2H2.append(self.sum_e2H2)
                self.proceso_error_predic1.append(self.error_predic1)
                self.proceso_error_predic2.append(self.error_predic2)
                self.proceso_W1[i]=self.W1
                self.proceso_W2[i]=self.W2
                self.proceso_H1[i]=self.H1
                self.proceso_H2[i]=self.H2
                self.proceso_a_W1.append(self.a_W1)
                self.proceso_a_W2.append(self.a_W2)
                self.proceso_ceros_W1.append(self.ceros_W1)
                self.proceso_ceros_W2.append(self.ceros_W2)
                self.proceso_polos_W1.append(self.polos_W1)
                self.proceso_polos_W2.append(self.polos_W2)          
                self.proceso_flujo1.append(self.flujo1)
                self.proceso_flujo2.append(self.flujo2)
                self.proceso_dflujo1.append(self.dflujo1)
                self.proceso_dflujo2.append(self.dflujo2)            
        else:
            H1 = np.array(self.H_i[0,:])
            H2 = np.array(self.H_i[1,:])
            W1 = self.W_i[:,0]
            W2 = self.W_i[:,1]
            W1 = W1**(2/self.p)
            W2 = W2**(2/self.p)
            #==================================== Cálculo de los flujos ==============================================================#

            W1_  = np.concatenate((W1,np.flip(W1)[1:-1]))
            W2_  = np.concatenate((W2,np.flip(W2)[1:-1]))
            r_W1 = np.fft.ifft(W1_).real
            r_W2 = np.fft.ifft(W2_).real
            a_   = solve_toeplitz((r_W1[0:self.orden_filtro_tracto], r_W1[0:self.orden_filtro_tracto]), r_W1[1:self.orden_filtro_tracto+1])
            a_W1 = np.concatenate(([1.0], -a_))
            a_   = solve_toeplitz((r_W2[0:self.orden_filtro_tracto], r_W2[0:self.orden_filtro_tracto]), r_W2[1:self.orden_filtro_tracto+1])
            a_W2 = np.concatenate(([1.0], -a_))
            ''' 
            if self.pre_enfasis:
            #====== Cálculo del flujo 1
                flujo_w1 = sig.lfilter(a_W1,[1,-self.alfa],self.speech)
                maximo   = max(abs(flujo_w1))
                if maximo > 1:
                    flujo_w1 = np.array([val/maximo for val in flujo_w1])

                #====== Cálculo del flujo 2
                flujo_w2 = sig.lfilter(a_W2,[1,-self.alfa],self.speech)
                maximo   = max(abs(flujo_w2))
                if maximo > 1:
                    flujo_w2 = np.array([val/maximo for val in flujo_w2])
                num = np.zeros(len(a_W2))
            else:
                #====== Cálculo del flujo 1
                flujo_w1 = sig.lfilter(a_W1,[1,-alfa],self.speech-np.mean(self.speech))
                maximo   = max(abs(flujo_w1))
                if maximo > 1:
                    flujo_w1 = np.array([val/maximo for val in flujo_w1])
                #====== Cálculo del flujo 2
                flujo_w2 = sig.lfilter(a_W2,[1,-alfa],self.speech-np.mean(self.speech))
                maximo   = max(abs(flujo_w2))
                if maximo > 1:
                    flujo_w2 = np.array([val/maximo for val in flujo_w2])
                num = np.zeros(len(a_W2))
            ''' 
            #====== Cálculo del flujo 1
            flujo_w1 = sig.lfilter(a_W1,[1,-self.alfa],self.speech)
            maximo   = max(abs(flujo_w1))
            if maximo > 1:
                flujo_w1 = np.array([val/maximo for val in flujo_w1])

            #====== Cálculo del flujo 2
            flujo_w2 = sig.lfilter(a_W2,[1,-self.alfa],self.speech)
            maximo   = max(abs(flujo_w2))
            if maximo > 1:
                flujo_w2 = np.array([val/maximo for val in flujo_w2])
            num = np.zeros(len(a_W2))
            #==================================== Cálculo de las funciones glóticas ==============================================================#
            dflujo_w1 = np.gradient(flujo_w1,self.tiempos)
            maximo    = max(abs(dflujo_w1))
            if maximo > 1:
                dflujo_w1 = np.array([val/maximo for val in dflujo_w1])

            dflujo_w2 = np.gradient(flujo_w2,self.tiempos)
            maximo    = max(abs(dflujo_w2))
            if maximo > 1:
                dflujo_w2= np.array([val/maximo for val in dflujo_w2])


            #==================================== Cálculo de los polos y ceros ==============================================================#

            num      = np.zeros(len(a_W1))
            num[0]   = 1
            ceros_W1,polos_W1,_ = sig.tf2zpk(num,a_W1)
            num[0]   = 1
            ceros_W2,polos_W2,_ = sig.tf2zpk(num,a_W2)

            #==================================== Cálculo del error de predición ==============================================================#

            # Como para obtener los coeficientes del filtro se trabajó con la señal con pre-énfasis, para calcular el error de predición
            # también tengo que considerar la señal modificada.
            if self.pre_enfasis:
                b = [1,-1]
                a = [1]
                señal = sig.lfilter(b,a,self.speech)
            else:
                señal = self.speech
            #====== Error de predición 1
            ep1   = sig.lfilter(b=a_W1,a=[1],x=señal)
            maximo   = max(abs(np.array(ep1)))
            if maximo > 1:
                ep1 = np.array([val/maximo for val in ep1])
            #====== Error de predición 2
            ep2   = sig.lfilter(b=a_W2,a=[1],x=señal)
            maximo   = max(abs(np.array(ep2)))
            if maximo > 1:
                ep2 = np.array([val/maximo for val in ep2])

            #==================================== Selección de mejor candidato ==============================================================#

            parametro_e1H1 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
            parametro_e2H2 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1])  
            
            if  parametro_e1H1< parametro_e2H2:
                self.e1H1 = H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1])
                self.e2H2 = H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1])
                self.sum_e1H1 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
                self.sum_e2H2 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]) 
                self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:len(self.tiempos)-self.orden_filtro_tracto-1]
                self.error_predic1 = ep1[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]
                self.error_predic2 = ep2[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
                self.W1 = W1
                self.W2 = W2
                self.H1 = H1
                self.H2 = H2
                self.a_W1 = a_W1
                self.a_W2 = a_W2
                self.ceros_W1 = ceros_W1
                self.ceros_W2 = ceros_W2
                self.polos_W1 = polos_W1
                self.polos_W2 = polos_W2           
                self.flujo1 = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                self.flujo2 = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                self.dflujo1 = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]
                self.dflujo2 = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
            else:
                self.e1H1 = H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1])
                self.e2H2 = H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1])
                self.sum_e1H1 = np.sum(H2[:len(H2)-self.orden_filtro_tracto-1]*np.abs(ep2[self.long_ventana_espect-1:len(ep2)-self.orden_filtro_tracto-1]))/np.sum(H2[:len(H2)-self.orden_filtro_tracto-1])
                self.sum_e2H2 = np.sum(H1[:len(H1)-self.orden_filtro_tracto-1]*np.abs(ep1[self.long_ventana_espect-1:len(ep1)-self.orden_filtro_tracto-1]))/np.sum(H1[:len(H1)-self.orden_filtro_tracto-1])
                self.tiempos_eH = self.tiempos[self.long_ventana_espect-1:len(self.tiempos)-self.orden_filtro_tracto-1]
                self.error_predic1 = ep2[self.orden_filtro_tracto-1:len(ep1)-self.orden_filtro_tracto-1]
                self.error_predic2 = ep1[self.orden_filtro_tracto-1:len(ep2)-self.orden_filtro_tracto-1]
                self.W1 = W2
                self.W2 = W1
                self.H1 = H2
                self.H2 = H1
                self.a_W1 = a_W2
                self.a_W2 = a_W1
                self.ceros_W1 = ceros_W2
                self.ceros_W2 = ceros_W1
                self.polos_W1 = polos_W2
                self.polos_W2 = polos_W1       
                self.flujo1 = flujo_w2[self.orden_filtro_tracto-1:len(flujo_w2)-self.orden_filtro_tracto-1]
                self.flujo2 = flujo_w1[self.orden_filtro_tracto-1:len(flujo_w1)-self.orden_filtro_tracto-1]
                self.dflujo1 = dflujo_w2[self.orden_filtro_tracto-1:len(dflujo_w2)-self.orden_filtro_tracto-1]
                self.dflujo2 = dflujo_w1[self.orden_filtro_tracto-1:len(dflujo_w1)-self.orden_filtro_tracto-1]

    def Run(self):
        self.Espectrograma()
        self.NNMF()
        self.GIF()