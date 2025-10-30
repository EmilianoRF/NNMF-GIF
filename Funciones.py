
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt

import glob
import PyPDF2
import os



def Sincronizar(señal_estimada,señal):

    corr = sig.correlate(señal_estimada,señal)
    lags = sig.correlation_lags(len(señal_estimada), len(señal), mode='full')
    lag = lags[np.argmax(corr)]

    return np.roll(señal_estimada, -lag)

def Escalar(señal_estimada,señal):
    factor = np.dot(señal,señal_estimada)/np.sum(señal_estimada**2)
    return señal_estimada*factor

def FFT(señal, fs):
    """
    Calcula la FFT de la señal.
    
    Devuelve:

        FFT de la señal.
        El vector de frecuencias.
    """
    fft   = np.fft.fft(señal)
    N     = len(señal)
    freqs = np.fft.fftfreq(N, 1/fs)
    return fft[:N // 2],freqs[:N // 2] 


def NNMF(X,W,H,max_iter,norma='fro',iter_prueba = 15):
    '''
    returns X,W,H,error
    '''
    W_i = W
    H_i = H
    i = 1
    max_i = max_iter
    debajo_error = False

    epsilon = np.finfo(np.float32).eps

    X_aprox = np.matmul(W,H)
    error= []
    error.append(np.linalg.norm(X-X_aprox, ord=norma))
    while i <= iter_prueba:
        H_i = H_i * (np.matmul(W_i.transpose(),X)) /( np.matmul((np.matmul(W_i.transpose(),W_i)),H_i) + epsilon)
        X_aprox =  np.matmul(W_i,H_i)
        error_ = np.linalg.norm(X-X_aprox, ord='fro')
        error.append(error_)
        i+=1
    
    i = 1
    
    while not debajo_error and i <= max_i:

        W_i = W_i * (np.matmul(X,H_i.transpose()))/(np.matmul(W_i,np.matmul(H_i,H_i.transpose())) + epsilon)

        for i_ in range(0,W_i.shape[1]):
            maximo    = np.max(W_i[:,i_])
            W_i[:,i_] = W_i[:,i_]/maximo
        
        H_i = H_i * (np.matmul(W_i.transpose(),X)) /( np.matmul((np.matmul(W_i.transpose(),W_i)),H_i) + epsilon)

        X_aprox =  np.matmul(W_i,H_i)

        error_ = np.linalg.norm(X-X_aprox, ord='fro')
        error.append(error_)

        if error_ <= debajo_error:
            debajo_error = True
            
        i += 1
    return X_aprox,W_i,H_i,error

def STFT(señal,ventana,duracion_ventana,desplazamiento,fs):

    longitud_ventana = int(np.round(fs*duracion_ventana/1000))

    if ventana == 'boxcar':
        ventana  = sig.windows.boxcar(longitud_ventana)
    '''
    Si 
        Ns = cantidad de elementos de la señal.
        Nv = cantidad de elementos de la ventana.
        d  = cantidad de elementos que se desplaza la ventana.

    La cantidad de ventanas que voy a tener va a ser igual a 
        Cv = 1 + (NS-Nv)/d

    '''
    cantidad_ventanas = int( 1 + np.floor(( len(señal) - longitud_ventana) / desplazamiento))

    matriz_STFT     = np.zeros([longitud_ventana,cantidad_ventanas],dtype=complex)

    for i in range(cantidad_ventanas):
        # Esto me da la posición inicial en la señal a partir de la cual voy a hacer el ventaneo
        inicio   = i*desplazamiento

        # Esto me da una lista de la forma [0, 1, 2, ...,longitud_ventana - 1 ]
        desp_rel = np.arange(longitud_ventana)
        '''
        La suma me da un resultado de la forma 

        [inicio + 0,inicio + 1,inicio + 2, ..., inicio + longitud_ventana - 1]

        '''
        posicion_ventana   = inicio + desp_rel 

        señal_ventaneada = np.multiply(ventana, señal[posicion_ventana])
        matriz_STFT[:,i] = np.fft.fft(señal_ventaneada)
    return matriz_STFT

def NNMF_GIF(señal,fs):
    # Filtro de pre-énfasis H(z) = 1 - z^-1
    b = [1,-1]
    a = [1]
    señal_ = señal
    señal = sig.lfilter(b,a,señal)
    duracion_ventana = 5.5 # En ms
    desplazamiento   = 1    
    ventana = 'boxcar'
    matriz_STFT = STFT(señal,ventana,duracion_ventana,desplazamiento,fs)
    # Me quedo con la mitad+1 de las frecuencias
    matriz_STFT          = matriz_STFT[0:matriz_STFT.shape[0]//2+1, :]
    energia = []
    for i in range(0,matriz_STFT.shape[1]):
        energia.append(np.sum(np.abs(matriz_STFT[:,i])**2))
    maximo = max(energia)
    energia = [valor/maximo for valor in energia]
    # Creo el espectrograma teniendo en cuenta la compresión del paper
    p = 0.55
    espectrograma =np.abs(matriz_STFT)**p 
    indice_maximo = np.where(energia == max(energia))[0][0]
    indice_minimo = np.where(energia == min(energia))[0][0]
    X = espectrograma
    N = X.shape[0]
    M = X.shape[1]
    R = 2
    W = np.array([np.array(espectrograma[:,indice_maximo]),np.array(espectrograma[:,indice_minimo])]).transpose()
    H = np.random.rand(R, M)
    max_i = 2000
    X_aprox,W_i,H_i,error = NNMF(X,W,H,max_i)
    H1 = np.array(H_i[0,:])/max(H_i[0,:])
    H2 = np.array(H_i[1,:])/max(H_i[1,:])
    W1 = W_i[:,0]
    W2 = W_i[:,1]
    W1 = W1**(2/p)
    W2 = W2**(2/p)
    W1_ =np.concatenate((W1,np.flip(W1)[1:-1]))
    W2_ =np.concatenate((W2,np.flip(W2)[1:-1]))
    r_W1 = np.fft.ifft(W1_).real
    r_W2 = np.fft.ifft(W2_).real
    
    orden = 12
    a_   = solve_toeplitz((r_W1[0:orden-1], r_W1[0:orden-1]), r_W1[1:orden])
    a_W1 = np.concatenate(([1.0], -a_))
    a_ = solve_toeplitz((r_W2[0:orden-1], r_W2[0:orden-1]), r_W2[1:orden])
    a_W2 = np.concatenate(([1.0], -a_))

    alfa = 1
    flujo_w1 = sig.lfilter(a_W1,[1,-alfa],señal_)
    flujo_w1 = [val/max(abs(flujo_w1)) for val in flujo_w1]
    flujo_w2 = sig.lfilter(a_W2,[1,-alfa],señal_)
    flujo_w2 = [val/max(abs(flujo_w2)) for val in flujo_w2]

    return flujo_w1,flujo_w2,a_W1,a_W2,H1,H2,espectrograma,X_aprox



def unir_pdfs(ruta_entrada, nombre_salida):
    """
    Une todos los archivos PDF en la carpeta especificada en un único PDF.

    Parámetros:
        ruta_entrada (str): Ruta de la carpeta que contiene los archivos PDF.
        nombre_salida (str): Nombre del archivo PDF resultante (puede incluir o no .pdf).
    """
    # Asegurarse de que el nombre tenga la extensión .pdf
    if not nombre_salida.lower().endswith(".pdf"):
        nombre_salida += ".pdf"

    # Buscar todos los archivos PDF en la carpeta
    patron_busqueda = os.path.join(ruta_entrada, "*.pdf")
    filepaths = glob.glob(patron_busqueda)

    if not filepaths:
        print("No se encontraron archivos PDF en la ruta especificada.")
        return

    # Crear el objeto merger
    pdf_merger = PyPDF2.PdfMerger()

    # Agregar los PDFs al merger
    for filepath in filepaths:
        pdf_merger.append(filepath)

    # Escribir el archivo de salida en la misma carpeta de entrada
    salida = os.path.join(ruta_entrada, nombre_salida)
    with open(salida, "wb") as output_pdf:
        pdf_merger.write(output_pdf)


parula_data = [
    [0.2081, 0.1663, 0.5292],
    [0.2116, 0.1898, 0.5777],
    [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771],
    [0.1959, 0.2645, 0.7279],
    [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303],
    [0.0591, 0.3598, 0.8683],
    [0.0117, 0.3875, 0.8820],
    [0.0050, 0.4086, 0.8828],
    [0.0165, 0.4266, 0.8786],
    [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641],
    [0.0629, 0.4737, 0.8554],
    [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384],
    [0.0793, 0.5200, 0.8312],
    [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240],
    [0.0488, 0.5772, 0.8228],
    [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135],
    [0.0239, 0.6287, 0.8038],
    [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768],
    [0.0267, 0.6642, 0.7607],
    [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254],
    [0.0843, 0.6928, 0.7062],
    [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646],
    [0.1801, 0.7177, 0.6424],
    [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954],
    [0.3022, 0.7376, 0.5712],
    [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244],
    [0.4420, 0.7481, 0.5033],
    [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661],
    [0.5709, 0.7485, 0.4494],
    [0.6099, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188],
    [0.6834, 0.7435, 0.4044],
    [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768],
    [0.7858, 0.7356, 0.3633],
    [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360],
    [0.8824, 0.7274, 0.3217],
    [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886],
    [0.9739, 0.7314, 0.2666],
    [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164],
    [0.9955, 0.7861, 0.1967],
    [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633],
    [0.9695, 0.8481, 0.1475],
    [0.9620, 0.8705, 0.1309],
    [0.9584, 0.8949, 0.1132],
    [0.9598, 0.9218, 0.0948],
    [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538]
]

parula_map = LinearSegmentedColormap.from_list('parula', parula_data)
