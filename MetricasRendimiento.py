import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt



class MetricasRendimiento:
    def __init__(self, señal,señal_estimada,fs_r,tiempos):
        
        # Copias originales de lase señales que no voy a modificar
        self.señal = señal
        self.señal_estimada = señal_estimada
        self.Error = -1
        self.fs = fs_r
        self.tiempos = tiempos
        self.Escalar()
        self.CalcularError()
    def Escalar(self):
        factor = np.dot(self.señal,self.señal_estimada)/np.sum(self.señal_estimada**2)
        self.señal_estimada = self.señal_estimada*factor

    def CalcularError(self):
        self.Curva_error = np.abs(self.señal-self.señal_estimada)
        numerador   = np.mean(np.abs(self.señal - self.señal_estimada))
        denominador = np.sqrt(np.mean(self.señal_estimada**2))
        self.Error  = numerador/denominador