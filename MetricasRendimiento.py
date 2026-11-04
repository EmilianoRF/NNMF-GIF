
import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt

import Funciones as Funciones

class MetricasRendimiento:
    def __init__(self, señal_referencia,señal_estimada,tiempos):
        
        self.señal_referencia = señal_referencia
        self.señal_estimada = señal_estimada
        self.tiempos = tiempos
        self.Error = -1
        self.Sincronizar()
        self.Escalar()
        self.CalcularError()
        
    def Escalar(self):
        factor = np.dot(self.señal_referencia,self.señal_estimada)/np.sum(self.señal_estimada**2)
        self.señal_estimada = self.señal_estimada*factor

    def CalcularError(self):
        numerador   = np.median(np.abs(self.señal_referencia - self.señal_estimada))
        denominador = np.sqrt(np.mean(self.señal_referencia**2))
        self.Error  = numerador/denominador

    def Sincronizar(self):
        se,sr,t= Funciones.Sincronizar2(self.señal_estimada,self.señal_referencia ,self.tiempos)
        to_mascara = 10.0
        tf_mascara = 40.0
        mask = (t >= to_mascara) & (t <= tf_mascara)
        self.señal_referencia = sr[mask]
        self.señal_estimada = se[mask]
        self.tiempos = t[mask]
'''
import numpy as np
import pandas as pd
import Funciones as Funciones


class MetricasRendimiento:
    """
    Compara la señal estimada con la de referencia y expone, además del Error,
    los parámetros aerodinámicos NAQ, QOQ, H1-H2, HRF y PSP.

    El Error (y el factor de escala) se calculan SIEMPRE sobre la señal que se recibe,
    sin ninguna transformación previa.

    Parámetros
    ----------
    señal_referencia, señal_estimada : señales a comparar (se asumen flujos glóticos,
                 que es lo que necesitan los parámetros aerodinámicos)
    tiempos    : vector de tiempos [ms]
    fs         : frecuencia de muestreo [Hz]. Si es None se estima desde `tiempos` (en ms).
    excluir_primer_pulso : el primer pulso detectado es parcial (la ventana lo corta),
                 por lo que se excluye de las medias y de los espectros H1-H2.

    Señales (ya sincronizadas, recortadas a 10-40 ms y con la estimada escalada):
        señal_referencia, señal_estimada     -> las señales recibidas; sobre ellas se calcula el Error
        funcion_referencia, funcion_estimada -> derivadas (np.gradient) de las anteriores,
                                                usadas para los parámetros aerodinámicos
        tiempos                              -> tiempos [ms] de todas ellas
        factor_escala                        -> factor por mínimos cuadrados aplicado a la estimada

    El factor de escala se aplica igual a la señal estimada y a su derivada.

    Parámetros aerodinámicos (media sobre los pulsos glóticos válidos de la ventana).
    Se calculan la PRIMERA vez que se accede a alguno:
        NAQ, QOQ, H1H2, HRF, PSP              -> señal estimada
        NAQ_ref, QOQ_ref, H1H2_ref, ...       -> señal de referencia
        Dif_NAQ, Dif_QOQ, Dif_H1H2, ...       -> referencia - estimada
        N_GCIs, N_GCIs_ref                    -> cantidad de GCIs detectados
        vq_estimada, vq_referencia            -> dict con el detalle (valores por pulso,
                                                 GCIs, espectros H1/H2, señales usadas)
        espectro_estimada, espectro_referencia -> dict con el espectro promedio de los pulsos:
                                                 f [Hz], S [dB], S_pulsos (un espectro por fila),
                                                 f_h1, a_h1, f_h2, a_h2, h1h2, n_pulsos
    Método:
        ResumenVQ()                           -> DataFrame (Estimada / Referencia / Dif)
    """
    PARAMETROS_VQ = ('NAQ', 'QOQ', 'H1H2', 'HRF', 'PSP')
    _NOMBRES_VQ = (('vq_estimada', 'vq_referencia', 'N_GCIs', 'N_GCIs_ref')
                   + PARAMETROS_VQ
                   + tuple(p + '_ref' for p in PARAMETROS_VQ)
                   + tuple('Dif_' + p for p in PARAMETROS_VQ)
                   + ('espectro_estimada', 'espectro_referencia'))

    def __init__(self, señal_referencia, señal_estimada, tiempos,
                 fs=None, excluir_primer_pulso=True):

        self.señal_referencia = np.asarray(señal_referencia, dtype=float).ravel()
        self.señal_estimada = np.asarray(señal_estimada, dtype=float).ravel()
        self.tiempos = tiempos
        self.fs = fs
        self.excluir_primer_pulso = excluir_primer_pulso
        self.Error = -1
        self.factor_escala = 1.0
        self.Sincronizar()
        self.Escalar()
        self.CalcularError()

    def Escalar(self):
        factor = np.dot(self.señal_referencia, self.señal_estimada) / np.sum(self.señal_estimada**2)
        self.factor_escala = factor
        self.señal_estimada = self.señal_estimada * factor
        self.funcion_estimada = self.funcion_estimada * factor   # la derivada es lineal: mismo factor

    def CalcularError(self):
        numerador = np.median(np.abs(self.señal_referencia - self.señal_estimada))
        denominador = np.sqrt(np.mean(self.señal_referencia**2))
        self.Error = numerador / denominador

    def Sincronizar(self):
        se, sr, t = Funciones.Sincronizar2(self.señal_estimada, self.señal_referencia, self.tiempos)
        se = np.asarray(se, dtype=float)
        sr = np.asarray(sr, dtype=float)
        t = np.asarray(t, dtype=float)
        # Derivadas de las señales ya sincronizadas (antes de recortar,
        # para que no haya efecto de borde en los extremos de la máscara)
        dse = np.gradient(se)
        dsr = np.gradient(sr)
        to_mascara = 10.0
        tf_mascara = 40.0
        mask = (t >= to_mascara) & (t <= tf_mascara)
        self.señal_referencia = sr[mask]
        self.señal_estimada = se[mask]
        self.funcion_referencia = dsr[mask]
        self.funcion_estimada = dse[mask]
        self.tiempos = t[mask]

    # ------------------------------------------------------------------
    # Parámetros aerodinámicos (cálculo diferido)
    # ------------------------------------------------------------------
    def __getattr__(self, nombre):
        # Solo se invoca cuando el atributo todavía no existe
        if nombre in MetricasRendimiento._NOMBRES_VQ:
            self.CalcularParametrosVQ()
            return self.__dict__[nombre]
        raise AttributeError(f"'{type(self).__name__}' no tiene el atributo '{nombre}'")

    def _vq_de_senal(self, gf, gfd, fs):
        """Parámetros VQ a partir del flujo `gf` y su derivada `gfd` (tiempos: self.tiempos [ms])."""
        gf = np.asarray(gf, dtype=float).ravel()
        gfd = np.asarray(gfd, dtype=float).ravel()
        t = np.asarray(self.tiempos, dtype=float).ravel()

        detalle = []
        NAQ, QOQ, H1H2, HRF, PSP, gci = Funciones.get_vq_params(
            gf, gfd, fs, return_gci=True, detalle=detalle)

        idx = np.clip(np.round(gci * fs).astype(int), 0, len(gf) - 1)
        valores = {'NAQ': NAQ[:, 1], 'QOQ': QOQ[:, 1], 'H1H2': H1H2[:, 1],
                   'HRF': HRF[:, 1], 'PSP': PSP[:, 1]}

        medias = {}
        for k, x in valores.items():
            x = np.where(x != 0, x, np.nan)        # 0 = pulso no evaluado
            if self.excluir_primer_pulso:
                x[:1] = np.nan
            medias[k] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

        if self.excluir_primer_pulso:
            detalle = [d for d in detalle if d['n'] != 0]

        espectro = self._espectro_promedio(detalle)

        return dict(gf=gf, gfd=gfd, t=t, idx=idx, t_gci=t[idx],
                    valores=valores, medias=medias, detalle=detalle,
                    espectro=espectro)

    def CalcularParametrosVQ(self):
        """Calcula NAQ, QOQ, H1-H2, HRF y PSP (estimada y referencia). Se llama sola al acceder a ellos."""
        fs = self.fs
        if fs is None:
            fs = 1000.0 / np.median(np.diff(np.asarray(self.tiempos, dtype=float)))  # tiempos en ms
        fs = int(round(fs))

        est = self._vq_de_senal(self.señal_estimada, self.funcion_estimada, fs)
        ref = self._vq_de_senal(self.señal_referencia, self.funcion_referencia, fs)

        self.vq_estimada = est
        self.vq_referencia = ref
        self.espectro_estimada = est['espectro']
        self.espectro_referencia = ref['espectro']
        self.N_GCIs = len(est['t_gci'])
        self.N_GCIs_ref = len(ref['t_gci'])
        for p in self.PARAMETROS_VQ:
            setattr(self, p, est['medias'][p])
            setattr(self, p + '_ref', ref['medias'][p])
            setattr(self, 'Dif_' + p, ref['medias'][p] - est['medias'][p])   # referencia - estimada

    # ------------------------------------------------------------------
    # Espectro promedio H1-H2
    # ------------------------------------------------------------------
    @staticmethod
    def _extraer_espectro(d):
        """Dado un elemento de `detalle` (un pulso), devuelve (f [Hz], S [dB], f_h1 [Hz], f_h2 [Hz])."""
        return d['f'], d['S_db'], d['f_h1'], d['f_h2']

    def _espectro_promedio(self, detalle):
        """Promedia (en dB) los espectros de los pulsos de `detalle` y ubica H1 y H2."""
        if len(detalle) == 0:
            return None
        extraidos = [self._extraer_espectro(d) for d in detalle]

        f = np.asarray(extraidos[0][0], dtype=float).ravel()
        S = []
        for fi, Si, _, _ in extraidos:
            fi = np.asarray(fi, dtype=float).ravel()
            Si = np.asarray(Si, dtype=float).ravel()
            if fi.shape == f.shape and np.allclose(fi, f):
                S.append(Si)
            else:                                   # distinto eje de frecuencias: se interpola
                S.append(np.interp(f, fi, Si))
        S = np.vstack(S)
        S_prom = np.nanmean(S, axis=0)

        f_h1 = float(np.nanmean([e[2] for e in extraidos]))   # frecuencia media de H1
        f_h2 = float(np.nanmean([e[3] for e in extraidos]))   # frecuencia media de H2
        a_h1 = float(np.interp(f_h1, f, S_prom))              # amplitud en el espectro promedio
        a_h2 = float(np.interp(f_h2, f, S_prom))

        return dict(f=f, S=S_prom, S_pulsos=S,
                    f_h1=f_h1, a_h1=a_h1, f_h2=f_h2, a_h2=a_h2,
                    h1h2=a_h1 - a_h2, n_pulsos=S.shape[0])

    def ResumenVQ(self):
        """DataFrame con los parámetros: columnas Estimada / Referencia / Dif (referencia - estimada)."""
        return pd.DataFrame(
            {'Estimada':   [getattr(self, p) for p in self.PARAMETROS_VQ],
             'Referencia': [getattr(self, p + '_ref') for p in self.PARAMETROS_VQ],
             'Dif':        [getattr(self, p + '_ref') - getattr(self, p) for p in self.PARAMETROS_VQ]},
            index=['NAQ', 'QOQ', 'H1-H2 [dB]', 'HRF', 'PSP'])
'''