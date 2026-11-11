
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
from scipy import signal as sig
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from scipy.signal import butter, filtfilt, hilbert, correlate, correlation_lags


import glob
import PyPDF2
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import scipy.signal as sig
from scipy.signal import find_peaks


def energiaTeager(señal,tiempos):
        energia = [señal[n]**2 - señal[n-1]*señal[n+1] for n in range(1,len(señal)-1)]
        energia = energia/max(energia)
        tiempos_ = tiempos[1:len(tiempos)-1]
        return energia,tiempos_


def get_vq_params(gf, gfd, fs, return_gci=False, f0_range=(50.0, 500.0),
                  gci_threshold=0.15, detalle=None):
    """
    Estima NAQ, QOQ, H1-H2, HRF y PSP a partir del flujo glótico (gf) y su
    derivada (gfd). Los GCIs se detectan automáticamente desde gfd.

    Port de get_vq_params.m (Covarep, John Kane, Trinity College Dublin, LGPL v3+).

    Parámetros
    ----------
    gf, gfd       : listas o arrays 1D de la misma longitud
    fs            : frecuencia de muestreo [Hz]
    return_gci    : si True, devuelve además los GCIs detectados [s]
    f0_range      : (f0_min, f0_max) en Hz para la detección de GCIs
    gci_threshold : umbral relativo al máximo de -gfd para aceptar un GCI
    detalle       : lista opcional. Si se pasa, se le agrega un dict por cada pulso
                    con H1-H2 evaluado, con las claves:
                        n     : índice del pulso (posición en el vector de GCIs)
                        F0    : frecuencia fundamental del pulso [Hz]
                        f     : vector de frecuencias [Hz]
                        S_db  : espectro del pulso [dB]
                        f_h1  : frecuencia del pico H1 [Hz]
                        f_h2  : frecuencia del pico H2 [Hz]

    Devuelve
    --------
    NAQ, QOQ, H1H2, HRF, PSP : arrays (M, 2) con columnas [tiempo (s), valor]
    gci (opcional)           : array (M,) con los GCIs en segundos

    Requiere: numpy, scipy
    """
    import numpy as np
    from scipy.signal import find_peaks

    EPS = 1e-12

    # ------------------------------------------------------------------
    # Funciones auxiliares
    # ------------------------------------------------------------------
    def mag2db(x):
        return 20.0 * np.log10(np.maximum(x, EPS))

    def estimate_T0(x, frame_ms=50.0, min_corr=0.3):
        """Período típico (muestras) por autocorrelación en tramas."""
        lag_min = int(fs / f0_range[1])
        lag_max = int(fs / f0_range[0])
        flen = max(int(frame_ms * 1e-3 * fs), 2 * lag_max)
        hop = flen // 2

        lags = []
        for s in range(0, len(x) - flen + 1, hop):
            fr = x[s:s + flen] - np.mean(x[s:s + flen])
            e0 = np.dot(fr, fr)
            if e0 < EPS or lag_max <= lag_min:
                continue
            n = 1 << (2 * flen - 1).bit_length()
            spec = np.fft.rfft(fr, n)
            ac = np.fft.irfft(spec * np.conj(spec))[:lag_max + 1] / e0
            k = lag_min + int(np.argmax(ac[lag_min:lag_max + 1]))
            if ac[k] > min_corr:
                lags.append(k)

        if lags:
            return float(np.median(lags))

        # Fallback: autocorrelación global
        xc = x - np.mean(x)
        n = 1 << (2 * len(xc) - 1).bit_length()
        spec = np.fft.rfft(xc, n)
        ac = np.fft.irfft(spec * np.conj(spec))[:lag_max + 1]
        return float(lag_min + np.argmax(ac[lag_min:lag_max + 1]))

    def detect_gci(x_gfd):
        """GCIs = picos negativos de gfd (índices base 0)."""
        T0_est = estimate_T0(x_gfd)
        distance = max(int(0.6 * T0_est), 1)
        x = -x_gfd
        peaks, _ = find_peaks(x, height=gci_threshold * np.max(x), distance=distance)
        return peaks.astype(int)

    def find_amid_t(glot_adj, Amid, Tz):
        """Inicio y fin de la fase cuasi-abierta (índices base 0)."""
        n = Tz
        while glot_adj[n] > Amid and n > 2:
            n -= 1
        T1 = n
        n = Tz
        while glot_adj[n] > Amid and n < len(glot_adj) - 3:
            n += 1
        T2 = n
        return T1, T2

    def psp_get_a(X):
        """Coeficiente 'a' de la parábola (sección 3.2.1, Alku et al. 1997)."""
        NE_min = 0.01
        N = 3
        X = np.asarray(X, dtype=float).ravel()
        X2 = X ** 2
        k2 = np.arange(len(X), dtype=float) ** 2

        k2_sum = np.sum(k2[:N - 1])
        k4_sum = np.sum(k2[:N - 1] ** 2)
        X_k1_sum = np.sum(X[:N - 1])
        X_k1_k2_sum = np.sum(X[:N - 1] * k2[:N - 1])
        X2_k1_sum = np.sum(X2[:N - 1])

        while True:
            k2_sum += k2[N - 1]
            k4_sum += k2[N - 1] * k2[N - 1]
            X_k1_sum += X[N - 1]
            X_k1_k2_sum += X[N - 1] * k2[N - 1]
            X2_k1_sum += X2[N - 1]

            # Ecuación 5
            a = (N * X_k1_k2_sum - X_k1_sum * k2_sum) / (N * k4_sum - k2_sum ** 2)

            resid = X[:N] - a * k2[:N]
            # Ecuación 4
            b = np.sum(resid) / N
            NE = np.sum((resid - b) ** 2) / X2_k1_sum

            if NE < NE_min and N < len(X):
                N += 1
            else:
                break
        return a

    # ------------------------------------------------------------------
    # Preparación
    # ------------------------------------------------------------------
    gf = np.asarray(gf, dtype=float).ravel()
    gfd = np.asarray(gfd, dtype=float).ravel()
    fs_int = int(round(fs))

    GCI = detect_gci(gfd)
    M = len(GCI)

    F0min = 20
    F0max = 500
    NAQ = np.zeros(M)
    QOQ = np.zeros(M)
    H1H2 = np.zeros(M)
    HRF = np.zeros(M)
    PSP = np.zeros(M)

    glot_shift = int(round(0.5 / 1000 * fs))
    qoq_level = 0.5          # umbral para QOQ
    T0_num = 3               # nº de pulsos locales para el espectro armónico
    min_harm_num = 5
    HRF_freq_max = 5000      # frecuencia máxima para HRF
    PSP_fft_size = 2048      # según Alku et al. (1997)

    if M < 2:
        out = [np.column_stack([GCI / fs, v]) for v in (NAQ, QOQ, H1H2, HRF, PSP)]
        return (*out, GCI / fs) if return_gci else tuple(out)

    # ------------------------------------------------------------------
    # Procesamiento pulso a pulso
    # ------------------------------------------------------------------
    for n in range(M):
        if n == 0:
            start = 0
            stop = GCI[n]
            T0 = GCI[n + 1] - GCI[n]
        else:
            start = GCI[n - 1]
            stop = GCI[n]
            T0 = GCI[n] - GCI[n - 1]

        if T0 == 0:
            continue
        F0 = fs / T0
        if not (F0min < F0 < F0max):
            continue

        # Compensación de la deriva de la línea cero
        if start != stop:
            line = np.linspace(gf[start], gf[stop], stop - start + 1)
        else:
            line = np.zeros(1)
        gf_seg_comp = gf[start:stop + 1] - line

        stop2 = stop + glot_shift if stop + glot_shift <= len(gfd) - 1 else stop
        gfd_seg = gfd[start:stop2 + 1]

        # NAQ y QOQ
        d_peak = np.max(np.abs(gfd_seg))
        max_idx = int(np.argmax(gf_seg_comp))
        f_ac = gf_seg_comp[max_idx]
        Amid = f_ac * qoq_level
        T1, T2 = find_amid_t(gf_seg_comp, Amid, max_idx)

        NAQ[n] = (f_ac / d_peak) / T0 if d_peak > 0 else 0.0
        QOQ[n] = (T2 - T1) / (fs / F0)

        # Trama para H1-H2 / HRF
        half = int(round((T0 * T0_num) / 2))
        f_start = GCI[n] - half if GCI[n] - half >= 0 else 0
        f_stop = GCI[n] + half if GCI[n] + half <= len(gfd) - 1 else len(gfd) - 1
        f_frame = gfd[f_start:f_stop + 1] * (2 ** 15)
        f_win = f_frame * np.hamming(len(f_frame))
        f_spec = mag2db(np.abs(np.fft.fft(f_win, fs_int)))[: fs_int // 2 + 1]

        # H1-H2 / HRF
        h_idx, _ = find_peaks(f_spec, distance=max(int(F0 / 2), 1))
        h_amp = f_spec[h_idx]
        HRF_harm_num = int(np.floor(HRF_freq_max / F0))
        if len(h_idx) >= min_harm_num and HRF_harm_num >= 2:
            harmonics = np.arange(1, HRF_harm_num + 1) * F0
            f0_idx = np.argmin(np.abs(h_idx[:, None] - harmonics[None, :]), axis=0)
            H1H2[n] = h_amp[f0_idx[0]] - h_amp[f0_idx[1]]
            HRF[n] = np.sum(h_amp[f0_idx[1:]]) / h_amp[f0_idx[0]]

            # Detalle por pulso (espectro y posición de H1/H2) para uso externo
            if detalle is not None:
                detalle.append({
                    'n': n,
                    'F0': float(F0),
                    # FFT de fs_int puntos -> 1 Hz por bin, el índice equivale a Hz
                    'f': np.arange(len(f_spec), dtype=float),
                    'S_db': f_spec,
                    'f_h1': float(h_idx[f0_idx[0]]),
                    'f_h2': float(h_idx[f0_idx[1]]),
                })

        # PSP
        if n > 0:
            seg = gf[GCI[n - 1]:GCI[n] + 1].copy()
            seg -= np.min(seg)                       # mínimo a cero
            mx = np.max(seg)
            if mx > 0:
                seg /= mx                            # escala a la unidad
            X = mag2db(np.abs(np.fft.fft(seg, PSP_fft_size)))     # zero-pad
            X = X[np.linspace(1, fs, len(X)) <= fs / 2]           # hasta Nyquist
            a = psp_get_a(X)

            # Medida de máximo decaimiento espectral
            Tp = int(round(fs / F0))
            a_max_sig = np.ones(Tp) / Tp
            X_a_max = mag2db(np.abs(np.fft.fft(a_max_sig, PSP_fft_size)))
            X_a_max = X_a_max[np.linspace(1, fs, len(X_a_max)) <= fs / 2]
            a_max = psp_get_a(X_a_max)

            PSP[n] = a / a_max

    # ------------------------------------------------------------------
    # Añadir el tiempo a los parámetros
    # ------------------------------------------------------------------
    t = GCI / fs
    NAQ = np.column_stack([t, NAQ])
    QOQ = np.column_stack([t, QOQ])
    H1H2 = np.column_stack([t, H1H2])
    HRF = np.column_stack([t, HRF])
    PSP = np.column_stack([t, PSP])

    if return_gci:
        return NAQ, QOQ, H1H2, HRF, PSP, t
    return NAQ, QOQ, H1H2, HRF, PSP
def IF_SNF_OneMetric(x, fs, numMetric, f0=None,bw = None):
    """
    Inverse filtering using a Single Notch Filter optimized by one metric.

    Parameters:
        x         : array-like, input signal
        fs        : float, sampling frequency (Hz)
        numMetric : int, metric index to optimize (1, 2 or 3)
        f0        : float (optional), starting frequency for search (Hz)
                    if None, search starts at 300 Hz

    Returns:
        gvv : inverse filtered signal
        br  : numerator filter coefficient
        ar  : denominator filter coefficients
        F1  : estimated first formant frequency (Hz)
        BW1 : estimated first formant bandwidth (Hz)
    """
    x0 = np.array(x, dtype=float).flatten()

    if f0 is not None:
        fAR = np.arange(f0, 960, 10)   # f0:10:950
    else:
        fAR = np.arange(300, 960, 10)  # 300:10:950
    if bw is None:
        BW = np.array([70.0])
    else:
        BW = np.array([bw])    

    n_fAR = len(fAR)
    n_BW  = len(BW)

    metric_01     = np.full((n_fAR, n_BW), np.nan)
    metric_01min  = np.full(n_fAR, np.nan)
    metric_01_I   = np.zeros(n_fAR, dtype=int)

    for ii, fPole in enumerate(fAR):
        for jj, BWpole in enumerate(BW):
            x1, _, _ = SNF(x0, fPole, fs, BWpole)
            m = metricsIF(x1, x0)
            metric_01[ii, jj] = m[numMetric - 1]

        metric_01min[ii] = np.min(metric_01[ii, :])
        metric_01_I[ii]  = np.argmin(metric_01[ii, :])

    nMinfAR1 = np.argmin(metric_01min)
    nMinBW   = metric_01_I[nMinfAR1]

    F1  = fAR[nMinfAR1]
    BW1 = BW[nMinBW]

    gvv, br, ar = SNF(x0, F1, fs, BW1)
    gvv = gvv.flatten()

    return gvv, br, ar, F1, BW1

def _tukeywin(n, r=0.5):
    """Tukey (tapered cosine) window — equivalent to MATLAB's tukeywin."""
    if r <= 0:
        return np.ones(n)
    if r >= 1:
        return np.hanning(n)
    w    = np.ones(n)
    half = int(np.floor(r * n / 2))
    for i in range(half):
        w[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (r * n)))
    for i in range(n - half, n):
        w[i] = w[n - 1 - i]
    return w


def metricsIF(x, x0):
    """
    Compute inverse filtering quality metrics 1, 2 and 3.

    Parameters:
        x  : array-like, inverse filtered (IF) signal
        x0 : array-like, original (non-filtered) signal

    Returns:
        metrics : array of 3 values [metric1, metric2, metric3]
    """
    x  = np.array(x,  dtype=float).flatten()
    x0 = np.array(x0, dtype=float).flatten()

    # Tukey window
    w  = _tukeywin(len(x), 0.5)
    x  = x  * w
    x0 = x0 * w

    lenx = len(x)
    x1   = x.copy()

    # Differences
    x0diff     = np.diff(x0)
    x1diff     = np.diff(x1)
    x0diffdiff = sig.filtfilt([1, -1], [1], x0)
    x1diffdiff = sig.filtfilt([1, -1], [1], x1)

    # Central segment (25% to 75%)
    i0 = int(np.floor(lenx * 0.25))-1
    i1 = int(np.floor(lenx * 0.75))

    x1         = x1[i0:i1]
    x0         = x0[i0:i1]
    x1diff     = x1diff[i0:i1]
    x0diff     = x0diff[i0:i1]
    x1diffdiff = x1diffdiff[i0:i1]
    x0diffdiff = x0diffdiff[i0:i1]

    metrics = np.full(3, np.nan)

    # Metric 1: sum(|Δ²x|)
    metrics[0] = np.sum(np.abs(x1diffdiff)) / np.sum(np.abs(x0diffdiff))

    # Metric 2: F99(x) - F1(x)
    metrics[1] = (np.percentile(x1, 99) - np.percentile(x1, 1)) / \
                 (np.percentile(x0, 99) - np.percentile(x0, 1))

    # Metric 3: sum(|Δx|)
    metrics[2] = np.sum(np.abs(x1diff)) / np.sum(np.abs(x0diff))

    return metrics


def vtractz(p, bw, fs):
    """
    Produce a vector of z-domain poles corresponding to input pole frequencies
    and bandwidths, assuming an all-pole model of the vocal tract.

    Parameters:
        p  : array-like, pole frequencies in Hz
        bw : array-like, pole bandwidths in Hz
        fs : float, sampling frequency in Hz

    Returns:
        zp : numpy array of complex z-domain poles (pairs a+jb, a-jb)
    """
    p  = np.atleast_1d(np.array(p, dtype=float))
    bw = np.atleast_1d(np.array(bw, dtype=float))

    n  = len(p)
    zp = np.zeros(2 * n, dtype=complex)

    B = np.pi * bw / fs          # Bandwidths in radians
    P = 2 * np.pi * p / fs       # Pole frequencies in radians

    # Z-plane pole magnitudes
    r = -np.cos(B) + 2 - np.sqrt((np.cos(B) - 1) * (np.cos(B) - 3))

    for m in range(n):
        if p[m] < fs / 4:        # 1st quadrant (positive real part)
            a = r[m] / np.sqrt(1 + np.tan(P[m])**2)
        else:                    # 2nd quadrant (negative real part)
            a = -r[m] / np.sqrt(1 + np.tan(P[m])**2)

        b = a * np.tan(P[m])     # Imaginary part

        zp[2 * m]     = complex(a,  b)
        zp[2 * m + 1] = complex(a, -b)

    return zp


def SNF(x, fc, fs, BW=None):
    """
    Single Notch Filter with unitary gain at DC.

    Parameters:
        x  : array-like, input signal
        fc : array-like, notch frequency/ies in Hz
        fs : float, sampling frequency in Hz
        BW : array-like (optional), bandwidths in Hz (default: 70 Hz for each fc)

    Returns:
        x_filt : filtered signal
        b      : numerator filter coefficients
        a      : denominator filter coefficients
    """
    fc = np.atleast_1d(np.array(fc, dtype=float))
    x  = np.array(x, dtype=float)

    if BW is None:
        BW = 70 * np.ones(len(fc))
    else:
        BW = np.atleast_1d(np.array(BW, dtype=float))

    zp = vtractz(fc, BW, fs)       # Z-domain poles

    # zp2tf with no zeros, poles=zp, gain=1
    # scipy's zpk2tf returns (b, a) → here 'a' is the denominator
    _, a = sig.zpk2tf([], zp, 1)

    # Gain normalization: b = sum(a) ensures unity gain at DC (z=1)
    b = np.array([np.sum(a)])

    x_filt = sig.lfilter(a,b, x)

    return x_filt, b, a

def IF_OnePole_fC_BW(x, fs):
    """
    Inverse filtering of bandlimited (<1 kHz) signals.

    Parameters:
        x  : array-like, input signal (LP filtered <1kHz, no DC component)
        fs : float, sampling frequency (Hz)

    Returns:
        gvv : inverse filtered signal
        br  : numerator filter coefficient (scalar)
        ar  : denominator filter coefficients (array)
        er  : minimum error metric achieved
        F1  : estimated first formant frequency (Hz)
        BW1 : estimated first formant bandwidth (Hz)
    """
    x = np.array(x, dtype=float)
    x0 = x.copy()
    lenx = len(x)

    fAR = np.arange(200, 1110, 10)   # 200:10:1100
    BW  = np.arange(30,   60,  10)   # 30:10:50

    n_fAR = len(fAR)
    n_BW  = len(BW)

    fPole  = np.zeros((n_fAR, n_BW))
    BWPole = np.zeros((n_fAR, n_BW))
    er     = np.zeros((n_fAR, n_BW))

    # Butterworth post-IF filter (fixed, computed once)
    b_but, a_but = sig.butter(4, 2500 / (fs / 2))

    # Store SNF coefficients
    b1 = np.zeros((n_fAR, n_BW))
    a1 = np.zeros((n_fAR, n_BW, 5))   # cheby2 order-2 → denominator has 5 coeffs (2*n+1)

    i_start = int(np.floor(lenx * 0.25))
    i_end   = int(np.floor(lenx * 0.75))

    for jj, bw_val in enumerate(BW):
        for ii, f_val in enumerate(fAR):

            fPole[ii, jj]  = f_val
            BWPole[ii, jj] = bw_val

            x1, b_snf, a_snf = SNF(x0, f_val, fs, bw_val)

            b1[ii, jj]          = b_snf[0]
            a1[ii, jj, :len(a_snf)] = a_snf

            aux1 = x1[i_start:i_end].copy()

            # Post inverse-filter smoothing
            aux1 = sig.filtfilt(b_but, a_but, aux1)

            metric1    = np.abs(np.diff(np.diff(aux1)))
            er[ii, jj] = np.sum(metric1)

    # Find minimum error indices
    nx      = np.argmin(er, axis=0)          # best fAR index for each BW
    ery     = er[nx, np.arange(n_BW)]        # minimum er per BW column
    ny      = np.argmin(ery)                 # best BW index

    F1  = fPole[nx[ny],  ny]
    BW1 = BWPole[nx[ny], ny]

    br  = np.array([b1[nx[ny], ny]])
    ar  = a1[nx[ny], ny, :]
    ar  = ar[ar != 0] if np.any(ar != 0) else ar   # trim trailing zeros

    gvv, _, _ = SNF(x, F1, fs, BW1)
    er_min    = ery[ny]

    return gvv, br, ar, er_min, F1, BW1


def LP_Filter(x, fs, fc=1100, order=10):
    x = np.array(x, dtype=float)

    # Diseño del filtro Chebyshev tipo II en forma directa (coeficientes b, a)
    b, a = sig.cheby2(order, 40, fc / (fs / 2), btype='low', output='ba')

    # Aplicación del filtro con filtfilt para evitar desfase
    y = sig.filtfilt(b, a, x)

    return y,b,a


def HP_Filter(x, fs, fc=60, order=6):
    """
    High pass filter for voice analysis.

    Parameters:
        x     : array-like, input signal
        fs    : float, sampling frequency (Hz)
        fc    : float, cutoff frequency in Hz (default: 60)
        order : int, filter order (default: 6)

    Returns:
        x : filtered signal
    """
    x = np.array(x, dtype=float)

    b, a = sig.butter(order, fc / (fs / 2), btype='high')
    x    = sig.filtfilt(b, a, x)

    return x,b,a



def Detrend(tiempos,señal,orden=10):
    polinomio = Polynomial.fit(tiempos,señal, deg=orden)
    trend     = polinomio(tiempos)
    return señal - trend
def Normalizar(señal):
    minimo = np.min(señal)
    maximo = np.max(señal)
    return (señal - minimo) / (maximo - minimo)
def NormalizarDflujo(señal):
    return señal/abs(min(señal))


def INF(flujo_ovv_original, fs=8000,fmin=200,fmax=1000,step=1,Q=10):
    flujo_ovv_original = np.array(flujo_ovv_original)
    frec_formante     = -1
    error_minimo      = np.inf

    for f_i in range(fmin, fmax+1,step):
        b, a = sig.iirnotch(w0=f_i, fs=fs,Q=Q)
        flujo_i = sig.lfilter(b, a, flujo_ovv_original )

        flujo_i_ = flujo_i[round(len(flujo_i)*0.25):round(len(flujo_i)*0.75)]
        b_,a_ = sig.butter(4,2500,'low',fs=fs)
        flujo_i_ = sig.filtfilt(b_,a_,flujo_i_)
        error = np.sum(np.abs(np.gradient(np.gradient(flujo_i_))))       
        if error < error_minimo:
            error_minimo = error
            frec_formante = f_i    
    b, a = sig.iirnotch(w0=frec_formante,fs=fs, Q=Q)
    flujo = sig.lfilter(b, a, flujo_ovv_original)
    
    return flujo,frec_formante,error_minimo,b,a



def plot_freq_pc(señal,fs,label,a1,polos1,ceros1,a2,polos2,ceros2):
    fig = plt.figure(figsize=(10,3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2,1])  # 3:1 → 75% vs 25%

    ax0 = fig.add_subplot(gs[0])  # respuesta en frecuencia
    ax1 = fig.add_subplot(gs[1])  # polos y ceros

    theta = np.linspace(0, 2*np.pi, 200)

    fftseñal, fseñal = FFT(señal, fs)

    fw1,hw1 = sig.freqz([1],a1,fs=fs)
    fw2,hw2 = sig.freqz([1],a2,fs=fs)

    color1 = 'red'
    color2 = 'blue'
    color3 = 'k'

    # --- Respuesta en frecuencia ---
    ax0.plot(fw1, 10*np.log10(np.abs(hw1)), label=r'$W_1$', color=color1)
    ax0.plot(fw2, 10*np.log10(np.abs(hw2)), label=r'$W_2$', color=color2)
    ax0.plot(fseñal, 10*np.log10(np.abs(fftseñal)), label=label, color=color3, zorder=-1)
    ax0.set_xlim(0,fs/2)
    ax0.set_xticks(np.arange(0, fs/2, 100), minor=True)
    ax0.grid(True, linestyle='dashed', color='gray', alpha=0.3, which='both')
    ax0.set_xlabel('Frecuencia [Hz]')
    ax0.set_ylabel('[dB]')
    ax0.legend(loc='upper right')

    # --- Diagrama de polos y ceros ---
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=1)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")

    ax1.scatter(np.real(ceros1), np.imag(ceros1),s=80, facecolors='none', edgecolors=color1, label='Ceros')
    ax1.scatter(np.real(polos1), np.imag(polos1),s=80, marker='x', color=color1, label='Polos')

    ax1.scatter(np.real(ceros2), np.imag(ceros2),s=80, facecolors='none', edgecolors=color2, label='Ceros')
    ax1.scatter(np.real(polos2), np.imag(polos2),s=80, marker='x', color=color2, label='Polos')

    ax1.set_xticks(np.arange(-1.25,1.25,0.25), minor=True)
    ax1.set_yticks(np.arange(-1.25,1.25,0.25), minor=True)
    ax1.set_ylabel(r"$Im\{z\}$")
    ax1.set_xlabel(r"$Re\{z\}$")
    ax1.grid(True, linestyle='dashed', color='gray', alpha=0.3)


def calcular_espectrograma(señal,
                           fs=8000,
                           ventana='Rectangular',
                           duracion_ventana=5.5,
                           desplazamiento=1,
                           pre_enfasis=True,
                           doble_pre_enfasis=False,
                           p=0.55,
                           nfft=128):
    if doble_pre_enfasis:
        señal = sig.lfilter([1, -2, 1], [1], señal)
    elif pre_enfasis:
        señal = sig.lfilter([1, -1], [1], señal)
    else:
        señal = señal - np.mean(señal)

    longitud_ventana = int(np.round(fs * duracion_ventana / 1000))
    if ventana == 'Rectangular':
        ventana = sig.windows.boxcar(longitud_ventana)
    elif ventana == 'Hanning':
        ventana = sig.windows.hann(longitud_ventana)
    elif ventana == 'Hamming':
        ventana = sig.windows.hamming(longitud_ventana)
    elif ventana == 'Blackman':
        ventana = sig.windows.blackman(longitud_ventana)
    elif ventana == 'Cosine':
        ventana = sig.windows.cosine(longitud_ventana)
    elif ventana == 'Tukey':
        ventana = sig.windows.tukey(longitud_ventana)
    elif ventana == 'Triangular':
        ventana = sig.windows.triang(longitud_ventana)         

    cantidad_ventanas = int(1 + np.floor((len(señal) - longitud_ventana) / desplazamiento))
    matriz_STFT = np.zeros([nfft, cantidad_ventanas], dtype=complex)

    for i in range(cantidad_ventanas):
        inicio = i * desplazamiento
        señal_ventaneada = ventana * señal[inicio:inicio + longitud_ventana]
        matriz_STFT[:, i] = np.fft.fft(señal_ventaneada, n=nfft)

    matriz_STFT = matriz_STFT[0:nfft // 2, :]   # <-- mitad de nfft
    espectrograma = np.abs(matriz_STFT) ** p

    return matriz_STFT, espectrograma



'''
def calcular_espectrograma(señal,
                           fs=8000,
                           ventana='Rectangular',
                           duracion_ventana=5.5,
                           desplazamiento=1,
                           pre_enfasis=True,
                           doble_pre_enfasis = False,
                           p=0.55):  
    if doble_pre_enfasis:
        señal = sig.lfilter([1, -2, 1], [1], señal)
    elif pre_enfasis:
        señal = sig.lfilter([1, -1], [1], señal)
    else:
        señal = señal - np.mean(señal)

    longitud_ventana = int(np.round(fs * duracion_ventana / 1000))
    if ventana == 'Rectangular':
        ventana = sig.windows.boxcar(longitud_ventana)
    elif ventana =='Hanning':
        ventana = sig.windows.hann(longitud_ventana)
    elif ventana == 'Hamming':
        ventana = sig.windows.hamming(longitud_ventana)
    elif ventana == 'Blackman':
        ventana = sig.windows.blackman(longitud_ventana)

    cantidad_ventanas = int(1 + np.floor((len(señal) - longitud_ventana) / desplazamiento))
    matriz_STFT = np.zeros([longitud_ventana, cantidad_ventanas], dtype=complex)

    for i in range(cantidad_ventanas):
        inicio = i * desplazamiento
        desp_rel = np.arange(longitud_ventana)
        posicion_ventana = inicio + desp_rel
        señal_ventaneada = np.multiply(ventana, señal[posicion_ventana])
        matriz_STFT[:, i] = np.fft.fft(señal_ventaneada)

    matriz_STFT = matriz_STFT[0:matriz_STFT.shape[0] // 2, :]
    espectrograma = np.abs(matriz_STFT) **p

    return matriz_STFT, espectrograma
'''

def Sincronizar_por_energia(señal,señal_referencia):
    matriz_señal, _ = calcular_espectrograma(señal)
    matriz_ref, _ = calcular_espectrograma(señal_referencia)

    energia_señal= np.sum(np.abs(matriz_señal)**2, axis=0)
    energia_ref    = np.sum(np.abs(matriz_ref)**2, axis=0)



    energia_señal = energia_señal -  np.mean(energia_señal)
    energia_ref    = energia_ref - np.mean(energia_ref)
    energia_señal = energia_señal / np.max(energia_señal)
    energia_ref    = energia_ref / np.max(energia_ref)

    corr = sig.correlate(energia_señal, energia_ref, mode='full')
    lags = sig.correlation_lags(len(energia_señal),len(energia_ref),mode='full')

    lag = lags[np.argmax(np.abs(corr))]
    if lag > 0:
        # speech adelantada
        señal_ = señal[lag:]
        #tiempo_ = tiempo[lag:]
    elif lag < 0:
        lag = abs(lag)
        señal_ = señal[:len(señal_referencia)]

        #self.tiempos = self.tiempos[:len(self.speech)]
        #self.dEGG = self.dEGG[:len(self.speech)]

    return señal_



def Sincronizar3(señal_estimada, señal_ref, t_estimada, fs=8000,
                max_delay=0.01, filtrar=False, banda=(50, 3000)):
    """
    Sincroniza 'señal_estimada' respecto a 'señal_ref' y ajusta su vector de tiempo.

    Parámetros:
    - señal_estimada: señal a desplazar
    - señal_ref: señal de referencia
    - t_estimada: vector de tiempo de la señal estimada
    - fs: frecuencia de muestreo
    - max_delay: delay máximo permitido (segundos)
    - filtrar: aplicar pasa banda
    - banda: (f_low, f_high)

    Retorna:
    - señal_alineada
    - t_alineado
    - lag (en muestras)
    - delay (en segundos)
    """

    # --- 1. Copias
    x = np.copy(señal_estimada)
    y = np.copy(señal_ref)

    # --- 2. Normalización
    x = x - np.mean(x)
    y = y - np.mean(y)

    x = x / (np.std(x) + 1e-8)
    y = y / (np.std(y) + 1e-8)

    # --- 3. Filtrado opcional
    if filtrar:
        f1, f2 = banda
        b, a = sig.butter(3, [f1, f2], btype='bandpass', fs=fs)
        x = sig.filtfilt(b, a, x)
        y = sig.filtfilt(b, a, y)

    # --- 4. Correlación FFT
    corr = sig.fftconvolve(x, y[::-1], mode='full')
    lags = sig.correlation_lags(len(x), len(y), mode='full')

    # --- 5. Limitar búsqueda
    max_lag = int(max_delay * fs)
    mask = (lags >= -max_lag) & (lags <= max_lag)

    lag = lags[mask][np.argmax(corr[mask])]
    delay = lag / fs

    # --- 6. Ajustar señal y tiempo (SIN wrap-around)
    if lag > 0:
        # señal estimada está atrasada → recorto inicio
        señal_alineada = señal_estimada[lag:]
        t_alineado = t_estimada[lag:]

    elif lag < 0:
        # señal estimada está adelantada → pad al inicio
        pad = abs(lag)
        señal_alineada = np.pad(señal_estimada, (pad, 0))
        
        dt = 1/fs
        t_inicio = t_estimada[0] - pad * dt
        t_pad = np.arange(t_inicio, t_estimada[0], dt)
        t_alineado = np.concatenate((t_pad, t_estimada))

    else:
        señal_alineada = señal_estimada.copy()
        t_alineado = t_estimada.copy()

    return señal_alineada, t_alineado

def Sincronizar(señal_estimada,señal_referencia):

    corr = sig.correlate(señal_estimada,señal_referencia)
    lags = sig.correlation_lags(len(señal_estimada), len(señal_referencia), mode='full')
    lag = lags[np.argmax(corr)]

    return np.roll(señal_estimada, -lag)




def Sincronizar2(señal_estimada, señal_referencia, tiempos_referencia):
    # Correlación cruzada completa entre ambas señales.
    # corr[k] mide qué tan parecidas son las señales para cada desplazamiento
    # relativo posible entre señal_estimada y señal_referencia.
    corr = sig.correlate(señal_estimada, señal_referencia)

    # Vector de "lags" (desplazamientos en muestras) asociado a cada valor de corr.
    # lags[k] es el desplazamiento que le corresponde a corr[k].
    lags = sig.correlation_lags(len(señal_estimada), len(señal_referencia))

    # El lag que maximiza la correlación es el desplazamiento que mejor alinea
    # ambas señales: es la estimación del "delay" entre estimada y referencia.
    lag = lags[np.argmax(corr)]

    # Con la convención de scipy, la relación en el punto óptimo es:
    #     señal_estimada[n] ≈ señal_referencia[n - lag]
    # Esto significa que si lag > 0, señal_estimada tiene "lag" muestras de más
    # al principio que no tienen correspondencia en señal_referencia
    # (está retrasada respecto a la referencia).
    if lag > 0:
        # Descarto las primeras "lag" muestras de la estimada, que son las que
        # sobran (no tienen con qué matchear en la referencia).
        estimada_sync   = señal_estimada[lag:]
        # La referencia se recorta por longitud, pero se mantiene desde su
        # propio inicio (índice 0), porque es la estimada la que estaba adelantada.
        referencia_sync = señal_referencia[:len(estimada_sync)]
        # El eje temporal sigue el de la referencia, también desde el inicio,
        # porque referencia_sync no se desplazó, solo se acortó.
        tiempos_sync    = tiempos_referencia[:len(estimada_sync)]

    elif lag < 0:
        # Caso simétrico: lag < 0 implica que es la referencia la que tiene
        # muestras "de más" al principio (está adelantada respecto a la estimada).
        # -lag > 0, entonces recorto la estimada desde el final para que quede
        # del mismo largo que la referencia ya recortada (ver abajo).
        estimada_sync   = señal_estimada[:len(señal_referencia) + lag]
        # Descarto las primeras "-lag" muestras de la referencia, que son las
        # que sobran (no tienen correspondencia en la estimada).
        referencia_sync = señal_referencia[-lag:]
        # El eje temporal se recorta de la misma forma que la referencia,
        # ya que es ella la que se desplazó.
        tiempos_sync    = tiempos_referencia[-lag:]

    else:
        # lag == 0: ya están alineadas, no hace falta recortar nada.
        estimada_sync   = señal_estimada
        referencia_sync = señal_referencia
        tiempos_sync    = tiempos_referencia

    # Ajuste final de seguridad: por cómo se calculan los índices en cada rama,
    # puede haber un desfasaje de 1 muestra entre los largos resultantes.
    # Se recorta todo al mínimo común para garantizar que las tres salidas
    # (estimada, referencia, tiempos) tengan exactamente el mismo largo.
    min_len = min(len(estimada_sync), len(referencia_sync))
    return estimada_sync[:min_len], referencia_sync[:min_len], tiempos_sync[:min_len]

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
