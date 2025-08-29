import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import dynamsignal as ds

def fft_spectrum(signal, dt, scaling='rms'):
    n = len(signal)
    n2 = n // 2
    h = hann(n)
    df = 1 / (n * dt)
    freq = np.linspace(0, (n2 - 1) * df, n2)
    mag = ds.utilities.fft_spectrum(signal,scaling)
    return freq, mag

def plotar_4graficos(x_h, dt_h, x_v, dt_v, x_a, dt_a, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    sinais = {
        'Horizontal': (x_h, dt_h),
        'Vertical': (x_v, dt_v),
        'Axial': (x_a, dt_a)
    }

    fig, axs = plt.subplots(2, 2, figsize=(10, 4)) 

    # --- Gráfico 1: Sinal no Tempo ---
    ax1 = axs[0, 0]
    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            t = np.arange(0, len(sinal) * dt, dt)
            ax1.plot(t, sinal, label=direcao)
    ax1.set_xlabel('Tempo [s]')
    ax1.set_ylabel('Amplitude [g]')
    ax1.set_title('Tempo')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid()

    # --- Gráfico 2: Espectro da Aceleração ---
    ax2 = axs[0, 1]
    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            freq, dep = fft_spectrum(sinal, dt, scaling='peak')
            ax2.plot(freq, dep, label=direcao)
    ax2.set_xlabel('Frequência [Hz]')
    ax2.set_ylabel('Amplitude [g]')
    ax2.set_title('Aceleração')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid()

    # --- Gráfico 3: Espectro da Velocidade ---
    ax3 = axs[1, 0]
    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            vi = 10000 * ds.utilities.time_integration(sinal, dt)
            freq, dep_v = fft_spectrum(vi, dt, scaling='rms')
            ax3.plot(freq, dep_v, label=direcao)
    ax3.set_xlabel('Frequência [Hz]')
    ax3.set_ylabel('Amplitude [mm/s]')
    ax3.set_title('Velocidade')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid()

    # --- Gráfico 4: Envelope da Aceleração ---
    ax4 = axs[1, 1]
    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            env = ds.timeparameters.envelope(sinal, fc=500, dt=dt)
            freq, dep_env = fft_spectrum(env, dt, scaling='peak-peak')
            ax4.plot(freq, dep_env, label=direcao)
    ax4.set_xlabel('Frequência [Hz]')
    ax4.set_ylabel('Amplitude [gE]')
    ax4.set_title('Envelope')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid()

    # --- Ajuste de layout para garantir que não corte legendas ---
    fig.tight_layout(pad=2.0, h_pad=2.0, w_pad=1.5)

    return fig

def plotar_graficos(xi, x, dt, title = None, subtitle = None, vel_scal = 'rms', acc_scal = 'peak', env_scal = 'peak-peak'):
    #   scaling : 'peak' , 'peak-peak' , 'rms'  
    npto = int(len(x))
    npto2 = int(npto/2)
    df = 1/(dt*npto)
    freq = np.linspace(0, (npto2-1)*df, npto2)
    t = np.arange(0,npto*dt,dt)

    # Velocidade a partir do sinal de entrada (xi) e do sinal limpo (x)
    x_limpo = ds.utilities.frequency_filter(x, 10, dt, type='highpass')
    v = 10000 * ds.utilities.time_integration(x_limpo, dt)
    xi_limpo = ds.utilities.frequency_filter(xi, 10, dt, type='highpass')
    vi = 10000 * ds.utilities.time_integration(xi_limpo, dt)    

    # DEP (aceleração)
    
    dep_xi = fft_spectrum(xi, dt, scaling=acc_scal)[1]
    dep_x = fft_spectrum(x, dt, scaling=acc_scal)[1]
 
    dep_vi = fft_spectrum(vi, dt, scaling=vel_scal)[1]
    i10 = int(10 / df)
    dep_vi[0:i10] = 0
    dep_v = fft_spectrum(v, dt, scaling=vel_scal)[1]
    dep_v[0:i10] = 0
    # Envelope do sinal de entrada e do sinal limpo (aceleração)
    x_env = np.copy(x) 
    aux_env = ds.timeparameters.envelope(x_env, fc=500, dt=dt)
    dep_env_x = fft_spectrum(aux_env, dt, scaling=env_scal)[1]
    dep_env_x[0:i10] = 0

    x_env = np.copy(xi) 
    aux_env = ds.timeparameters.envelope(x_env, fc=500, dt=dt)
    dep_env_xi = fft_spectrum(aux_env, dt, scaling=env_scal)[1]
    dep_env_xi[0:i10] = 0


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    # Plot aceleração: xi e x
    if subtitle is not None : 
        txt = '%s'%(subtitle[0:10])
    else : 
        txt = '  '    
    #ax1.plot(t, xi, label='Referência')
    #ax1.plot(t, x, label= txt, alpha=0.75)
    ax1.plot(t, x)
    ax1.plot(t,xi)
    ax1.set_xlabel('Tempo [s]')
    ax1.set_ylabel('Aceleração [g]')
    ax1.set_title( 'Aceleração - Data: %s'%(txt))
    ax1.legend(['Sinal Ajust','Sinal Base'])
    ax1.grid()

    # Plot DEP aceleração: xi e x
    ax2.plot(freq, dep_x)
    ax2.plot(freq,dep_xi)
    #ax2.plot(freq, dep_xf_limpo, label=txt, alpha=0.75)
    ax2.set_xlabel('Frequência [Hz]')
    ax2.set_ylabel('Aceleração Pico [g]')
    ax2.set_title('Espectro da Aceleração - Data: %s'%(txt))
    ax2.legend(['Sinal Ajust','Sinal Base'])
    ax2.grid()

    # Plot DEP: depi_vi e depi_vlimpo
    ax3.plot(freq, dep_vi, label='Referência')
    ax3.plot(freq, dep_v, label= txt, alpha=0.75)
    ax3.set_xlabel('Frequência [Hz]')
    ax3.set_ylabel('Velocidade RMS [mm/s]')
    ax3.set_title('Espectro da Velocidade - Data: %s'%(txt))
    ax3.legend(['Sinal Ajust','Sinal Base'])
    ax3.grid()

    # Plot Envelope: dep_env_xi e dep_env_x
    #ax4.plot(freq, dep_env_xi, label='Referência')
    ax4.plot(freq, dep_env_x)
    ax4.plot(freq,dep_env_xi)
    ax4.set_xlabel('Frequência [Hz]')
    ax4.set_ylabel('Envelope Pico-Pico [gE]')
    ax4.set_title('Envelope da Aceleração - Data: %s'%(txt))
    ax4.legend()
    ax4.legend(['Sinal Ajust','Sinal Base'])
    txt = 'Análise do Sinal' 
    if title is not None : 
        txt = title 
    else : 
        txt = '   '    
    plt.suptitle(txt, size=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

def plotar_acel(x_h, dt_h, x_v, dt_v, x_a, dt_a, title=None):
    sinais = {
        'Horizontal': (x_h, dt_h),
        'Vertical': (x_v, dt_v),
        'Axial': (x_a, dt_a)
    }

    fig, axs = plt.subplots(1, 1, figsize=(8, 4)) 
    ax2 = axs 

    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            freq, dep = fft_spectrum(sinal, dt, scaling='peak')
            ax2.plot(freq, dep, label=direcao)

    ax2.set_xlabel('Frequência [Hz]')
    ax2.set_ylabel('Amplitude [g]')
    #ax2.set_title('Espectro da Aceleração' if title is None else title)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(True)

    fig.tight_layout()
    return fig


def plotar_vel(x_h, dt_h, x_v, dt_v, x_a, dt_a, title=None):
    sinais = {
        'Horizontal': (x_h, dt_h),
        'Vertical': (x_v, dt_v),
        'Axial': (x_a, dt_a)
    }

    fig, axs = plt.subplots(1, 1, figsize=(8, 4)) 
    ax3 = axs  

    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            vi = 10000 * ds.utilities.time_integration(sinal, dt)
            freq, dep_v = fft_spectrum(vi, dt, scaling='rms')
            ax3.plot(freq, dep_v, label=direcao)
    ax3.set_xlabel('Frequência [Hz]')
    ax3.set_ylabel('Amplitude [mm/s]')
    #ax3.set_title('Espectro da Velocidade')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True)

    fig.tight_layout()
    return fig

def plotar_env(x_h, dt_h, x_v, dt_v, x_a, dt_a, title=None):
    sinais = {
        'Horizontal': (x_h, dt_h),
        'Vertical': (x_v, dt_v),
        'Axial': (x_a, dt_a)
    }
    fig, axs = plt.subplots(1, 1, figsize=(8, 4)) 
    ax4 = axs 

    for direcao, (sinal, dt) in sinais.items():
        if sinal is not None:
            env = ds.timeparameters.envelope(sinal, fc=500, dt=dt)
            freq, dep_env = fft_spectrum(env, dt, scaling='peak-peak')
            ax4.plot(freq, dep_env, label=direcao)
    ax4.set_xlabel('Frequência [Hz]')
    ax4.set_ylabel('Amplitude [gE]')
    #ax4.set_title('Envelope da Aceleração')
    ax4.legend(fontsize=7, loc='upper right')
    ax4.grid()

    fig.tight_layout()

    return fig