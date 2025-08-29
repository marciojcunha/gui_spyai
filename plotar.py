import numpy as np
import dynamsignal as ds
import matplotlib.pyplot as plt 
from scipy.signal.windows import hann

def fft_spectrum(x, scaling = 'peak'  ) : 
    #   x : np  float array 
    #   scaling : 'peak' , 'peak-peak' , 'rms'  
    npto = len(x)
    X = np.abs(np.fft.fft(2*hann(npto)*x/npto))[0:int(npto/2)]
    if scaling == 'peak' : 
        return 2*X 
    if scaling == 'peak-peak' : 
        return 4*X 
    return np.sqrt(2)*X


def plotar_graficos(xi, x,dt, title = None):
    npto = int(len(x))
    npto2 = int(npto/2)
    df = 1/(dt*npto)
    freq = np.linspace(0, (npto2-1)*df, npto2)
    t = np.arange(0,npto*dt,dt)

    # Velocidade a partir do sinal de entrada (xi) e do sinal limpo (x)
    xf = ds.utilities.frequency_filter(xi, 10, dt, type='highpass')
    vi = 10000 * ds.utilities.time_integration(xf, dt)

    xf_limpo = ds.utilities.frequency_filter(x, 10, dt, type='highpass')
    v_limpo = 10000 * ds.utilities.time_integration(xf_limpo, dt)

    # DEP (aceleração)
    jan = 2*hann(npto)/npto
    xf = np.copy(xi) #ds.utilities.frequency_filter(xi, 10, dt, type='highpass')
    aux_xf = np.fft.fft(xf*jan )
    dep_xf = np.sqrt(2)*(np.abs(aux_xf[0:npto2]))

    xf_limpo = np.copy(x) #ds.utilities.frequency_filter(x, 10, dt, type='highpass')
    aux_xf_limpo = np.fft.fft(x * jan)
    dep_xf_limpo = np.sqrt(2)*(np.abs(aux_xf_limpo[0:npto2]))

    # DEP do sinal de entrada e do sinal limpo (velocidade)
    aux_vi = np.fft.fft(vi * jan)
    depi_vi = np.sqrt(2)*(np.abs(aux_vi[0:npto2]))
    i10 = int(10 / df)
    depi_vi[0:i10] = 0

    aux_vlimpo = np.fft.fft(v_limpo * jan)
    depi_vlimpo = np.sqrt(2)*(np.abs(aux_vlimpo[0:npto2]))
    depi_vlimpo[0:i10] = 0

    # Envelope do sinal de entrada e do sinal limpo (aceleração)
    aux_env_xi = ds.timeparameters.envelope(xi, fc=500, dt=dt)
    aux_env_xi = np.fft.fft(2*aux_env_xi * hann(npto))
    dep_env_xi = np.sqrt(2)*(np.sqrt(np.abs(aux_env_xi[0:npto2])))
    dep_env_xi[0:i10] = 0

    aux_env_x = ds.timeparameters.envelope(x, fc=500, dt=dt)
    aux_env_x = np.fft.fft(2*aux_env_x * hann(npto))
    dep_env_x = np.sqrt(2)*(np.sqrt(np.abs(aux_env_x[0:npto2])))
    dep_env_x[0:i10] = 0

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Plot aceleração: xi e x
    ax1.plot(t, xi, label='Sinal Entrada (xi)')
    ax1.plot(t, x, label='Sinal Limpo (x)', alpha=0.75)
    ax1.set_xlabel('Tempo [s]')
    ax1.set_ylabel('Aceleração [g]')
    ax1.set_title('Tempo')
    ax1.legend()
    ax1.grid()

    # Plot DEP aceleração: xi e x
    ax2.plot(freq, dep_xf, label='DEP Entrada (xi)')
    ax2.plot(freq, dep_xf_limpo, label='DEP Limpa (x_limpo)', alpha=0.75)
    ax2.set_xlabel('Frequência [Hz]')
    ax2.set_ylabel('Aceleração [g]')
    ax2.set_title('DEP - Aceleração')
    ax2.legend()
    ax2.grid()

    # Plot DEP: depi_vi e depi_vlimpo
    ax3.plot(freq, depi_vi, label='DEP Entrada (vi)')
    ax3.plot(freq, depi_vlimpo, label='DEP Limpa (v_limpo)', alpha=0.75)
    ax3.set_xlabel('Frequência [Hz]')
    ax3.set_ylabel('DEP [mm/s]')
    ax3.set_title('DEP - Velocidade')
    ax3.legend()
    ax3.grid()

    # Plot Envelope: dep_env_xi e dep_env_x
    ax4.plot(freq, dep_env_xi, label='Envelope Entrada (xi)')
    ax4.plot(freq, dep_env_x, label='Envelope Limpo (x)', alpha=0.75)
    ax4.set_xlabel('Frequência [Hz]')
    ax4.set_ylabel('Envelope [gE]')
    ax4.set_title('Envelope')
    ax4.legend()
    ax4.grid()
    txt = 'Análise do Sinal' 
    if title is not None : 
        txt = title 
    plt.suptitle(txt, size=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plotar_graficos_lucas(xi, x,dt, title = None,
                    subtitle = None, vel_scal = 'rms', acc_scal = 'peak', env_scal = 'peak-peak'):
    #   scaling : 'peak' , 'peak-peak' , 'rms'  
    npto = int(len(x))
    npto2 = int(npto/2)
    df = 1/(dt*npto)
    freq = np.linspace(0, (npto2-1)*df, npto2)
    t = np.arange(0,npto*dt,dt)

    # Velocidade a partir do sinal de entrada (xi) e do sinal limpo (x)
    xf = ds.utilities.frequency_filter(xi, 10, dt, type='highpass')
    vi = 10000 * ds.utilities.time_integration(xf, dt)

    xf_limpo = ds.utilities.frequency_filter(x, 10, dt, type='highpass')
    v_limpo = 10000 * ds.utilities.time_integration(xf_limpo, dt)

    # DEP (aceleração)
    
    xf = np.copy(xi) #ds.utilities.frequency_filter(xi, 10, dt, type='highpass')
    dep_xf = fft_spectrum(xf, scaling=acc_scal)
    #aux_xf = np.fft.fft(xf*jan)
    #dep_xf = np.sqrt(2)*(np.abs(aux_xf[0:npto2]))

    xf_limpo = np.copy(x) #ds.utilities.frequency_filter(x, 10, dt, type='highpass')
    dep_xf_limpo = fft_spectrum(xf_limpo, scaling=acc_scal)
    #aux_xf_limpo = np.fft.fft(x * jan)
    #dep_xf_limpo = np.sqrt(2)*(np.abs(aux_xf_limpo[0:npto2]))

    # DEP do sinal de entrada e do sinal limpo (velocidade)
    #aux_vi = np.fft.fft(vi * jan)
    #depi_vi = np.sqrt(2)*(np.abs(aux_vi[0:npto2]))
    depi_vi = fft_spectrum(vi, scaling=vel_scal)
    i10 = int(10 / df)
    depi_vi[0:i10] = 0

    depi_vlimpo = fft_spectrum(v_limpo, scaling=vel_scal)
    #aux_vlimpo = np.fft.fft(v_limpo * jan)
    #depi_vlimpo = np.sqrt(2)*(np.abs(aux_vlimpo[0:npto2]))
    depi_vlimpo[0:i10] = 0

    # Envelope do sinal de entrada e do sinal limpo (aceleração)
    xenvi = np.copy(xi) 
    aux_env_xi = ds.timeparameters.envelope(xenvi, fc=500, dt=dt)
    dep_env_xi = fft_spectrum(aux_env_xi, scaling=env_scal)
    # aux_env_xi = np.fft.fft(2*aux_env_xi * hann(npto))
    # dep_env_xi = np.sqrt(2)*(np.sqrt(np.abs(aux_env_xi[0:npto2])))
    dep_env_xi[0:i10] = 0

    xenv = np.copy(x) 
    aux_env_x = ds.timeparameters.envelope(xenv, fc=500, dt=dt)
    dep_env_x = fft_spectrum(aux_env_xi, scaling=env_scal)
    # aux_env_x = np.fft.fft(2*aux_env_x * hann(npto))
    # dep_env_x = np.sqrt(2)*(np.sqrt(np.abs(aux_env_x[0:npto2])))
    dep_env_x[0:i10] = 0

    # plt.figure()
    # plt.plot(dep_env_x)
    # plt.show(block=False)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    # Plot aceleração: xi e x
    if subtitle is not None : 
        txt = 'Medido em %s'%(subtitle[0:10])
    else : 
        txt = 'Sinal analisado'    
    #ax1.plot(t, xi, label='Referência')
    #ax1.plot(t, x, label= txt, alpha=0.75)
    ax1.plot(t, x)
    ax1.set_xlabel('Tempo [s]')
    ax1.set_ylabel('Aceleração [g]')
    ax1.set_title( 'Aceleração - Data: %s'%(subtitle[0:10]))
    ax1.legend()
    ax1.grid()

    # Plot DEP aceleração: xi e x
    ax2.plot(freq, dep_xf)
    #ax2.plot(freq, dep_xf_limpo, label=txt, alpha=0.75)
    ax2.set_xlabel('Frequência [Hz]')
    ax2.set_ylabel('Aceleração Pico [g]')
    ax2.set_title('Espectro da Aceleração - Data: %s'%(subtitle[0:10]))
    ax2.legend()
    ax2.grid()

    # Plot DEP: depi_vi e depi_vlimpo
    ax3.plot(freq, depi_vi, label='Referência')
    ax3.plot(freq, depi_vlimpo, label= txt, alpha=0.75)
    ax3.set_xlabel('Frequência [Hz]')
    ax3.set_ylabel('Velocidade RMS [mm/s]')
    ax3.set_title('Espectro da Velocidade - Data: %s'%(subtitle[0:10]))
    ax3.legend()
    ax3.grid()

    # Plot Envelope: dep_env_xi e dep_env_x
    #ax4.plot(freq, dep_env_xi, label='Referência')
    ax4.plot(freq, dep_env_x)
    ax4.set_xlabel('Frequência [Hz]')
    ax4.set_ylabel('Envelope Pico-Pico [gE]')
    ax4.set_title('Envelope da Aceleração - Data: %s'%(subtitle[0:10]))
    ax4.legend()
    ax4.grid()
    txt = 'Análise do Sinal' 
    if title is not None : 
        txt = title 
    plt.suptitle(txt, size=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()    