import numpy as np

def ringingRemovalFilt(xin, yin, img_in, c0, f, cutoff, ord):
    """
    Filter to Remove Ringing from Waveform Inversion Result

    Parameters:
    - xin: x (N-element array) grid [in m] for the input image
    - yin: y (M-element array) grid [in m] for the input image
    - img_in: input image (M x N array)
    - c0: reference sound speed [m/s]
    - f: frequency [Hz]
    - cutoff: ranging from 0 (only keep DC) to 1 (up to ringing frequency)
    - ord: order of radial Butterworth filter cutoff (np.inf for sharp cutoff)

    Returns:
    - img_out: output image (M x N array)
    """
    # Number and spacing of input points (assumes a uniform spacing)
    Nxin = len(xin)
    dxin = np.mean(np.diff(xin))
    Nyin = len(yin)
    dyin = np.mean(np.diff(yin))

    # K-Space (frequency domain)
    kxin = np.fft.fftshift((np.arange(Nxin) / Nxin) / dxin)
    kxin[kxin >= 1 / (2 * dxin)] -= 1 / dxin

    kyin = np.fft.fftshift((np.arange(Nyin) / Nyin) / dyin)
    kyin[kyin >= 1 / (2 * dyin)] -= 1 / dyin

    Kxin, Kyin = np.meshgrid(kxin, kyin)

    # Cutoff Wavenumber
    k = 2 * f / c0
    kcutoff = k * cutoff

    # Radial Butterworth Filter
    radialFilt = 1 / (1 + ((Kxin**2 + Kyin**2) / (kcutoff**2))**ord)

    # Apply Filter
    img_fft = np.fft.fft2(img_in)               # FFT
    img_fft_shifted = np.fft.fftshift(img_fft)  # Shift zero frequency to center
    img_filtered = radialFilt * img_fft_shifted # Apply filter
    img_ifft_shifted = np.fft.ifftshift(img_filtered)  # Inverse shift
    img_out = np.real(np.fft.ifft2(img_ifft_shifted))  # Inverse FFT and take real part

    return img_out