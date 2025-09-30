import numpy as np


def stencilOptParams(vmin, vmax, f, h, g):
    """
    STENCILOPTPARAMS - Optimal Params for 9-Point Stencil
    INPUTS:
        vmin = minimum wave velocity [L/T]
        vmax = maximum wave velocity [L/T]
        f = frequency [1/T]
        h = grid spacing in X [L]
        g = (grid spacing in Y [L])/(grid spacing in X [L])
    OUTPUTS:
        b, d, e = optimal params according to Chen/Cheng/Feng/Wu 2013 Paper
    """

    # Define Grid and Compute Gmin, Gmax
    l, r = 100, 10
    Gmin, Gmax = vmin / (f * h), vmax / (f * h)

    # Generate Theta and G Values
    m = np.arange(1, l + 1)
    n = np.arange(1, r + 1)
    theta = (m - 1) * np.pi / (4 * (l - 1))
    G = 1 / (1 / Gmax + ((n - 1) / (r - 1)) * (1 / Gmin - 1 / Gmax))

    # Create Meshgrid
    TH, GG = np.meshgrid(theta, G, indexing='ij')

    # Compute P and Q
    P = np.cos(g * 2 * np.pi * np.cos(TH) / GG)
    Q = np.cos(2 * np.pi * np.sin(TH) / GG)

    # Compute S1, S2, S3, S4 Matrices
    S1 = (1 + 1 / (g ** 2)) * (GG ** 2) * (1 - P - Q + P * Q)
    S2 = (np.pi ** 2) * (2 - P - Q)
    S3 = (2 * np.pi ** 2) * (1 - P * Q)
    S4 = 2 * np.pi ** 2 + (GG ** 2) * ((1 + 1 / (g ** 2)) * P * Q - P - Q / (g ** 2))

    # Fix `b` Value or Compute It
    fixB = True
    if fixB:
        b = 5 / 6  # Fix the Value to 5/6 based on Laplacian Derived by Robert E. Lynch
        A = np.column_stack((S2.ravel(), S3.ravel()))
        y = S4.ravel() - b * S1.ravel()
        params = np.linalg.lstsq(A, y, rcond=None)[0]  # Least Squares Solution
        d, e = params[0], params[1]
    else:
        A = np.column_stack((S1.ravel(), S2.ravel(), S3.ravel()))
        y = S4.ravel()
        params = np.linalg.lstsq(A, y, rcond=None)[0]
        b, d, e = params[0], params[1], params[2]

    return b, d, e
