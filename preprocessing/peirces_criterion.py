#!/usr/bin/env python3
import numpy as np
import scipy.special
from functools import reduce

def peirce_dev(N: int, n: int = 1, m: int = 1) -> float:
    """Peirce's criterion
    
    Returns the squared threshold error deviation for outlier identification
    using Peirce's criterion based on Gould's methodology.
    
    Arguments:
        - int, total number of observations (N)
        - int, number of outliers to be removed (n)
        - int, number of model unknowns (m)
    Returns:
        float, squared error threshold (x2)
    """
    # Assign floats to input variables:
    N = float(N)
    n = float(n)
    m = float(m)

    # Check number of observations:
    if N > 1:
        # Calculate Q (Nth root of Gould's equation B):
        Q = (n ** (n / N) * (N - n) ** ((N - n) / N)) / N
        #
        # Initialize R values (as floats)
        r_new = 1.0
        r_old = 0.0  # <- Necessary to prompt while loop
        #
        # Start iteration to converge on R:
        while abs(r_new - r_old) > (N * 2.0e-16):
            # Calculate Lamda
            # (1/(N-n)th root of Gould's equation A'):
            ldiv = r_new ** n
            if ldiv == 0:
                ldiv = 1.0e-6
            Lamda = ((Q ** N) / (ldiv)) ** (1.0 / (N - n))
            # Calculate x-squared (Gould's equation C):
            x2 = 1.0 + (N - m - n) / n * (1.0 - Lamda ** 2.0)
            # If x2 goes negative, return 0:
            if x2 < 0:
                x2 = 0.0
                r_old = r_new
            else:
                # Use x-squared to update R (Gould's equation D):
                r_old = r_new
                r_new = np.exp((x2 - 1) / 2.0) * scipy.special.erfc(
                    np.sqrt(x2) / np.sqrt(2.0)
                )
    else:
        x2 = 0.0
    return x2

def stats(values):
    n = len(values)
    sum = values.sum()
    avg = sum / n
    var = np.var(values)
    std = np.std(values)
    return {"n": n, "sum": sum,"avg": avg,"var": var,"std": std}





def separate_outliers(v):
    result = None
    s = stats(v)
    nbrRemoved = 0
    k = None
    

    while True:
        k = nbrRemoved + 1
        r = np.sqrt(peirce_dev(s['n'], k))
        max = r * s['std']
        def outlierReduce(r, x):
            if np.abs(x - s['avg'] < max):
                r['trimmed'].append(x)
            else:
                r['outliers'].append(x)
            return r

        initial = {"original": v, 'trimmed': [], 'outliers': []}
        result = reduce(outlierReduce, v, initial)
        
        #DO STUFF
        if nbrRemoved <= k:
            break
    return result