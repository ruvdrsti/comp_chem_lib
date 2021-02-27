import numpy as np
def readData(filename):
    """
    Reads files
    
    input:
    filename: the path to the file you want opened

    output:
    tuple of arrays containing atomic numbers and atomic coords
    """
    rawdata = np.loadtxt(filename, skiprows=1)
    atomic_numbers = rawdata[:, 0]
    atomic_coords = rawdata[:, 1:]
    return atomic_numbers, atomic_coords