from scipy.spatial.distance import cdist
import compChem
class molecule:
    def __init__(self, input_file):
        """
        sets up the object

        input:
        input_file: fath to a file containing the molecular coords
        """
        self.atoms, self.coords = compChem.data_input.readData(input_file)
        self.distanceMatrix = cdist(self.coords, self.coords)

    def displayBondAngles(self):
        """
        dislpays all bond angles, further docs in the calculations.py file
        """
        return compChem.calculations.allBondAngles()
    
    
water = molecule("../bootcamp/projects/harmonic-vibrational-analysis/input/water.txt")