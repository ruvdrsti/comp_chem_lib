from scipy.spatial.distance import cdist
from compChem import data_input
from compChem import calculations
class molecule:
    def __init__(self, input_file):
        """
        sets up the object

        input:
        input_file: fath to a file containing the molecular coords
        """
        self.atoms, self.coords = data_input.readData(input_file)
        self.distanceMatrix = cdist(self.coords, self.coords)

    def displayBondAngles(self):
        """
        dislpays all bond angles, further docs in the calculations.py file
        """
        distanceMatrix = self.distanceMatrix
        return calculations.allBondAngles(self.coords)
    
    
water = molecule("../bootcamp/projects/harmonic-vibrational-analysis/input/water.txt")
print(water.displayBondAngles())