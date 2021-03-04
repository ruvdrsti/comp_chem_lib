# setting up psi4 options
import psi4
import numpy as np
import scipy.linalg as sp
psi4.set_output_file("output.dat", True)  # setting output file
psi4.set_memory(int(5e8))
numpy_memory = 2
psi4.set_options({'basis': 'STO-3G'})


class molecule:
    def __init__(self, geom_file):
        """
        sets up the molecule object
        
        input:
        geom_file: a link to a pubchem file    
            
        note:
        This class is designed to work in an iterative Restricted Closed Shell HF calculation. The guess matrix needs to be
        updated asap. This will always correspond to the curren fock-matrix.
        """
        if """pubchem""" in geom_file:
            self.id = psi4.geometry(geom_file)
        else:
            self.id = psi4.geometry(f"""
            {geom_file}
        
            units bohr
            """)
        self.id.update_geometry()
        self.wfn =  psi4.core.Wavefunction.build(self.id, psi4.core.get_global_option('basis'))
        self.basis = self.wfn.basisset()
        self.integrals = psi4.core.MintsHelper(self.basis)
        self.occupied = self.wfn.nalpha()  # only works for closed shell systems
        self.guessMatrix = "empty"
    
    
    def setGuess(self, new_guess):
        """
        sets the guessMatrix to a new value
        
        input:
        new_guess: numpy array that represents a new fock matrix
        """
        self.guessMatrix = new_guess


    def displayNucRep(self):
        """
        Will calculate the nuclear repulsion
        """
        return self.id.nuclear_repulsion_energy()


    def displayOverlap(self):
        """
        Will display the overlap matrix as np array
        """
        return self.integrals.ao_overlap().np


    def displayE_kin(self):
        """
        Will display kinetic energy as np array
        """
        return self.integrals.ao_kinetic().np


    def displayE_pot(self):
        """
        Will display the kinetic energy as np array
        """
        return self.integrals.ao_potential().np


    def displayHamiltonian(self):
        """
        Will display the hamiltonian as a np array
        """
        return self.displayE_kin() + self.displayE_pot()


    def displayElectronRepulsion(self):
        """
        Will display the interelectronic repulsion as a np array (4D array)
        """
        return self.integrals.ao_eri().np


    def transformToUnity(self):
        """
        Gives the matrix that will transform S into I_n
        
        note:
        functions return dimension objects, do not use equality
        """
        transformMatrix = self.integrals.ao_overlap()
        transformMatrix.power(-0.5, 1e-16)
        return transformMatrix.np


    def getEigenStuff(self):
        """
        calculates the eigenvectors and eigenvalues of the hamiltonian
        """
        return sp.eigh(self.guessMatrix, b=self.displayOverlap())


    def getDensityMatrix(self):
        """
        generates the densitiy matrix on the MO level
        """
        C = self.getEigenStuff()[1]
        A = 2*np.einsum("pq, qr->pr", C[:, :self.occupied], C[:, :self.occupied].T, optimize=True)
        return A


    def displayFockMatrix(self):
        """Will display the Fock matrix"""
        summation1 = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(), optimize=True)
        summation2 = np.einsum("npoq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(), optimize=True)
        self.fockMatrix = self.displayHamiltonian() + summation1 - 0.5*summation2
        return self.fockMatrix


    def getElectronicEnergy(self):
        """
        calculates the energy with the current fock matrix
        """
        sumMatrix = self.displayHamiltonian() + self.displayFockMatrix()
        return 0.5*np.einsum("pq,pq->", sumMatrix, self.getDensityMatrix())


    def getTotalEnergy(self):
        """
        Calculates the total energy
        """
        return self.getElectronicEnergy() + self.displayNucRep()


def iterator(target_molecule):
    """
    Function that performs the Hartree-Fock iterative calculations for the given molecule.
    
    input:
    target_molecule: a molecule object from the class molecule
    """
    # setting up entry parameters for the while loop
    E_new = 0  
    E_old = 0
    d_old = target_molecule.getDensityMatrix()
    convergence = False

    # step 2: start iterating
    itercount = 0
    while not convergence and itercount < 50:

        # calculating block: calculates energies
        E_new = target_molecule.getElectronicEnergy()
        E_total = target_molecule.getTotalEnergy()

        # generating block: generates new matrices
        F_n =  target_molecule.displayFockMatrix()
        target_molecule.setGuess(F_n)
        d_new = target_molecule.getDensityMatrix()

        # comparing block: will answer the "Are we there yet?" question
        rms_D = np.einsum("pq->", np.sqrt((d_old - d_new)**2), optimize=True)
        if abs(E_old - E_new) < 1e-6 and rms_D < 1e-4:
            convergence = True


        # maintenance block: keeps everything going
        print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D: .8f}")
        E_old = E_new
        d_old = d_new
        itercount += 1