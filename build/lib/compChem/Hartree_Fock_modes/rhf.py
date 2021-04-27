import psi4
import numpy as np
import scipy.linalg as sp
psi4.set_output_file("output.dat", True)  # setting output file
psi4.set_memory(int(5e8))
numpy_memory = 2
from compChem.Hartree_Fock_modes.hf_backbone import Molecule
class RHFMolecule(Molecule):
    """
    Will extend the backbone to work for RHF
    
    input:
    the geometry you want to make a molecule out of
    """
    def __init__(self, geometry):
        super().__init__(geometry)

    def getEigenStuff(self):
        """
        calculates the eigenvectors and eigenvalues of the hamiltonian
        """
        F = self.guessMatrix_a
        return sp.eigh(F, b=self.displayOverlap())


    def getDensityMatrix(self):
        """
        generates the densitiy matrix
        """
        C = self.getEigenStuff()[1]
        D = np.einsum("pq, qr->pr", C[:, :self.alpha], C[:, :self.alpha].T, optimize=True)
        return D


    def displayFockMatrix(self):
        """
        Will display the Fock matrix
        """
        coulomb = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(), optimize=True)
        exchange = np.einsum("npoq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(), optimize=True)
        F = self.displayHamiltonian() + 2*coulomb - exchange
        return F


    def getElectronicEnergy(self):
        """
        calculates the energy with the current fock matrix
        """
        sumMatrix = self.displayHamiltonian() + self.guessMatrix_a
        E = np.einsum("pq,pq->", sumMatrix, self.getDensityMatrix(), optimize=True)
        return E 


    def getTotalEnergy(self):
        """
        Calculates the total energy
        """
        return self.getElectronicEnergy() + self.displayNucRep()



    def iterator(self, criterion='density', iteration=5000, mute=False):
        """
        Function that performs the Hartree-Fock iterative calculations for the given molecule.
        
        input:
        criterion: "energy" or "density", sets the criterion that we want to evaluate. Default "density"
        iteration: maximum amount of iterations allowed. Default 500
        
        note:
        the molecule needs to have its guessmatrices set before entering
        """
        assert criterion == "energy" or criterion == "density", f" {criterion}: not a valid criterion"
        # setting up entry parameters for the while loop
        E_new = 0  
        E_old = 0
        d_old = self.getDensityMatrix()
        convergence = False

        # step 2: start iterating
        itercount = 0
        while not convergence and itercount < iteration:

            # calculating block: calculates energies
            E_new = self.getElectronicEnergy()
            E_total = self.getTotalEnergy()

            # generating block: generates new matrices UHF: account for alpha and beta
            F_a = self.displayFockMatrix()
            self.setGuess(F_a, "alpha") # see doctring setGuess method
            d_new = self.getDensityMatrix()

            # comparing block: will answer the "Are we there yet?" question
            rms_D = np.sqrt(np.einsum("pq->", (d_old - d_new)**2, optimize=True))
            if criterion == "density":
                if rms_D < self.converge:
                    convergence = True
            else:
                if abs(E_old - E_new) < self.converge:
                    convergence = True


            # maintenance block: keeps everything going
            if not mute:
                print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D: .8f}")
            E_old = E_new
            d_old= d_new
            itercount += 1
        
        self.E_0 = E_total
        return E_total, itercount


    def getSpinContamination(self, mixedGuess=True):
        """Will display the spin contamination

        input:
        mixedGuess: False if you do not want to use a mixed guess
        """
        # transform p to MO basis, where mo basis = the eigenfunctions of the f_a operator
        a = self.getDensityMatrix()
        b = self.getDensityMatrix()
        f_a, f_b = self.displayFockMatrix(), self.displayFockMatrix()
        p = (a+b)/2
        m = (a-b)/2
        c = sp.eigh(f_a, self.overlap)[1] # we only need the c matrix, not the eigenvalues themselves,


        # pay attention, c matrices are not unitary
        c_inv = np.linalg.inv(c) # we need the inverse for later
        p_trans = np.einsum("pq, qr, rs->ps", c_inv, p, c_inv.T, optimize=True)
        m_trans = np.einsum("pq, qr, rs->ps", c_inv, m, c_inv.T, optimize=True)


        # transform the fock matrices to NO basis
        d = sp.eigh(p_trans)[1]
        d = d[:, ::-1] #invert all collumns

        d_inv = np.linalg.inv(d)
        m_no = np.einsum("pq, qr, rs->ps", d_inv, m_trans, d_inv.T, optimize=True)

        return self.beta - (self.alpha + self.beta)/2 + 2*np.trace(m_no.dot(m_no))


