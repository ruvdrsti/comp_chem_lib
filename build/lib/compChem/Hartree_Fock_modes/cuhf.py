import psi4
import numpy as np
import scipy.linalg as sp
psi4.set_output_file("output.dat", True)  # setting output file
psi4.set_memory(int(5e8))
numpy_memory = 2
from hf_backbone import Molecule
class CUHFMolecule(Molecule):
    """
    Will extend the backbone to work for cuhf

    input:
    geometry: the geometry you want to make a molecule out of
    """
    def __init__(self, geometry):
        super().__init__(geometry)

    def getEigenStuff(self, spin):
        """
        calculates the eigenvectors and eigenvalues of the hamiltonian
        input:
        spin: a string, either "alpha" or "beta"
        """
        if spin == "alpha":
            F = self.guessMatrix_a
        else:
            F = self.guessMatrix_b        
        return sp.eigh(F, b=self.displayOverlap())


    def getDensityMatrix(self, spin):
        """
        generates the densitiy matrices on the MO level, D_alpha, D_beta
        
        input:
        spin: a string, either "alpha" or "beta"
        """
        assert spin == "alpha" or spin == "beta", f"{spin}: no valid spin"
        if spin == "alpha":
            occ = self.alpha
            guess = self.guessMatrix_a
        else:
            occ = self.beta
            guess = self.guessMatrix_b
        C = self.getEigenStuff(spin)[1]
        if np.all(guess == self.displayHamiltonian()):
            if spin == "beta":
                k = 1
                HOMO_LUMO = C[:, occ-1:occ+1].copy()
                HOMO = HOMO_LUMO[:, 0].copy()
                LUMO = HOMO_LUMO[:, 1].copy()
                HOMO_LUMO[:, 0] += k*LUMO
                HOMO_LUMO[:, 1] += -k*HOMO
                HOMO_LUMO *= 1/np.sqrt(2)
                
                C[:, occ-1:occ+1] = HOMO_LUMO
            
        
        D = np.einsum("pq, qr->pr", C[:, :occ], C[:, :occ].T, optimize=True)
        return D


    def displayFockMatrix(self, spin):
        """
        Will display the Fock matrix
        
        input:
        spin: a string, either "alpha" or "beta"
        """
        coulomb_a = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix("alpha"), optimize=True)
        coulomb_b = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix("beta"), optimize=True)
        exchange = np.einsum("npoq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(spin), optimize=True)
        F = self.displayHamiltonian() + coulomb_a + coulomb_b - exchange
        return F


    def getElectronicEnergy(self):
        """
        calculates the energy with the current fock matrix
        """
        sumMatrix_alpha = self.displayHamiltonian() + self.guessMatrix_a
        E_alpha = 0.5*np.einsum("pq,pq->", sumMatrix_alpha, self.getDensityMatrix("alpha"), optimize=True)
        sumMatrix_beta = self.displayHamiltonian() + self.guessMatrix_b
        E_beta = 0.5*np.einsum("pq,pq->", sumMatrix_beta, self.getDensityMatrix("beta"), optimize=True)
        return E_alpha + E_beta 


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
        assert self.guessMatrix_a != "empty" and self.guessMatrix_b != "empty", "make a guess first"
        assert criterion == "energy" or criterion == "density", f" {criterion}: not a valid criterion"
        # setting up entry parameters for the while loop
        E_new = 0  
        E_old = 0
        d_old_alpha = self.getDensityMatrix("alpha")
        d_old_beta = self.getDensityMatrix("beta")
        convergence = False

        # step 2: start iterating
        itercount = 0
        while not convergence and itercount < iteration:

            # calculating block: calculates energies
            E_new = self.getElectronicEnergy()
            E_total = self.getTotalEnergy()

            # generating block: generates new matrices UHF: account for alpha and beta
            F_a, F_b = self.basischanger()
            self.setGuess(F_a, "alpha")
            self.setGuess(F_b, "beta") 
            d_new_alpha = self.getDensityMatrix("alpha")
            d_new_beta = self.getDensityMatrix("beta")

            # comparing block: will answer the "Are we there yet?" question
            rms_D_a = np.sqrt(np.einsum("pq->", (d_old_alpha - d_new_alpha)**2, optimize=True))
            rms_D_b = np.sqrt(np.einsum("pq->", (d_old_beta - d_new_beta)**2, optimize=True))
            if criterion == "density":
                if rms_D_a < self.converge and rms_D_b < self.converge:
                    convergence = True
            else:
                if abs(E_old - E_new) < self.converge:
                    convergence = True


            # maintenance block: keeps everything going
            if not mute:
                print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D_a: .8f}")
            E_old = E_new
            d_old_alpha = d_new_alpha
            d_old_beta = d_new_beta
            itercount += 1
        
        self.E_0 = E_total
        return E_total, itercount

    
    def setConvergence(self, new_convergence):
        """ sets the convergence to desired value"""
        self.converge = new_convergence
    
    
    def basischanger(self):
        """
        changes to NO basis, applies CUHF constraint, then changes back
        """
        # transform p to MO basis, where mo basis = the eigenfunctions of the f_a operator
        a = self.getDensityMatrix("alpha")
        b = self.getDensityMatrix("beta")
        f_a, f_b = self.displayFockMatrix("alpha"), self.displayFockMatrix("beta")
        p = (a+b)/2
        c = sp.eigh(f_a, self.overlap)[1] # we only need the c matrix, not the eigenvalues themselves,
        
        delta = (f_b-f_a)/2
        f_cs = (f_a+f_b)/2
        # pay attention, c matrices are not unitary
        c_inv = np.linalg.inv(c) # we need the inverse for later
        p_trans = np.einsum("pq, qr, rs->ps", c_inv, p, c_inv.T, optimize=True)
        delta_trans = np.einsum("pq, qr, rs->ps", c.T, delta, c, optimize=True)
        
        
         # transform the fock matrices to NO basis
        d = sp.eigh(p_trans)[1]
        d = d[:, ::-1] #invert all collumns
    
        d_inv = np.linalg.inv(d)
        delta_no = np.einsum("pq, qr, rs->ps", d_inv, delta_trans, d_inv.T, optimize=True)
        
        alpha = self.alpha 
        beta = self.beta
        #alter first blocks
        
        delta_no[alpha:, :beta] = 0
        delta_no[:beta, alpha:] = 0
        delta_ao = np.einsum("pq, qr, rs->ps",c_inv.T.dot(d), delta_no, d.T.dot(c_inv), optimize=True)
        
        
        return f_cs - delta_ao, f_cs + delta_ao