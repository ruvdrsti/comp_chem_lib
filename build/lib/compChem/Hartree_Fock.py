import psi4
import numpy as np
import scipy.linalg as sp
psi4.set_output_file("output.dat", True)  # setting output file
psi4.set_memory(int(5e8))
numpy_memory = 2
psi4.set_options({'basis': 'cc-pvdz', 'reference': 'uhf', 'scf_type': 'df'})
class Molecule:
    def __init__(self, geom_file):
        """
        sets up the molecule object
        
        input:
        geom_file: a link to a pubchem file    
            
        note:
        This class is designed to work in an iterative HF calculation. The guess matrix needs to be
        updated asap. This will always correspond to the current fock-matrix.
        """
        if """pubchem""" in geom_file:
            self.id = psi4.geometry(geom_file)
        else:
            self.id = psi4.geometry(f"""
            {geom_file}""")
        self.id.update_geometry()
        self.wfn =  psi4.core.Wavefunction.build(self.id, psi4.core.get_global_option('basis'))
        self.basis = self.wfn.basisset()
        self.integrals = psi4.core.MintsHelper(self.basis)
        self.alpha = self.wfn.nalpha()
        self.beta = self.wfn.nbeta()
        # only works for closed shell systems
        self.guessMatrix_a = "empty"
        self.guessMatrix_b = "empty"
        
        
        #setting up the inegrals
        self.nuc_rep = self.id.nuclear_repulsion_energy()
        self.overlap = self.integrals.ao_overlap().np
        self.kin = self.integrals.ao_kinetic().np
        self.pot = self.integrals.ao_potential().np
        self.elrep = self.integrals.ao_eri().np

        # defining convergence via user interactions
        self.converge = 1e-6
        

    
    def setGuess(self, new_guess=None, spin=None):
        """
        sets the guessMatrix to a new value
        
        input:
        new_guess: numpy array that represents a new fock matrix
        spin: a string, either "alpha" or "beta"
        """
        if self.guessMatrix_a == "empty" and self.guessMatrix_b == "empty":
            self.guessMatrix_a = self.displayHamiltonian()
            self.guessMatrix_b = self.displayHamiltonian()
        else:
            assert spin == "alpha" or spin == "beta", f"{spin}: no valid spin"
            if spin == "alpha":
                self.guessMatrix_a = new_guess
            else:
                self.guessMatrix_b = new_guess


    def displayNucRep(self):
        """
        Will calculate the nuclear repulsion
        """

        return self.nuc_rep


    def displayOverlap(self):
        """
        Will display the overlap matrix as np array
        """
        return self.overlap

    def displayE_kin(self):
        """
        Will display kinetic energy as np array
        """
        return self.kin


    def displayE_pot(self):
        """
        Will display the kinetic energy as np array
        """
        return self.pot


    def displayHamiltonian(self):
        """
        Will display the hamiltonian as a np array
        """
        return self.displayE_kin() + self.displayE_pot()


    def displayElectronRepulsion(self):
        """
        Will display the interelectronic repulsion as a np array (4D array)
        """
        return self.elrep


    def transformToUnity(self):
        """
        Gives the matrix that will transform S into I_n
        
        note:
        functions return dimension objects, do not use equality
        """
        transformMatrix = self.integrals.ao_overlap()
        transformMatrix.power(-0.5, 1e-16)
        return transformMatrix.np


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
                HOMO_LUMO = C[occ-1:occ+1]
                HOMO = HOMO_LUMO[0]
                LUMO = HOMO_LUMO[1]
                HOMO_LUMO[0] += k*LUMO
                HOMO_LUMO[1] += -k*HOMO
                HOMO_LUMO *= 1/np.sqrt(2)
                
                C[occ-1:occ+1] = HOMO_LUMO
            
        
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



    def iterator(self, criterion='density', iteration=500):
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
            F_a =  self.displayFockMatrix("alpha")
            self.setGuess(F_a, "alpha")
            F_b = self.displayFockMatrix("beta")
            self.setGuess(F_b, "beta") 
            d_new_alpha = self.getDensityMatrix("alpha")
            d_new_beta = self.getDensityMatrix("beta")

            # comparing block: will answer the "Are we there yet?" question
            rms_D_a = np.einsum("pq->", np.sqrt((d_old_alpha - d_new_alpha)**2), optimize=True)
            rms_D_b = np.einsum("pq->", np.sqrt((d_old_beta - d_new_beta)**2), optimize=True)
            if criterion == "density":
                if rms_D_a < self.converge and rms_D_b < self.converge:
                    convergence = True
            else:
                if abs(E_old - E_new) < self.converge:
                    convergence = True


            # maintenance block: keeps everything going
            print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D_a: .8f}")
            E_old = E_new
            d_old_alpha = d_new_alpha
            d_old_beta = d_new_beta
            itercount += 1
        
        return E_total

    
    def setConvergence(self, new_convergence):
        """ sets the convergence to desired value"""
        self.converge = new_convergence
    
        