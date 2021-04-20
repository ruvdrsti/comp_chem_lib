import psi4
import numpy as np
import scipy.linalg as sp
psi4.set_output_file("output.dat", True)  # setting output file
psi4.set_memory(int(5e8))
numpy_memory = 2
class Molecule:
    def __init__(self, geom_file, change=False):
        """
        sets up the molecule object
        
        input:
        geom_file: a link to a pubchem file  
        change: tuple that contains a custom electron arrangement, of the shape (n_a, n_b), allows for custom electron configurations if needed.  
            
        note:
        The molecule object will be the backbone of the rhf, uhf, and cuhf. These classes will make use of this class.
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
        if change:
            self.alpha = change[0]
            self.beta = change[1]
        
        
        
        
        #setting up the inegrals
        self.nuc_rep = self.id.nuclear_repulsion_energy()
        self.overlap = self.integrals.ao_overlap().np
        self.kin = self.integrals.ao_kinetic().np
        self.pot = self.integrals.ao_potential().np
        self.elrep = self.integrals.ao_eri().np

        # defining convergence via user interactions
        self.converge = 1e-6

        self.setGuess()
        

    
    def setGuess(self, new_guess=None, spin=None):
        """
        sets the guessMatrix to a new value
        
        input:
        new_guess: numpy array that represents a new fock matrix
        spin: a string, either "alpha" or "beta"

        note:
        this method is kept as is, since both uhf and cuhf will require separate guessmatrices. Rhf will not, but will only use one of the guessmatrix fields.
        """
        if self.guessMatrix_a == "empty" and self.guessMatrix_b == "empty":
            self.guessMatrix_a = self.displayHamiltonian()
            self.guessMatrix_b = self.displayHamiltonian()
        else:
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

    
    def setConvergence(self, new_convergence):
        """ sets the convergence to desired value"""
        self.converge = new_convergence