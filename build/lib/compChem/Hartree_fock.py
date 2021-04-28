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

        #setting up the inegrals
        self.nuc_rep = self.id.nuclear_repulsion_energy()
        self.overlap = self.integrals.ao_overlap().np
        self.kin = self.integrals.ao_kinetic().np
        self.pot = self.integrals.ao_potential().np
        self.elrep = self.integrals.ao_eri().np

        self.guessMatrix_a = self.kin + self.pot
        self.guessMatrix_b = self.kin + self.pot
        if change:
            self.alpha = change[0]
            self.beta = change[1]
        
        self.E_0 = 0
        
        # defining convergence via user interactions
        self.converge = 1e-6

        

    
    def setGuess(self, new_guess=None, spin=None):
        """
        sets the guessMatrix to a new value
        
        input:
        new_guess: numpy array that represents a new fock matrix
        spin: a string, either "alpha" or "beta"

        note:
        this method is kept as is, since both uhf and cuhf will require separate guessmatrices. Rhf will not, but will only use one of the guessmatrix fields.
        """
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

    
    def setConvergence(self, new_convergence):
        """ sets the convergence to desired value"""
        self.converge = new_convergence


class RHFMolecule(Molecule):
    """
    Will extend the backbone to work for RHF
    
    input:
    the geometry you want to make a molecule out of
    """
    def __init__(self, geometry):
        super().__init__(geometry)
        self.mode = "rhf"

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



    def iterator(self, criterion='density', iteration=5000, mute=False, Force=False):
        """
        Function that performs the Hartree-Fock iterative calculations for the given molecule.
        
        input:
        criterion: "energy" or "density", sets the criterion that we want to evaluate. Default "density"
        iteration: maximum amount of iterations allowed. Default 5000
        Force: force all elements below value 1e-10 to become a true zero

        note:
        the molecule needs to have its guessmatrices set before entering
        """
        assert criterion == "energy" or criterion == "density", f" {criterion}: not a valid criterion"
        # setting up entry parameters for the while loop
        E_old = self.E_0
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
            if Force:
                F_a[abs(F_a) < 1e-10] = 0
            self.setGuess(F_a, "alpha") # see doctring setGuess method
            d_new = self.getDensityMatrix()

            # comparing block: will answer the "Are we there yet?" question
            rms_D = np.sqrt(np.einsum("pq->", (d_old - d_new)**2, optimize=True))
            if criterion == "density":
                if rms_D < self.converge:
                    convergence = True
            else:
                if abs(E_old - E_total) < self.converge:
                    convergence = True


            # maintenance block: keeps everything going
            if not mute:
                print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D: .8f}")
            E_old = E_total
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


class UHFMolecule(Molecule):
    """
    Will extend the backbone to work for RHF

    input:
    geometry: the geometry to make a molecule out of
    """
    def __init__(self, geometry):
        super().__init__(geometry)
        self.mode = "uhf"

    
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


    def getDensityMatrix(self, spin, mixedGuess=True):
        """
        generates the densitiy matrices on the MO level, D_alpha, D_beta
        
        input:
        spin: a string, either "alpha" or "beta"
        mixedGuess: False if you do not want to use a mixed guess
        """
        assert spin == "alpha" or spin == "beta", f"{spin}: no valid spin"
        if spin == "alpha":
            occ = self.alpha
            guess = self.guessMatrix_a
        else:
            occ = self.beta
            guess = self.guessMatrix_b
        C = self.getEigenStuff(spin)[1]
        if np.all(guess == self.displayHamiltonian()) and self.beta and mixedGuess:
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


    def displayFockMatrix(self, spin, mixedGuess=True):
        """
        Will display the Fock matrix
        
        input:
        spin: a string, either "alpha" or "beta"
        """
        coulomb_a = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix("alpha", mixedGuess=mixedGuess), optimize=True)
        coulomb_b = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix("beta", mixedGuess=mixedGuess), optimize=True)
        exchange = np.einsum("npoq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(spin, mixedGuess=mixedGuess), optimize=True)
        F = self.displayHamiltonian() + coulomb_a + coulomb_b - exchange
        return F


    def getElectronicEnergy(self, mixedGuess=True):
        """
        calculates the energy with the current fock matrix
        """
        sumMatrix_alpha = self.displayHamiltonian() + self.guessMatrix_a
        E_alpha = 0.5*np.einsum("pq,pq->", sumMatrix_alpha, self.getDensityMatrix("alpha", mixedGuess=mixedGuess), optimize=True)
        sumMatrix_beta = self.displayHamiltonian() + self.guessMatrix_b
        E_beta = 0.5*np.einsum("pq,pq->", sumMatrix_beta, self.getDensityMatrix("beta", mixedGuess=mixedGuess), optimize=True)
        return E_alpha + E_beta 


    def getTotalEnergy(self, mixedGuess=True):
        """
        Calculates the total energy
        """
        return self.getElectronicEnergy(mixedGuess=mixedGuess) + self.displayNucRep()



    def iterator(self, criterion='density', iteration=5000, mute=False, mixedGuess=True, Force=False):
        """
        Function that performs the Hartree-Fock iterative calculations for the given molecule.
        
        input:
        criterion: "energy" or "density", sets the criterion that we want to evaluate. Default "density"
        iteration: maximum amount of iterations allowed. Default 5000
        mute: False if you want to see all the iterations
        Force: force all elements below value 1e-10 to become a true zero
        
        note:
        the molecule needs to have its guessmatrices set before entering
        """
        assert criterion == "energy" or criterion == "density", f" {criterion}: not a valid criterion"
        # setting up entry parameters for the while loop
        E_old = self.E_0
        d_old_alpha = self.getDensityMatrix("alpha", mixedGuess=mixedGuess)
        d_old_beta = self.getDensityMatrix("beta", mixedGuess=mixedGuess)
        convergence = False

        # step 2: start iterating
        itercount = 0
        while not convergence and itercount < iteration:

            # calculating block: calculates energies
            E_new = self.getElectronicEnergy()
            E_total = self.getTotalEnergy(mixedGuess=mixedGuess)

            # generating block: generates new matrices UHF: account for alpha and beta
            F_a = self.displayFockMatrix("alpha", mixedGuess=mixedGuess)
            F_b = self.displayFockMatrix("beta", mixedGuess=mixedGuess)
            if Force:
                F_a[abs(F_a) < 1e-10] = 0
                F_b[abs(F_b) < 1e-10] = 0

            self.setGuess(F_a, "alpha")
            self.setGuess(F_b, "beta") 

            d_new_alpha = self.getDensityMatrix("alpha", mixedGuess=mixedGuess)
            d_new_beta = self.getDensityMatrix("beta",mixedGuess=mixedGuess)

            # comparing block: will answer the "Are we there yet?" question
            rms_D_a = np.sqrt(np.einsum("pq->", (d_old_alpha - d_new_alpha)**2, optimize=True))
            rms_D_b = np.sqrt(np.einsum("pq->", (d_old_beta - d_new_beta)**2, optimize=True))
            if criterion == "density":
                if rms_D_a < self.converge and rms_D_b < self.converge:
                    convergence = True
            else:
                if abs(E_old - E_total) < self.converge:
                    convergence = True


            # maintenance block: keeps everything going
            if not mute:
                print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D_a: .8f}, {rms_D_b : .8f}")
            E_old = E_new
            d_old_alpha = d_new_alpha
            d_old_beta = d_new_beta
            itercount += 1
        
        self.E_0 = E_total
        return E_total, itercount


    def getSpinContamination(self, mixedGuess=True):
        """Will display the spin contamination

        input:
        mixedGuess: False if you do not want to use a mixed guess
        """
        # transform p to MO basis, where mo basis = the eigenfunctions of the f_a operator
        a = self.getDensityMatrix("alpha", mixedGuess=True)
        b = self.getDensityMatrix("beta", mixedGuess=True)
        f_a, f_b = self.displayFockMatrix("alpha", mixedGuess=True), self.displayFockMatrix("beta",mixedGuess=True)
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


class CUHFMolecule(Molecule):
    """
    Will extend the backbone to work for cuhf

    input:
    geometry: the geometry you want to make a molecule out of
    """
    def __init__(self, geometry):
        super().__init__(geometry)
        self.mode = "cuhf"

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


    def getDensityMatrix(self, spin, mixedGuess=True):
        """
        generates the densitiy matrices on the MO level, D_alpha, D_beta
        
        input:
        spin: a string, either "alpha" or "beta"
        mixedGuess: False if you do not want to use a mixed guess
        """
        assert spin == "alpha" or spin == "beta", f"{spin}: no valid spin"
        if spin == "alpha":
            occ = self.alpha
            guess = self.guessMatrix_a
        else:
            occ = self.beta
            guess = self.guessMatrix_b
        C = self.getEigenStuff(spin)[1]
        if np.all(guess == self.displayHamiltonian()) and self.beta and mixedGuess:
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


    def displayFockMatrix(self, spin, mixedGuess=True):
        """
        Will display the Fock matrix
        
        input:
        spin: a string, either "alpha" or "beta"
        """
        coulomb_a = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix("alpha", mixedGuess=mixedGuess), optimize=True)
        coulomb_b = np.einsum("nopq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix("beta", mixedGuess=mixedGuess), optimize=True)
        exchange = np.einsum("npoq,pq->no", self.displayElectronRepulsion(), self.getDensityMatrix(spin, mixedGuess=mixedGuess), optimize=True)
        F = self.displayHamiltonian() + coulomb_a + coulomb_b - exchange
        return F


    def getElectronicEnergy(self, mixedGuess=True):
        """
        calculates the energy with the current fock matrix
        """
        sumMatrix_alpha = self.displayHamiltonian() + self.guessMatrix_a
        E_alpha = 0.5*np.einsum("pq,pq->", sumMatrix_alpha, self.getDensityMatrix("alpha", mixedGuess=mixedGuess), optimize=True)
        sumMatrix_beta = self.displayHamiltonian() + self.guessMatrix_b
        E_beta = 0.5*np.einsum("pq,pq->", sumMatrix_beta, self.getDensityMatrix("beta", mixedGuess=mixedGuess), optimize=True)
        return E_alpha + E_beta 


    def getTotalEnergy(self, mixedGuess=True):
        """
        Calculates the total energy
        """
        return self.getElectronicEnergy(mixedGuess=mixedGuess) + self.displayNucRep()



    def iterator(self, criterion='density', iteration=5000, mute=False, mixedGuess=True, Force=False):
        """
        Function that performs the Hartree-Fock iterative calculations for the given molecule.
        
        input:
        criterion: "energy" or "density", sets the criterion that we want to evaluate. Default "density"
        iteration: maximum amount of iterations allowed. Default 5000
        mixedGuess: False if you do not want to use a mixed guess
        Force: force all elements below value 1e-10 to become a true zero

        note:
        the molecule needs to have its guessmatrices set before entering
        """
        assert criterion == "energy" or criterion == "density", f" {criterion}: not a valid criterion"
        # setting up entry parameters for the while loop
        E_old = self.E_0
        d_old_alpha = self.getDensityMatrix("alpha", mixedGuess=mixedGuess)
        d_old_beta = self.getDensityMatrix("beta", mixedGuess=mixedGuess)
        convergence = False

        # step 2: start iterating
        itercount = 0
        while not convergence and itercount < iteration:

            # calculating block: calculates energies
            E_new = self.getElectronicEnergy()
            E_total = self.getTotalEnergy()

            # generating block: generates new matrices UHF: account for alpha and beta
            F_a, F_b = self.basischanger(mixedGuess=mixedGuess)

            if Force:
                F_a[abs(F_a) < 1e-10] = 0
                F_b[abs(F_b) < 1e-10] = 0
            self.setGuess(F_a, "alpha")
            self.setGuess(F_b, "beta") 

            d_new_alpha = self.getDensityMatrix("alpha", mixedGuess=mixedGuess)
            d_new_beta = self.getDensityMatrix("beta", mixedGuess=mixedGuess)

            # comparing block: will answer the "Are we there yet?" question
            rms_D_a = np.sqrt(np.einsum("pq->", (d_old_alpha - d_new_alpha)**2, optimize=True))
            rms_D_b = np.sqrt(np.einsum("pq->", (d_old_beta - d_new_beta)**2, optimize=True))
            if criterion == "density":
                if rms_D_a < self.converge and rms_D_b < self.converge:
                    convergence = True
            else:
                if abs(E_old - E_total) < self.converge:
                    convergence = True


            # maintenance block: keeps everything going
            if not mute:
                print(f"iteration: {itercount}, E_tot: {E_total: .8f}, E_elek: {E_new: .8f}, deltaE: {E_new - E_old: .8f}, rmsD: {rms_D_a: .8f}")
            E_old = E_total
            d_old_alpha = d_new_alpha
            d_old_beta = d_new_beta
            itercount += 1
        
        self.E_0 = E_total
        return E_total, itercount

    
    def setConvergence(self, new_convergence):
        """ sets the convergence to desired value"""
        self.converge = new_convergence
    
    
    def basischanger(self, mixedGuess=True):
        """
        changes to NO basis, applies CUHF constraint, then changes back
        """
        # transform p to MO basis, where mo basis = the eigenfunctions of the f_a operator
        a = self.getDensityMatrix("alpha", mixedGuess=mixedGuess)
        b = self.getDensityMatrix("beta", mixedGuess=mixedGuess)
        f_a, f_b = self.displayFockMatrix("alpha", mixedGuess=mixedGuess), self.displayFockMatrix("beta",mixedGuess=mixedGuess)
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

    def getSpinContamination(self, mixedGuess=True):
        """Will display the spin contamination

        input:
        mixedGuess: False if you do not want to use a mixed guess
        """
        # transform p to MO basis, where mo basis = the eigenfunctions of the f_a operator
        a = self.getDensityMatrix("alpha", mixedGuess=True)
        b = self.getDensityMatrix("beta", mixedGuess=True)
        f_a, f_b = self.displayFockMatrix("alpha", mixedGuess=True), self.displayFockMatrix("beta",mixedGuess=True)
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