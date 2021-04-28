import psi4
import numpy as np
from scipy.linalg import eigh
numpy_memory = 4
psi4.set_memory(int(5e8))
from compChem.Hartree_Fock_modes import rhf
class CISMolecule():
    def __init__(self, molecule):
        """
        Will set up some variables we will need from the Molecule object

        input
        molecule: a Molecule object from the compChem package
        constraint: True if you want a CUHF wavefunction
        """
        self.id = molecule
        self.occupied = self.id.alpha + self.id.beta
        self.available = self.id.integrals.nbf()*2
        self.virtual = self.available - self.occupied
        self.E_0 = molecule.E_0
    

    def getTwoElectronIntegrals(self, exchange=True):
        """returns two electron integrals in MO basis"""
        # getting the two electron integrals in correct basis => we need it in MO basis
        tei = self.id.elrep # given in chemists notation

        #change the basis of the tei
        tei_int = np.kron(np.eye(2), tei)
        tei_big = np.kron(np.eye(2), tei_int.T)

        C = self.C
        if exchange:
            tei_ao = tei_big.transpose(0, 2, 1, 3) - tei_big.transpose(0, 2, 3, 1) # accounts for both coulomb and exchange, switch to physisists notation
        else:
            tei_ao = tei_big.transpose(0, 2, 1, 3)
        tei_mo = np.einsum("pQRS,pP->PQRS", np.einsum("pqRS,qQ->pQRS", np.einsum("pqrS,rR->pqRS", np.einsum("pqrs,sS->pqrS", tei_ao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)
        
        return tei_mo
    

    def displayCISHamiltonian(self):
        """displays the CIS hamiltonian in MO basis"""
        # getting the orbital energies
        if str(type(self.id)) == "<class 'compChem.Hartree_Fock_modes.rhf.RHFMolecule'>": 
            epsilon_a, C_a = self.id.getEigenStuff()
            epsilon_b, C_b = self.id.getEigenStuff()
        else:
            epsilon_a, C_a = self.id.getEigenStuff("alpha")
            epsilon_b, C_b = self.id.getEigenStuff("beta")
        epsilon = np.append(epsilon_a, epsilon_b) # accounts for the fact that the energies might be different
        sortedorder = np.argsort(epsilon)
        epsilon = np.sort(epsilon)
        self.epsilon = epsilon
        
        # make the C matrix => it contains all the orbitals, both alpha and beta
        C = np.block([[C_a, np.zeros(C_a.shape)], [np.zeros(C_b.shape), C_b]]) # accounts for the fact that the orbitals might be different (uhf, cuhf)
        C = C[:, sortedorder] # sort the eigenfunctions
        self.C = C

        tei_mo = self.getTwoElectronIntegrals()
        
        #getting the excitations
        excitations = []
        for orbital in range(self.occupied): # for every occupied orbital
            for another_orbital in range(self.occupied, self.available): # we can make an excitation to every virtual orbital
                excitations.append((orbital, another_orbital))
        self.excitations = excitations
        # getting the hamiltonian
        dim = self.occupied*self.virtual
        H_cis = np.zeros((dim, dim))
        for row, excitation in enumerate(excitations):
            i, a = excitation
            for collumn, another_excitation in enumerate(excitations):
                j, b = another_excitation   
                H_cis[row, collumn] = (epsilon[a] - epsilon[i])*(i == j)*(a == b) + tei_mo[a, j, i, b]
        

        return H_cis


    def CalculateExcitations(self):
        """setting up some properties needed for later"""
        ham = self.displayCISHamiltonian()
        self.excitation_energies, self.coefs = eigh(ham)


    def GetExitations(self, filepath="NoNameGiven"):
        """Get the excitation energies and the contributions"""
        if filepath == "NoNameGiven":
            raise ValueError("no path specified")
        from pathlib import Path
        Path(f"{filepath}").touch()
        datafile = open(f"{filepath}", "w")
        self.CalculateExcitations(alternate=alternate)
        contrib = self.coefs**2
        energies = self.excitation_energies
        counterdict = {} # added to check how many times a certain excitation occurs
        datafile.writelines(f"scf energy for {self.mode}: {self.E_0}\n")
        for state, energy in enumerate(energies):
            datafile.writelines(f" {state + 1} : {energy}\n")
            for excitation, contribution in enumerate(contrib[:, state]):
                if contribution*100 > 1:
                    datafile.writelines(f"\t{contribution:.3f} : {self.excitations[excitation]}\n")
                    if self.excitations[excitation] not in counterdict:
                        counterdict[self.excitations[excitation]] = 0
                    counterdict[self.excitations[excitation]] += 1   
        datafile.close()
    
    
    def getCISEnergy(self, orbital=0):
        """
        will calculate the CIS energy
        
        input:
        orbital: you can enter the orbital from which you want the energy, default the lowest energy excitation
        """
        A = 0
        for number, excitation in enumerate(self.excitations):
            i, a = excitation
            A += (self.epsilon[a] - self.epsilon[i])*self.coefs[number, orbital]**2

        B = 0
        tei_mo = self.getTwoElectronIntegrals()
        for number, excitation in enumerate(self.excitations):
            i, a = excitation
            for another_number, another_excitation, in enumerate(self.excitations):
                j, b = another_excitation
                B += self.coefs[number, orbital]*self.coefs[another_number, orbital]*tei_mo[a, j, i, b]
        return A + B