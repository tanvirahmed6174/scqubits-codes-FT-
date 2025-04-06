# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:48:54 2024

@author: WANGLAB
"""

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
# from qutip.qip.operations import rz, cz_gate
# import cmath




#Variables

Jc = 0.025
g_a = 0.05
g_b = 0.05



# define fluxonium A
qbta = scq.Fluxonium(
    EC=1,
    EJ=4.43,
    EL=.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=20,
)

qbtb = scq.Transmon(
     EJ=10.875,
     EC=0.3,
     ng=0,
     ncut=110,
     truncated_dim=20)
qbta_ro = scq.Oscillator(E_osc=8, truncated_dim=5)
qbtb_ro = scq.Oscillator(E_osc=8, truncated_dim=5)
# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta,qbta_ro, qbtb,qbtb_ro])


hilbertspace.add_interaction(g=g_a, op1=qbta.n_operator, op2=qbta_ro.creation_operator, add_hc=True)
hilbertspace.add_interaction(g=g_b, op1=qbtb.n_operator, op2=qbtb_ro.creation_operator, add_hc=True)
# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=Jc,
    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)

# generate spectrum lookup table
hilbertspace.generate_lookup()

# Hamiltonian in dressed eigenbasis
(evals,) = hilbertspace["evals"]
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation = 60

# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])



#%%

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


EC = 1.08
EL = 0.8
EJ = 6
n_g_con = 2*(2*EC/EL)**.25
# g_a = 0.04*n_g_con/2.5
g_a = 0.03*n_g_con/2
# define fluxonium A
qbtA = scq.Fluxonium(
    EJ=EJ,
    EC=EC,
    EL=EL,
    flux=0.50,  # flux frustration point
    cutoff=110,
    truncated_dim=20,
)
qbtA_ro = scq.Oscillator(E_osc=7.27, truncated_dim=5)
# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbtA, qbtA_ro])

bare_states_A = qbtA.eigenvals()-qbtA.eigenvals()[0]


hilbertspace.add_interaction(g=g_a, op1=qbtA.n_operator, op2=qbtA_ro.creation_operator, add_hc=True)
# generate spectrum Lookup table
hilbertspace.generate_lookup()
# Hamiltonian in dressed eigenbasis
(evals,) = hilbertspace["evals"]
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation = 60
# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])


diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
evalues = (diag_dressed_hamiltonian_trunc.eigenenergies()-diag_dressed_hamiltonian_trunc.eigenenergies()[0])/6.28

e_11 = evalues[hilbertspace.dressed_index((1,1))]
e_10 = evalues[hilbertspace.dressed_index((1,0))]
e_01 = evalues[hilbertspace.dressed_index((0,1))]
e_00 = evalues[hilbertspace.dressed_index((0,0))]
# chi = hilbertspace.dressed_index((1,1))-hilbertspace.dressed_index((1,0))-hilbertspace.dressed_index((0,1))+hilbertspace.dressed_index((1,1,1))
# print("chi (MHz):",chi,chi_1)

chi= e_11-e_10-e_01+e_00
print("chi (MHz):",chi*1000)
#%% Chi vs Ro freq

import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Define fluxonium parameters
EC = 1.08
EL = 0.7
EJ = 5
n_g_con = 2 * (2 * EC / EL) ** 0.25
g_a = 0.03 * n_g_con / 2

# Sweep range for resonator frequency
resonator_freqs = np.linspace(5,8, 50)  # GHz
chi_values = []

# Loop over resonator frequencies
for E_osc in resonator_freqs:
    # Define fluxonium qubit
    qbtA = scq.Fluxonium(
        EJ=EJ,
        EC=EC,
        EL=EL,
        flux=0.50,  # flux frustration point
        cutoff=110,
        truncated_dim=20,
    )
    
    # Define resonator
    qbtA_ro = scq.Oscillator(E_osc=E_osc, truncated_dim=5)
    bare_states_A = qbtA.eigenvals()-qbtA.eigenvals()[0]
    # Define Hilbert space
    hilbertspace = scq.HilbertSpace([qbtA, qbtA_ro])
    
    # Compute bare states
    hilbertspace.add_interaction(g=g_a, op1=qbtA.n_operator, op2=qbtA_ro.creation_operator, add_hc=True)
    
    # Generate spectrum lookup table
    hilbertspace.generate_lookup()
    
    # Extract eigenvalues
    evals = hilbertspace["evals"][0]
    diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)
    )
    
    # Truncate Hamiltonian
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])
    
    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28
    
    # Compute chi
    e_11 = evalues[hilbertspace.dressed_index((1,1))]
    e_10 = evalues[hilbertspace.dressed_index((1,0))]
    e_01 = evalues[hilbertspace.dressed_index((0,1))]
    e_00 = evalues[hilbertspace.dressed_index((0,0))]
    
    chi = e_11 - e_10 - e_01 + e_00
    chi_values.append(chi * 1000)  # Convert to MHz

# Plot chi vs. resonator frequency
plt.figure(figsize=(8, 5))
plt.plot(resonator_freqs, chi_values, marker='o', linestyle='-', color='b')
plt.xlabel("Resonator Frequency (GHz)")
plt.ylabel("Chi (MHz)")
plt.title("Chi vs. Resonator Frequency")
plt.grid(True)
plt.show()




#%%
import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt

# Define constants
EC = 0.92
EL = 0.35
n_g_con = 2 * (2 * EC / EL) ** 0.25
g_a = 0.045/2 * n_g_con

# Define Fluxonium A
qbtA_ro = scq.Oscillator(E_osc=7, truncated_dim=5)

# Flux values to sweep over
flux_values = np.linspace(0, 1, 100)  # Sweep flux from 0 to 1 in 100 steps
chi_vals = []  # To store chi values for each flux

# Loop over flux values and compute chi for each flux
for flux in flux_values:
    # Update fluxonium with new flux value
    qbtA = scq.Fluxonium(
        EJ=4.08,
        EC=EC,
        EL=EL,
        flux=flux,  # Varying flux
        cutoff=110,
        truncated_dim=20,
    )

    # Define the common Hilbert space
    hilbertspace = scq.HilbertSpace([qbtA, qbtA_ro])

    # Add interaction between Fluxonium and oscillator
    hilbertspace.add_interaction(g=g_a, op1=qbtA.n_operator, op2=qbtA_ro.creation_operator, add_hc=True)

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Extract eigenvalues
    (evals,) = hilbertspace["evals"]

    # Dressed Hamiltonian in eigenbasis
    diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
    )

    # Truncate the Hamiltonian for faster computation
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

    # Get dressed eigenvalues in GHz
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract relevant dressed energy levels
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Calculate chi (MHz)
    chi = (e_11 - e_10 - e_01 + e_00) * 1e3  # Convert to MHz
    chi_vals.append(chi)

# Plot chi vs flux
plt.figure(figsize=(8, 6))
plt.plot(flux_values, chi_vals, label='Chi (MHz)', color='b')
plt.title('Dispersive Shift (Chi) vs Flux')
plt.xlabel('Flux ($\Phi/\Phi_0$)')
plt.ylabel('Chi (MHz)')
plt.grid(True)
plt.legend()
plt.show()


#%% Chi of the transmon


import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


EC = 0.04
EJ = 5
n_g_con = 2*(2*EC/EJ)**.25
# g_a = 0.04*n_g_con/2.5
g_a = 0.390*n_g_con
# define fluxonium A
qbtA = scq.Transmon(
     EJ=EJ,
     EC=EC,
     ng=0,
     ncut=110,
     truncated_dim=20)


qbtA_ro = scq.Oscillator(E_osc=7.4, truncated_dim=5)
# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbtA, qbtA_ro])


hilbertspace.add_interaction(g=g_a, op1=qbtA.n_operator, op2=qbtA_ro.creation_operator, add_hc=True)
# generate spectrum Lookup table
hilbertspace.generate_lookup()
# Hamiltonian in dressed eigenbasis
(evals,) = hilbertspace["evals"]
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation = 60
# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])


diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
evalues = (diag_dressed_hamiltonian_trunc.eigenenergies()-diag_dressed_hamiltonian_trunc.eigenenergies()[0])/6.28

e_11 = evalues[hilbertspace.dressed_index((1,1))]
e_10 = evalues[hilbertspace.dressed_index((1,0))]
e_01 = evalues[hilbertspace.dressed_index((0,1))]
e_00 = evalues[hilbertspace.dressed_index((0,0))]
# chi = hilbertspace.dressed_index((1,1))-hilbertspace.dressed_index((1,0))-hilbertspace.dressed_index((0,1))+hilbertspace.dressed_index((1,1,1))
# print("chi (MHz):",chi,chi_1)

chi= e_11-e_10-e_01+e_00
print("chi (MHz):",chi*1000)


#%% matrix elelment calculation

import scqubits as scq
qbta = scq.Fluxonium(
    EC=1.05,
    EJ = 6.5 ,
    EL=1.3,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ= 6,
     EC=1.05,
     ng=0,
     ncut=110,
     truncated_dim=6)


bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]



n_a = qbta.matrixelement_table('n_operator', evals_count=5)
n_b = qbtb.matrixelement_table('n_operator', evals_count=5)



ratio = (abs(n_a[1,0]/n_b[1,0]))**2
ratio


#%% Toms paper low freq calc chi

# Define given values (all in GHz)
E_C = 36.84e-3  # EC in GHz
E_10 = 1.18  # E10 in GHz
E_21 = 1.14  # E21 in GHz
omega_r = 8.0  # Resonator frequency in GHz
g = 500e-3  # Coupling strength in GHz

# Compute chi
hbar_chi = -8e3 * E_C * ((g**2) * (omega_r**2)) / (((omega_r**2) - (E_10**2)) * ((omega_r**2) - (E_21**2)))

# Display result
print(f"hbar * χ = {hbar_chi:.3f} MHz")

#%% toms paper low freq calc kappa


import numpy as np

# Given approximate values
C_q = 525e-15       # Capacitance of the qubit in Farads
C_res = 354e-15     # Capacitance of the resonator in Farads
C_c = 89e-15       # Coupling capacitance in Farads
omega_q = 2 * np.pi * 1.17e9   # Qubit frequency in rad/s (5 GHz)
omega_r = 2 * np.pi * 8e9   # Resonator frequency in rad/s (7 GHz)
g = 2 * np.pi * 321e6       # Coupling strength in rad/s (200 MHz)
k_ind = .5e6                 # Intrinsic dissipation rate in Hz (1 MHz)

# Calculate eta
eta = C_c / np.sqrt((C_res + C_c) * (C_q + C_c))

# Calculate kappa_q,ind
k_q_ind = (1 / (1 - eta**2)) * ((4 * omega_q**3 / omega_r**3) * (g**2 / omega_r**2) * k_ind)

# Print the result
print(f"Induced qubit decay rate (kappa_q,ind): {k_q_ind:.2e} Hz")

# Compute kappa_q
kappa_q = (g**2 / (omega_r - omega_q)**2) * k_ind

# Compute Purcell T1 time
T1_purcell = 1e6 / kappa_q

T1_purcell_corr = 1e6 / k_q_ind
