# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:27:17 2024

@author: WANGLAB
"""

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


qbta = scq.Fluxonium(
    EC=0.91,
    EJ = 3.863 ,
    EL=0.35,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ=9.9275,
     EC=0.220,
     ng=0,
     ncut=110,
     truncated_dim=10)


# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb])


# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.025,
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
total_truncation = 15

# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

evalues = (diag_dressed_hamiltonian_trunc.eigenenergies()-diag_dressed_hamiltonian_trunc.eigenenergies()[0])/6.28


# get the representation of the n_a operator in the dressed eigenbasis of the composite system
n_a = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbta.n_operator)
n_b = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtb.n_operator)
# truncate the operator after expressing in the dressed basis to speed up the simulation
n_a = truncate(n_a, total_truncation)
n_b = truncate(n_b, total_truncation)

# # convert the product states to the closes eigenstates of the dressed system
product_states_unsorted = [(0, 0), (1, 0), (0, 1),(2,0), (1, 1),(0,3) , (2,1),(0,2),(3,0)]#,(4,0),(1,2),(3,1),(2,2),(5,0),(4,1),(3,2),(0,4),(1,4),(2,3),(1,3)]

idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]

paired_data = list(zip(idxs_unsorted, product_states_unsorted))
sorted_data = sorted(paired_data, key=lambda x: x[0])
product_states = [data[1] for data in sorted_data]
idxs = [data[0] for data in sorted_data]
#sort after writing, paired data sort

states = [qt.basis(total_truncation, idx) for idx in idxs]

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]

index_to_state = {idx: f'{state[0]}{state[1]}' for idx, state in zip(idxs, product_states)}
# Function to get idsx value from (i, j) tuple
def get_idx(state_tuple):
    state_string = f'{state_tuple[0]}{state_tuple[1]}'
    for idx, state_str in index_to_state.items():
        if state_str == state_string:
            return idx
    return None  # Return None if state_tuple is not found


dim=total_truncation
Omega =np.zeros((dim ,dim))
freq_tran = np.zeros((dim ,dim))
computational_subspace = states[:5] 
def transition_frequency(s0: int, s1: int) -> float:
    return (
        (
            hilbertspace.energy_by_dressed_index(s1)
            - hilbertspace.energy_by_dressed_index(s0)
        )
        * 2
        * np.pi
    )
# Nested loop for i and j
for i in range(dim):
    for j in range(i+1, dim):
        # Calculate transition energy w for each pair i, j
        w = transition_frequency(i, j)/6.28 
        Omega[i][j] = w



Delta1 = 1000*(Omega[get_idx((1,0)),get_idx((2,0))]-Omega[get_idx((1,1)),get_idx((2,1))])
Delta2 = 1000*(Omega[get_idx((1,1)),get_idx((1,2))]-Omega[get_idx((0,1)),get_idx((0,2))])
Delta3 = 1000*(Omega[get_idx((1,0)),get_idx((1,3))]-Omega[get_idx((0,0)),get_idx((0,3))])
Delta4 = 1000*(Omega[get_idx((0,1)),get_idx((3,1))]-Omega[get_idx((0,0)),get_idx((3,0))])

Static_ZZ = 1000*(Omega[get_idx((1,0)),get_idx((1,1))]-Omega[get_idx((0,0)),get_idx((0,1))]) #MHz









n_diff = abs(np.round(n_a[0,2],5)-np.round(n_a[1,4],5))
n_A_01 = np.round(qbta.n_operator(energy_esys=True)[0][1],5)

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]
detuning  = abs(bare_states_a[1]-bare_states_b[1])

t_fsl = abs(n_A_01*2*np.pi/n_diff/detuning)


print(Static_ZZ)
print('Static_ZZ(MHz)=',Static_ZZ, 'bare_F_01 = ',bare_states_a[1],'bare_F_12 =',bare_states_a[2]-bare_states_a[1],'bare_T_01=',bare_states_b[1],'bare_F_03=',bare_states_a[3])




e_11 = evalues[hilbertspace.dressed_index((1,1))]
e_10 = evalues[hilbertspace.dressed_index((1,0))]
e_01 = evalues[hilbertspace.dressed_index((0,1))]
e_00 = evalues[hilbertspace.dressed_index((0,0))]
e_20 = evalues[hilbertspace.dressed_index((2,0))]
e_30 = evalues[hilbertspace.dressed_index((3,0))]
# e_01 = evalues[hilbertspace.dressed_index((0,1))]
# e_00 = evalues[hilbertspace.dressed_index((0,0))]

# drive_freq=Omega[get_idx((0,0)),get_idx((0,1))]
drive_freq = e_11-e_10
A=.22*.5
def cosine_drive(t: float, args: dict) -> float:
    return A *np.cos(6.28*drive_freq* t)



print('Static_ZZ(MHz)= ',(e_11-e_10-e_01+e_00)*1e3)
print('dressed_F_01(GHz)= ',(e_10-e_00)*1)
print('dressed_F_12(GHz)= ',(e_20-e_10)*1)
print('dressed_F_03(GHz)= ',(e_30-e_00)*1)
print('dressed_T_01(GHz)= ',(e_01-e_00)*1)

tlist = np.linspace(0, 400, 400)
H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * n_a, cosine_drive],  
]


result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index(product_states[1])),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index(product_states[0])),
    tlist,
    e_ops=[state * state.dag() for state in states]
)




plt.figure()
for idx, res in zip(idxs[:5], result.expect[:5]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium) in state |1>")

plt.figure()
for idx, res in zip(idxs[:5], result2.expect[:5]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium) in state |0>")








#%%
# Given initial parameters
eps_1F = .1
eps_2F_values = np.linspace(1, 10, 5)  # Varying eps_2F from 2 to 4
eps_1T = 1
eps_2T = 0.1  # Keeping eps_2T constant

del_values = np.linspace(0, 800, 2000)
Delta = Delta2

plt.figure(figsize=(8, 6))

def stark_effect(Omega, delta):
    return 0.5 * np.sqrt(Omega**2 + delta**2) - delta / 2
for eps_2F in eps_2F_values:
    # Calculate Omega values for each eps_2F
    Omega_1112 = abs(n_a[get_idx((1,1)), get_idx((1,2))]) * (eps_1F + eps_2F) + abs(n_b[get_idx((1,1)), get_idx((1,2))]) * (eps_1T + eps_2T)
    Omega_0102 = abs(n_a[get_idx((0,1)), get_idx((0,2))]) * (eps_1F + eps_2F) + abs(n_b[get_idx((0,1)), get_idx((0,2))]) * (eps_1T + eps_2T)
    
    # Calculate Drive_ZZ using the updated Omega values
    Drive_ZZ = -stark_effect(30 * Omega_1112, del_values) + stark_effect(30 * Omega_0102, del_values - (Delta))
    
    plt.plot(del_values, abs(Drive_ZZ + abs(Static_ZZ)), label=f'|Drive_ZZ- Static_ZZ|, eps_2F={eps_2F:.2f}', linewidth=1.5)

plt.title('Total ZZ vs (red) detuning')
plt.xlabel('detuning (MHz)')
plt.ylabel('|ZZ value (Mhz)|')
plt.legend()
plt.grid(True)
plt.show()



#%% temp

import scqubits as scq
import qutip as qt
import numpy as np

def calculate_quantum_properties(EJ_fluxonium, EJ_transmon):
    # define fluxonium A
    qbta = scq.Fluxonium(
        EC=.92,
        EJ=EJ_fluxonium,
        EL=.35,
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=5,
    )

    # define transmon
    qbtb = scq.Transmon(
        EJ=EJ_transmon,
        EC=0.2,
        ng=0,
        ncut=110,
        truncated_dim=5
    )

    # define the common Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # add interaction between two qubits
    hilbertspace.add_interaction(
        g_strength=0.024,
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Hamiltonian in dressed eigenbasis
    (evals,) = hilbertspace["evals"]
    diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
    )

    total_truncation = 20

    # truncate operators to desired dimension
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

    # get the representation of the n_a operator in the dressed eigenbasis of the composite system
    n_a = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbta.n_operator)
    n_b = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtb.n_operator)
    n_a = truncate(n_a, total_truncation)
    n_b = truncate(n_b, total_truncation)

    product_states_unsorted = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 3), (2, 1), (0, 2), (3, 0)]
    idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]

    paired_data = list(zip(idxs_unsorted, product_states_unsorted))
    sorted_data = sorted(paired_data, key=lambda x: x[0])
    product_states = [data[1] for data in sorted_data]
    idxs = [data[0] for data in sorted_data]

    states = [qt.basis(total_truncation, idx) for idx in idxs]

    bare_states_a = qbta.eigenvals() - qbta.eigenvals()[0]
    bare_states_b = qbtb.eigenvals() - qbtb.eigenvals()[0]

    index_to_state = {idx: f'{state[0]}{state[1]}' for idx, state in zip(idxs, product_states)}

    def get_idx(state_tuple):
        state_string = f'{state_tuple[0]}{state_tuple[1]}'
        for idx, state_str in index_to_state.items():
            if state_str == state_string:
                return idx
        return None  # Return None if state_tuple is not found

    dim = total_truncation
    Omega = np.zeros((dim, dim))

    def transition_frequency(s0: int, s1: int) -> float:
        return (
            (
                hilbertspace.energy_by_dressed_index(s1)
                - hilbertspace.energy_by_dressed_index(s0)
            )
            * 2
            * np.pi
        )

    for i in range(dim):
        for j in range(i + 1, dim):
            w = transition_frequency(i, j) / 6.28 
            Omega[i][j] = w

    Static_ZZ = 1000 * (Omega[get_idx((1, 0)), get_idx((1, 1))] - Omega[get_idx((0, 0)), get_idx((0, 1))])  # MHz

    F_01 = bare_states_a[1]
    F_12 = bare_states_a[2] - bare_states_a[1]
    T_01 = bare_states_b[1]

    return Static_ZZ, F_01, F_12, T_01
#%%
# Example usage
EJ_fluxonium = 3.33
EJ_transmon = 6.5
Static_ZZ, F_01, F_12, T_01 = calculate_quantum_properties(EJ_fluxonium, EJ_transmon)
print(f'Static_ZZ = {Static_ZZ:.2f}, F_01 = {F_01:.2f}, F_12 = {F_12:.2f}, T_01 = {T_01:.2f}')
#%%
import numpy as np

def calculate_EJ(JJ_length, Jc):
    # Constants
    phi_0 = 2.067e-15  # Magnetic flux quantum in Wb
    h = 6.626e-34  # Planck's constant in Js
    pi = np.pi
    
    # Convert Jc from uA/um^2 to A/m^2
    Jc_A_m2 = Jc * 1e6
    
    # Calculate EJ in Hz
    EJ_Hz = (Jc_A_m2 * JJ_length * 2e3 * 1e-9 * phi_0) / (2 * pi * h)
    
    # Convert EJ to GHz
    EJ_GHz = EJ_Hz / 1e9
    
    return EJ_GHz

# Example usage
JJ_length = 380  # in nm
Jc = 0.356  # in uA/um^2
EJ = calculate_EJ(JJ_length, Jc)*1e-10
print(f"EJ = {EJ:.2f} GHz")



#%% 3 mmode system


import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
# from qutip.qip.operations import rz, cz_gate
# import cmath


qbta = scq.Fluxonium(
    EC=1,
    EJ = 4.5 ,
    EL=1,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)
qbtb = scq.Fluxonium(
    EC=1,
    EJ = 4 ,
    EL=1,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)
qbtc = scq.Transmon(
     EJ=17,
     EC=0.2,
     ng=0,
     ncut=110,
     truncated_dim=10)

# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb,qbtc])


# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.020,
    op1=qbta.n_operator,
    op2=qbtc.n_operator,
)
hilbertspace.add_interaction(
    g_strength=0.020,
    op1=qbtb.n_operator,
    op2=qbtc.n_operator,
)

# generate spectrum lookup table
hilbertspace.generate_lookup()


# Convert bare index to dressed index and print them
print(f"{'Bare State':>12} -> {'Dressed Index':>15}")
for bare_index in range(20):
    dressed_index = hilbertspace.bare_index(bare_index)
    
    # Convert tuple to a string for clean output
    dressed_index_str = ", ".join(map(str, dressed_index))
    
    print(f"{bare_index:>12} -> {dressed_index_str:>15}")
    

#%% Static _ZZ calc


import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
# from qutip.qip.operations import rz, cz_gate
# import cmath
#Variables
qbta = scq.Fluxonium(
    EC=.92,
    EJ = 4.08 ,
    EL=.35,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ=10.2,
     EC=0.22,
     ng=0,
     ncut=110,
     truncated_dim=10)

# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb])


# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.026,
    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)


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

ZZ= e_11-e_10-e_01+e_00
print("ZZ (MHz):",ZZ*1000)

#%%Static_ZZ vs g vs EJ

import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(EJ, g_strength):
    # Define fluxonium
    qbta = scq.Fluxonium(
        EC=0.92,
        EJ=4.08,
        EL=0.35,
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with varying EJ
    qbtb = scq.Transmon(
        EJ=EJ,  # Variable EJ
        EC=0.22,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength/1000,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Transmon EJ and g_strength
EJ_range = np.linspace(8, 14, 20)  # Transmon EJ from 8 to 14 GHz
g_strength_range = np.linspace(0.008, 0.03, 20)*1000  # g_strength from 0.01 to 0.03

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(EJ_range)))
omega_01_values = np.zeros((len(g_strength_range), len(EJ_range)))

# Loop over the ranges of EJ and g_strength, calculate Static ZZ and omega_01
for i, g_strength in enumerate(g_strength_range):
    for j, EJ in enumerate(EJ_range):
        Static_ZZ, omega_01 = compute_static_zz(EJ, g_strength)
        Static_ZZ_values[i, j] = Static_ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
omega_01_grid, g_strength_grid = np.meshgrid(omega_01_values[0, :], g_strength_range)

# Plot the 2D color map
plt.figure(figsize=(8, 6))
plt.contourf(omega_01_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(label='Static ZZ (MHz)')
plt.xlabel('Transmon ω01 (GHz)')
plt.ylabel('g_strength')
plt.title('Static ZZ as a function of Transmon EJ (ω01) and g_strength')
plt.show()

#%%


import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(EJ, g_strength):
    # Define fluxonium
    qbta = scq.Fluxonium(
        EC=0.92,
        EJ=4.08,
        EL=0.35,
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with varying EJ
    qbtb = scq.Transmon(
        EJ=EJ,  # Variable EJ
        EC=0.22,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )
    
    # qbta = scq.Fluxonium(
    #     EC=.1,
    #     EJ = 4 ,
    #     EL=1,
    #     flux=0.5,  # flux frustration point
    #     cutoff=110,
    #     truncated_dim=10,
    # )


    # qbtb = scq.Transmon(
    #      EJ=EJ,
    #      EC=0.2,
    #      ng=0,
    #      ncut=110,
         # truncated_dim=10)

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Transmon EJ and g_strength
EJ_range = np.linspace(9.7, 10.3, 10)  # Transmon EJ from 8 to 14 GHz
g_strength_range = np.linspace(0.015, 0.025, 10)  # g_strength from 0.01 to 0.03

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(EJ_range)))
omega_01_values = np.zeros((len(g_strength_range), len(EJ_range)))

# Loop over the ranges of EJ and g_strength, calculate Static ZZ and omega_01
for i, g_strength in enumerate(g_strength_range):
    for j, EJ in enumerate(EJ_range):
        Static_ZZ, omega_01 = compute_static_zz(EJ, g_strength)
        Static_ZZ_values[i, j] = Static_ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
omega_01_grid, g_strength_grid = np.meshgrid(omega_01_values[0, :], g_strength_range)

# Plot the 2D color map
fig, ax1 = plt.subplots(figsize=(8, 6))
contour = ax1.contourf(omega_01_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(contour, label='Static ZZ (MHz)')
ax1.set_xlabel('Transmon ω01 (GHz)')
ax1.set_ylabel('g_strength')
ax1.set_title('Static ZZ as a function of Transmon EJ (ω01) and g_strength')

# Create a second x-axis (top) to display EJ values
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Align the two x-axes
# Increase the number of ticks on the top axis (EJ) for more granularity
ax2.set_xticks(np.linspace(min(omega_01_grid[0]), max(omega_01_grid[0]), len(EJ_range[::5])))  # More tick marks for EJ
ax2.set_xticklabels([f"{EJ:.1f}" for EJ in EJ_range[::5]])  # EJ values as labels
ax2.set_xlabel('Transmon EJ (GHz)')

plt.show()

#%%

import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(EJ, g_strength):
    # Define fluxonium
    qbta = scq.Fluxonium(
        EC=0.92,
        EJ=4.08,
        EL=0.35,
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with varying EJ
    qbtb = scq.Transmon(
        EJ=EJ,  # Variable EJ
        EC=0.22,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Transmon EJ and g_strength
EJ_range = np.linspace(8, 14, 10)  # Transmon EJ from 8 to 14 GHz
g_strength_range = np.linspace(0.01, 0.018, 10)  # g_strength from 0.01 to 0.03

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(EJ_range)))
omega_01_values = np.zeros((len(g_strength_range), len(EJ_range)))

# Modify the ZZ calculation to take the absolute value of ZZ
for i, g_strength in enumerate(g_strength_range):
    for j, EJ in enumerate(EJ_range):
        Static_ZZ, omega_01 = compute_static_zz(EJ, g_strength)
        Static_ZZ_values[i, j] = np.abs(Static_ZZ)  # Take the absolute value of Static ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
omega_01_grid, g_strength_grid = np.meshgrid(omega_01_values[0, :], g_strength_range)

# Plot the 2D color map with absolute ZZ
fig, ax1 = plt.subplots(figsize=(8, 6))
contour = ax1.contourf(omega_01_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(contour, label='|Static ZZ| (MHz)')
ax1.set_xlabel('Transmon ω01 (GHz)')
ax1.set_ylabel('g_strength')
ax1.set_title('Absolute Static ZZ as a function of Transmon EJ (ω01) and g_strength')

# Create a second x-axis (top) to display EJ values
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())  # Align the two x-axes
# Increase the number of ticks on the top axis (EJ) for more granularity
ax2.set_xticks(np.linspace(min(omega_01_grid[0]), max(omega_01_grid[0]), len(EJ_range[::5])))  # More tick marks for EJ
ax2.set_xticklabels([f"{EJ:.1f}" for EJ in EJ_range[::5]])  # EJ values as labels
ax2.set_xlabel('Transmon EJ (GHz)')

plt.show()

#%%

import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(fluxonium_EJ, g_strength):
    # Define fluxonium with varying EJ
    # qbta = scq.Fluxonium(
    #     EC=0.92,
    #     EJ=fluxonium_EJ,  # Variable EJ for fluxonium
    #     EL=0.35,
    #     flux=0.5,  # flux frustration point
    #     cutoff=110,
    #     truncated_dim=10,
    # )

    # # Define transmon with fixed EJ
    # qbtb = scq.Transmon(
    #     EJ=9.738,  # Fixed EJ for transmon
    #     EC=0.22,
    #     ng=0,
    #     ncut=110,
    #     truncated_dim=10,
    # )
    
    qbta = scq.Fluxonium(
        EC=1,
        EJ=fluxonium_EJ,  # Variable EJ for fluxonium
        EL=1,
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with fixed EJ
    qbtb = scq.Transmon(
        EJ=4.5,  # Fixed EJ for transmon
        EC=0.07,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Fluxonium EJ and g_strength
fluxonium_EJ_range = np.linspace(3.9, 4.2, 10)  # Fluxonium EJ from 5 to 15 GHz
g_strength_range = np.linspace(0.01, 0.03, 20)  # g_strength from 0.01 to 0.03

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(fluxonium_EJ_range)))
omega_01_values = np.zeros((len(g_strength_range), len(fluxonium_EJ_range)))

# Modify the ZZ calculation to take the absolute value of ZZ
for i, g_strength in enumerate(g_strength_range):
    for j, fluxonium_EJ in enumerate(fluxonium_EJ_range):
        Static_ZZ, omega_01 = compute_static_zz(fluxonium_EJ, g_strength)
        Static_ZZ_values[i, j] = np.abs(Static_ZZ)  # Take the absolute value of Static ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
fluxonium_EJ_grid, g_strength_grid = np.meshgrid(fluxonium_EJ_range, g_strength_range)

# Plot the 2D color map with absolute ZZ
fig, ax1 = plt.subplots(figsize=(8, 6))
contour = ax1.contourf(fluxonium_EJ_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(contour, label='|Static ZZ| (MHz)')
ax1.set_xlabel('Fluxonium EJ (GHz)')
ax1.set_ylabel('g_strength')
ax1.set_title('Absolute Static ZZ as a function of Fluxonium EJ and g_strength')

plt.show()


#%% ZZ with contour line Varying g_strength vs Fluxonium EJ

import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(fluxonium_EJ, g_strength):
    # Define fluxonium with varying EJ
    qbta = scq.Fluxonium(
        EC=0.92,
        EJ=fluxonium_EJ,  # Variable EJ for fluxonium
        EL=0.35,
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with fixed EJ
    qbtb = scq.Transmon(
        EJ=9.738,  # Fixed EJ for transmon
        EC=0.22,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Fluxonium EJ and g_strength
fluxonium_EJ_range = np.linspace(3.9, 4.3, 10)  # Fluxonium EJ from 5 to 15 GHz
g_strength_range = np.linspace(0.01, 0.04, 20)  # g_strength from 0.01 to 0.03

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(fluxonium_EJ_range)))
omega_01_values = np.zeros((len(g_strength_range), len(fluxonium_EJ_range)))

# Modify the ZZ calculation to take the absolute value of ZZ
for i, g_strength in enumerate(g_strength_range):
    for j, fluxonium_EJ in enumerate(fluxonium_EJ_range):
        Static_ZZ, omega_01 = compute_static_zz(fluxonium_EJ, g_strength)
        Static_ZZ_values[i, j] = np.abs(Static_ZZ)  # Take the absolute value of Static ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
fluxonium_EJ_grid, g_strength_grid = np.meshgrid(fluxonium_EJ_range, g_strength_range)

# Plot the 2D color map with contour lines for each ZZ value
fig, ax1 = plt.subplots(figsize=(8, 6))

# Create filled contour plot for |Static ZZ|
contour = ax1.contourf(fluxonium_EJ_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(contour, label='|Static ZZ| (MHz)')

# Add contour lines for specific ZZ values
contour_lines = ax1.contour(fluxonium_EJ_grid, g_strength_grid, Static_ZZ_values, colors='white', levels=np.linspace(np.min(Static_ZZ_values), np.max(Static_ZZ_values), 10))
ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f MHz')

# Set axis labels and title
ax1.set_xlabel('Fluxonium EJ (GHz)')
ax1.set_ylabel('g_strength')
ax1.set_title('Absolute Static ZZ with Contour Lines for each ZZ Value')

plt.show()
#%% 
import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(fluxonium_EL, g_strength):
    # Define fluxonium with varying EL and fixed EJ
    qbta = scq.Fluxonium(
        EC=0.92,
        EJ=4.08,  # Fixed EJ for fluxonium
        EL=fluxonium_EL,  # Variable EL for fluxonium
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with fixed EJ
    qbtb = scq.Transmon(
        EJ=9.738,  # Fixed EJ for transmon
        EC=0.22,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Fluxonium EL and g_strength
fluxonium_EL_range = np.linspace(0.34, 0.36, 10)  # Fluxonium EL from 0.2 to 0.6 GHz
g_strength_range = np.linspace(0.01, 0.03, 20)  # g_strength from 0.01 to 0.04

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(fluxonium_EL_range)))
omega_01_values = np.zeros((len(g_strength_range), len(fluxonium_EL_range)))

# Modify the ZZ calculation to take the absolute value of ZZ
for i, g_strength in enumerate(g_strength_range):
    for j, fluxonium_EL in enumerate(fluxonium_EL_range):
        Static_ZZ, omega_01 = compute_static_zz(fluxonium_EL, g_strength)
        Static_ZZ_values[i, j] = np.abs(Static_ZZ)  # Take the absolute value of Static ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
fluxonium_EL_grid, g_strength_grid = np.meshgrid(fluxonium_EL_range, g_strength_range)

# Plot the 2D color map with contour lines for each ZZ value
fig, ax1 = plt.subplots(figsize=(8, 6))

# Create filled contour plot for |Static ZZ|
contour = ax1.contourf(fluxonium_EL_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(contour, label='|Static ZZ| (MHz)')

# Add contour lines for specific ZZ values
contour_lines = ax1.contour(fluxonium_EL_grid, g_strength_grid, Static_ZZ_values, colors='white', levels=np.linspace(np.min(Static_ZZ_values), np.max(Static_ZZ_values), 10))
ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f MHz')

# Set axis labels and title
ax1.set_xlabel('Fluxonium EL (GHz)')
ax1.set_ylabel('g_strength')
ax1.set_title('Absolute Static ZZ with Contour Lines for varying Fluxonium EL')

plt.show()

#%% Varying EJ Transmon

import scqubits as scq
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Static ZZ and omega_01
def compute_static_zz(transmon_EJ, g_strength):
    # Define fluxonium with fixed EL and EJ
    qbta = scq.Fluxonium(
        EC=0.92,
        EJ=4.08,  # Fixed EJ for fluxonium
        EL=0.35,  # Fixed EL for fluxonium
        flux=0.5,  # flux frustration point
        cutoff=110,
        truncated_dim=10,
    )

    # Define transmon with varying EJ
    qbtb = scq.Transmon(
        EJ=transmon_EJ,  # Variable EJ for transmon
        EC=0.22,
        ng=0,
        ncut=110,
        truncated_dim=10,
    )

    # Define the Hilbert space
    hilbertspace = scq.HilbertSpace([qbta, qbtb])

    # Add interaction with varying g_strength
    hilbertspace.add_interaction(
        g_strength=g_strength,  # Variable g_strength
        op1=qbta.n_operator,
        op2=qbtb.n_operator,
    )

    # Generate spectrum lookup table
    hilbertspace.generate_lookup()

    # Get dressed eigenenergies
    (evals,) = hilbertspace["evals"]

    # Convert energies from GHz to natural units (radians/ns)
    diag_dressed_hamiltonian = 2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)

    # Truncate operators to desired dimension
    total_truncation = 60
    def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
        return qt.Qobj(operator[:dimension, :dimension])

    diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
    evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

    # Extract dressed energies for the states (1,1), (1,0), (0,1), and (0,0)
    e_11 = evalues[hilbertspace.dressed_index((1, 1))]
    e_10 = evalues[hilbertspace.dressed_index((1, 0))]
    e_01 = evalues[hilbertspace.dressed_index((0, 1))]
    e_00 = evalues[hilbertspace.dressed_index((0, 0))]

    # Static ZZ calculation
    ZZ = (e_11 - e_10 - e_01 + e_00) * 1000  # Convert to MHz

    # Calculate omega_01 (transition frequency between 0 and 1 for the transmon)
    omega_01 = evalues[hilbertspace.dressed_index((0, 1))]  # Transition from (0,0) to (0,1)

    return ZZ, omega_01

# Parameter ranges for Transmon EJ and g_strength
transmon_EJ_range = np.linspace(9.6, 9.8, 10)  # Transmon EJ from 8 to 14 GHz
g_strength_range = np.linspace(0.01, 0.04, 20)  # g_strength from 0.01 to 0.04

# Arrays to store Static ZZ and omega_01 values
Static_ZZ_values = np.zeros((len(g_strength_range), len(transmon_EJ_range)))
omega_01_values = np.zeros((len(g_strength_range), len(transmon_EJ_range)))

# Modify the ZZ calculation to take the absolute value of ZZ
for i, g_strength in enumerate(g_strength_range):
    for j, transmon_EJ in enumerate(transmon_EJ_range):
        Static_ZZ, omega_01 = compute_static_zz(transmon_EJ, g_strength)
        Static_ZZ_values[i, j] = np.abs(Static_ZZ)  # Take the absolute value of Static ZZ
        omega_01_values[i, j] = omega_01

# Use numpy meshgrid to create 2D grids for contour plotting
transmon_EJ_grid, g_strength_grid = np.meshgrid(transmon_EJ_range, g_strength_range)

# Plot the 2D color map with contour lines for each ZZ value
fig, ax1 = plt.subplots(figsize=(8, 6))

# Create filled contour plot for |Static ZZ|
contour = ax1.contourf(transmon_EJ_grid, g_strength_grid, Static_ZZ_values, levels=100, cmap='viridis')
plt.colorbar(contour, label='|Static ZZ| (MHz)')

# Add contour lines for specific ZZ values
contour_lines = ax1.contour(transmon_EJ_grid, g_strength_grid, Static_ZZ_values, colors='white', levels=np.linspace(np.min(Static_ZZ_values), np.max(Static_ZZ_values), 10))
ax1.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f MHz')

# Set axis labels and title
ax1.set_xlabel('Transmon EJ (GHz)')
ax1.set_ylabel('g_strength')
ax1.set_title('Absolute Static ZZ with Contour Lines for varying Transmon EJ')

plt.show()
