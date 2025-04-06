# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:27:57 2024

@author: WANGLAB
"""

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


qbta = scq.Fluxonium(
    EC=1.2,
    EJ = 5.6 ,
    EL=.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)

qbtb = scq.Transmon(
     EJ=5,
     EC=0.04,
     ng=0,
     ncut=110,
     truncated_dim=10,
     )

qbtc = scq.Fluxonium(
    EC=1.2,
    EJ = 5.5 ,
    EL=0.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]
bare_states_c = qbtc.eigenvals()-qbtc.eigenvals()[0]

# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb, qbtc])


# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.02,
    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)

hilbertspace.add_interaction(
    g_strength=0.02,
    op1=qbtc.n_operator,
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
total_truncation = 30

# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

evalues = (diag_dressed_hamiltonian_trunc.eigenenergies()-diag_dressed_hamiltonian_trunc.eigenenergies()[0])/6.28


# get the representation of the n_a operator in the dressed eigenbasis of the composite system
n_a = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbta.n_operator)
n_b = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtb.n_operator)
n_c = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtc.n_operator)


# truncate the operator after expressing in the dressed basis to speed up the simulation
n_a = truncate(n_a, total_truncation)
n_b = truncate(n_b, total_truncation)
n_c = truncate(n_c, total_truncation)

for i in range(13):
    print(f"Index {i}: {hilbertspace.bare_index(i)}")


evals = evalues    
idx000 = hilbertspace.dressed_index((0, 0, 0))
idx110 = hilbertspace.dressed_index((1, 1, 0))
idx010 = hilbertspace.dressed_index((0, 1, 0))
idx100 = hilbertspace.dressed_index((1, 0, 0))
idx001 = hilbertspace.dressed_index((0, 0, 1))
idx111 = hilbertspace.dressed_index((1, 1, 1))
idx011 = hilbertspace.dressed_index((0, 1, 1))
idx101 = hilbertspace.dressed_index((1, 0, 1))

idx200 = hilbertspace.dressed_index((2, 0, 0))
idx020 = hilbertspace.dressed_index((0, 2, 0))
idx002 = hilbertspace.dressed_index((0, 0, 2))



print(f"Fluxonium A 0-1: {evals[idx100]:.6f} GHz")
print(f"Transmon 0-1: {evals[idx010]:.6f} GHz")
print(f"Fluxonium B 0-1: {evals[idx001]:.6f} GHz")
print(f"Fluxonium A 1-2: {evals[idx200] - evals[idx100]:.6f} GHz")
print(f"Transmon 1-2: {evals[idx020] - evals[idx010]:.6f} GHz")
print(f"Fluxonium B 1-2: {evals[idx002] - evals[idx001]:.6f} GHz")



ZZI0 =  abs(evals[idx110] - evals[idx100] - evals[idx010] + evals[idx000])*1e3 #MHz
ZZI1 =  abs(evals[idx111] - evals[idx101] - evals[idx011] + evals[idx001])*1e3  #MHz   



I0ZZ =  abs(evals[idx011] - evals[idx001] - evals[idx010] + evals[idx000])*1e3 #MHz
I1ZZ =  abs(evals[idx111] - evals[idx101] - evals[idx110] + evals[idx100])*1e3  #MHz    
    
ZZ13_0 = abs(evals[idx101] - evals[idx100] - evals[idx001] + evals[idx000]) * 1e3  # MHz
ZZ13_1 = abs(evals[idx111] - evals[idx110] - evals[idx011] + evals[idx010]) * 1e3  # MHz
ZZ13 = (ZZ13_0 + ZZ13_1) / 2
# Calculate and print ZZ interactions for each control qubit
ZZ_1 = (I0ZZ + I1ZZ) / 2
ZZ_3 = (ZZI0 + ZZI1) / 2
print(f"ZZ Interaction AB = {ZZ_1:.3f} MHz")
print(f"ZZ Interaction BC = {ZZ_3:.3f} MHz")
print(f"ZZ Interaction AC = {ZZ13:.6f} MHz")



#%% using the ftf parameters.json

import scqubits as scq
import qutip as qt
import numpy as np
import json
from matplotlib import pyplot as plt

# -----------------------------------------------------------
# Load parameters from ftf_parameters.json
# -----------------------------------------------------------
with open(r"Z:\Tanvir\Tanvir Files and codes\Codes_11_14_2024\ftf_parameters.json", "r") as file:
    all_params = json.load(file)

# Select the desired parameter set (e.g., "ftf_set1")
selected_set = "ftf_set1"
params = all_params[selected_set]

# Use parameters from the JSON file
fluxonium_a_params = params["fluxonium_a"]
transmon_params = params["transmon"]
fluxonium_b_params = params["fluxonium_b"]
JC = params["coupling"]["JC"]
A_A = params["drive_amplitudes"]["A_A"]
A_B = params["drive_amplitudes"]["A_B"]

# -----------------------------------------------------------
# Define the qubits using the loaded parameters
# -----------------------------------------------------------
qbta = scq.Fluxonium(
    EC=fluxonium_a_params["EC"],
    EJ=fluxonium_a_params["EJ"],
    EL=fluxonium_a_params["EL"],
    flux=0.5,
    cutoff=110,
    truncated_dim=10,
)

qbtb = scq.Transmon(
    EJ=transmon_params["EJ"],
    EC=transmon_params["EC"],
    ng=0,
    ncut=110,
    truncated_dim=10,
)

qbtc = scq.Fluxonium(
    EC=fluxonium_b_params["EC"],
    EJ=fluxonium_b_params["EJ"],
    EL=fluxonium_b_params["EL"],
    flux=0.5,
    cutoff=110,
    truncated_dim=10,
)

# -----------------------------------------------------------
# Bare state calculations
# -----------------------------------------------------------
bare_states_a = qbta.eigenvals() - qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals() - qbtb.eigenvals()[0]
bare_states_c = qbtc.eigenvals() - qbtc.eigenvals()[0]

# -----------------------------------------------------------
# Define the common Hilbert space
# -----------------------------------------------------------
hilbertspace = scq.HilbertSpace([qbta, qbtb, qbtc])

# Add interactions between qubits
hilbertspace.add_interaction(
    g_strength=JC,
    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)
hilbertspace.add_interaction(
    g_strength=JC,
    op1=qbtc.n_operator,
    op2=qbtb.n_operator,
)

# Generate spectrum lookup table
hilbertspace.generate_lookup()

# -----------------------------------------------------------
# Hamiltonian in dressed eigenbasis
# -----------------------------------------------------------
(evals,) = hilbertspace["evals"]
diag_dressed_hamiltonian = (
    2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)
)

# Truncate operators to desired dimension
total_truncation = 30

def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)
evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

# -----------------------------------------------------------
# Get the representation of the n_a, n_b, n_c operators
# -----------------------------------------------------------
n_a = truncate(hilbertspace.op_in_dressed_eigenbasis(qbta.n_operator), total_truncation)
n_b = truncate(hilbertspace.op_in_dressed_eigenbasis(qbtb.n_operator), total_truncation)
n_c = truncate(hilbertspace.op_in_dressed_eigenbasis(qbtc.n_operator), total_truncation)

# -----------------------------------------------------------
# ZZ interactions
# -----------------------------------------------------------
evals = evalues    
idx000 = hilbertspace.dressed_index((0, 0, 0))
idx110 = hilbertspace.dressed_index((1, 1, 0))
idx010 = hilbertspace.dressed_index((0, 1, 0))
idx100 = hilbertspace.dressed_index((1, 0, 0))
idx001 = hilbertspace.dressed_index((0, 0, 1))
idx111 = hilbertspace.dressed_index((1, 1, 1))
idx011 = hilbertspace.dressed_index((0, 1, 1))
idx101 = hilbertspace.dressed_index((1, 0, 1))

idx200 = hilbertspace.dressed_index((2, 0, 0))
idx020 = hilbertspace.dressed_index((0, 2, 0))
idx002 = hilbertspace.dressed_index((0, 0, 2))



print(f"Fluxonium A 0-1: {evals[idx100]:.6f} GHz")
print(f"Transmon 0-1: {evals[idx010]:.6f} GHz")
print(f"Fluxonium B 0-1: {evals[idx001]:.6f} GHz")
print(f"Fluxonium A 1-2: {evals[idx200] - evals[idx100]:.6f} GHz")
print(f"Transmon 1-2: {evals[idx020] - evals[idx010]:.6f} GHz")
print(f"Fluxonium B 1-2: {evals[idx002] - evals[idx001]:.6f} GHz")



ZZI0 =  abs(evals[idx110] - evals[idx100] - evals[idx010] + evals[idx000])*1e3 #MHz
ZZI1 =  abs(evals[idx111] - evals[idx101] - evals[idx011] + evals[idx001])*1e3  #MHz   



I0ZZ =  abs(evals[idx011] - evals[idx001] - evals[idx010] + evals[idx000])*1e3 #MHz
I1ZZ =  abs(evals[idx111] - evals[idx101] - evals[idx110] + evals[idx100])*1e3  #MHz    
    
ZZ13_0 = abs(evals[idx101] - evals[idx100] - evals[idx001] + evals[idx000]) * 1e3  # MHz
ZZ13_1 = abs(evals[idx111] - evals[idx110] - evals[idx011] + evals[idx010]) * 1e3  # MHz
ZZ13 = (ZZ13_0 + ZZ13_1) / 2
# Calculate and print ZZ interactions for each control qubit
ZZ_1 = (I0ZZ + I1ZZ) / 2
ZZ_3 = (ZZI0 + ZZI1) / 2
print(f"ZZ Interaction AB = {ZZ_1:.3f} MHz")
print(f"ZZ Interaction BC = {ZZ_3:.3f} MHz")
print(f"ZZ Interaction AC = {ZZ13:.6f} MHz")

# Continue with your simulation and plotting as required...





#%% Create idxs

# Create idxs with values from 0 to 12
idxs = list(range(15))

# Generate product_states by applying hilbertspace.bare_index to each index
product_states = [hilbertspace.bare_index(i) for i in idxs]

# Print the results
for idx, state, energy in zip(idxs, product_states, evalues):
    print(f"{idx} -> {state}-> {energy}")
    
states = [qt.basis(total_truncation, idx) for idx in idxs]
index_to_state = {idx: f'{state[0]}{state[1]}{state[2]}' for idx, state in zip(idxs, product_states)}


# Create a dictionary to store the results
e_values_dict = {}

# Iterate over the states in index_to_state
for idx, state in index_to_state.items():
    # Convert the state string into a tuple of integers (e.g., '001' -> (0, 0, 1))
    state_tuple = tuple(map(int, state))
    
    # Use hilbertspace.dressed_index and evalues to get the energy
    globals()[f"e_{state}"] = evalues[hilbertspace.dressed_index(state_tuple)]


#%% Driving qubit A

n_a_000_010 = n_a[hilbertspace.dressed_index((0,0,0)),hilbertspace.dressed_index((0,1,0))]
n_b_000_010 = n_b[hilbertspace.dressed_index((0,0,0)),hilbertspace.dressed_index((0,1,0))]


eta_A = -n_a_000_010/n_b_000_010


drive_freq_A =evalues[hilbertspace.dressed_index((1,1,0))]- evalues[hilbertspace.dressed_index((1,0,0))]
# A_A=.22*3
def cosine_drive(t: float, args: dict) -> float:
    return A_A *np.cos(6.28*drive_freq_A* t)
tlist = np.linspace(0, 400, 400)
H_qbt_drive_A = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a+eta_A*n_b), cosine_drive],  
]

result = qt.sesolve(
    H_qbt_drive_A,
    qt.basis(total_truncation, hilbertspace.dressed_index((1,0,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive_A,
    qt.basis(total_truncation, hilbertspace.dressed_index((0,0,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)

plt.figure()
for idx, res in zip(idxs[:15], result.expect[:15]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium_A) in state |1>")

plt.figure()
for idx, res in zip(idxs[:10], result2.expect[:10]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium_A) in state |0>")

#%% driving qubit B

drive_freq_B =evalues[hilbertspace.dressed_index((0,1,1))]- evalues[hilbertspace.dressed_index((0,0,1))]

n_c_000_010 = n_c[hilbertspace.dressed_index((0,0,0)),hilbertspace.dressed_index((0,1,0))]
n_b_000_010 = n_b[hilbertspace.dressed_index((0,0,0)),hilbertspace.dressed_index((0,1,0))]


eta_B = -n_c_000_010/n_b_000_010

# A_B=.22*3
def cosine_drive(t: float, args: dict) -> float:
    return A_B *np.cos(6.28*drive_freq_B* t)
tlist = np.linspace(0, 400, 400)
H_qbt_drive_B = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_c+eta_B*n_b), cosine_drive],  
]

result = qt.sesolve(
    H_qbt_drive_B,
    qt.basis(total_truncation, hilbertspace.dressed_index((0,0,1))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive_B,
    qt.basis(total_truncation, hilbertspace.dressed_index((0,0,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)

plt.figure()
for idx, res in zip(idxs[:10], result.expect[:10]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium_B) in state |1>")

plt.figure()
for idx, res in zip(idxs[:10], result2.expect[:10]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium_B) in state |0>")

#%% Simultaneous

drive_freq_B =evalues[hilbertspace.dressed_index((0,1,1))]- evalues[hilbertspace.dressed_index((0,0,1))]
# A_B=.22*1.5
def cosine_drive(t: float, args: dict) -> float:
    return A_B *np.cos(6.28*drive_freq_B* t)
tlist = np.linspace(0, 400, 400)
# H_qbt_drive_B = [
#     diag_dressed_hamiltonian_trunc,
#     [2 * np.pi * (n_c+(eta_A+eta_B)*n_b), cosine_drive],  
# ]

result = qt.sesolve(
    H_qbt_drive_B+ H_qbt_drive_A,
    qt.basis(total_truncation, hilbertspace.dressed_index((1,0,1))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive_B+ H_qbt_drive_A,
    qt.basis(total_truncation, hilbertspace.dressed_index((0,0,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)

plt.figure()
for idx, res in zip(idxs[:11], result.expect[:11]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium_A and Fluxonium_B) in state |1>")

plt.figure()
for idx, res in zip(idxs[:11], result2.expect[:11]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium_B) in state |0>")
