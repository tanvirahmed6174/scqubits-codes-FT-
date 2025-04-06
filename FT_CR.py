# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:38:20 2024

@author: WANGLAB
"""

#%% Transmon EC, EJ calculation
import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


target_01 = 4.099
target_02 = 3.969

qbtb = scq.Transmon(
     EJ=9,
     EC=0.26,
     ng=0,
     ncut=110,
     truncated_dim=10)

bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]

alpha = (bare_states_b[1]) - (bare_states_b[2]-bare_states_b[1])
qubit_01 = bare_states_b[1]
qubit_02 = bare_states_b[2]/2

#%%

import scqubits as scq
import numpy as np
from scipy.optimize import minimize

# Target frequencies
TARGET_01 = 4.099
TARGET_02 = 3.969

# Define the cost function to minimize
def cost_function(params):
    EJ, EC = params  # Extract parameters to optimize

    # Create the transmon with current parameters
    qbtb = scq.Transmon(
        EJ=EJ,
        EC=EC,
        ng=0,
        ncut=110,
        truncated_dim=10
    )

    # Calculate the eigenvalues
    bare_states_b = qbtb.eigenvals() - qbtb.eigenvals()[0]

    # Calculate qubit parameters
    qubit_01 = bare_states_b[1]  # 01 transition
    qubit_02 = bare_states_b[2] / 2  # 02 transition

    # Error between calculated and target values
    error_01 = (qubit_01 - TARGET_01) ** 2
    error_02 = (qubit_02 - TARGET_02) ** 2

    return error_01 + error_02  # Total error

# Initial guesses for EJ and EC
initial_guess = [10.5, 0.2247]

# Perform optimization
result = minimize(cost_function, initial_guess, method='Nelder-Mead', options={'disp': True})

# Extract the optimized parameters
optimized_EJ, optimized_EC = result.x

# Print the results
print("Optimized EJ:", optimized_EJ)
print("Optimized EC:", optimized_EC)

# Verify the optimized results
qbtb_optimized = scq.Transmon(
    EJ=optimized_EJ,
    EC=optimized_EC,
    ng=0,
    ncut=110,
    truncated_dim=10
)
bare_states_b_optimized = qbtb_optimized.eigenvals() - qbtb_optimized.eigenvals()[0]
optimized_qubit_01 = bare_states_b_optimized[1]
optimized_qubit_02 = bare_states_b_optimized[2] / 2

print("Optimized qubit 01 frequency:", optimized_qubit_01)
print("Optimized qubit 02 frequency:", optimized_qubit_02)



#%%
from scipy.optimize import minimize
import scqubits as scq

# Target values for alpha and qubit_01
target_alpha = 0.23
target_qubit_01 = 4.099

# Define the cost function to minimize
def cost_function(params):
    EJ, EC = params
    # Create a Transmon object with given EJ and EC
    qbtb = scq.Transmon(
        EJ=EJ,
        EC=EC,
        ng=0,
        ncut=110,
        truncated_dim=10
    )
    
    # Calculate bare states
    bare_states_b = qbtb.eigenvals() - qbtb.eigenvals()[0]
    
    # Calculate alpha and qubit_01
    alpha = bare_states_b[1] - (bare_states_b[2] - bare_states_b[1])
    qubit_01 = bare_states_b[1]
    
    # Calculate the squared error
    error_alpha = (alpha - target_alpha) ** 2
    error_qubit_01 = (qubit_01 - target_qubit_01) ** 2
    
    # Return the total error
    return error_alpha + error_qubit_01

# Define bounds and initial guess
bounds = [(8, 15), (0.19, 0.28)]
initial_guess = [10, 0.2]

# Modify the bounds for Nelder-Mead (not directly supported)
def constrained_cost_function(params):
    EJ, EC = params
    if not (8 <= EJ <= 15 and 0.18 <= EC <= 0.28):
        return 1e6  # Large penalty for out-of-bounds values
    return cost_function(params)

# Perform optimization using Nelder-Mead
result = minimize(constrained_cost_function, initial_guess, method="Nelder-Mead", options={"disp": True, "maxiter": 1000, "xatol": 1e-5, "fatol": 1e-5})

# Extract the optimal EJ and EC
optimal_EJ, optimal_EC = result.x

print(f"Optimal EJ: {optimal_EJ:.4f}")
print(f"Optimal EC: {optimal_EC:.4f}")



#%%
import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


qbta = scq.Fluxonium(
    EC=0.91,
    EJ = 3.955 ,
    EL=0.36,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ=11,
     EC=0.1952,
     ng=0,
     ncut=110,
     truncated_dim=10)


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
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation = 50

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
product_states_unsorted = [(0, 0), (1, 0), (0, 1),(2,0), (1, 1),(0,3) , (2,1),(0,2),(3,0),(1,2)]#,(4,0),(1,2),(3,1),(2,2),(5,0),(4,1),(3,2),(0,4),(1,4),(2,3),(1,3)]

idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]

paired_data = list(zip(idxs_unsorted, product_states_unsorted))
sorted_data = sorted(paired_data, key=lambda x: x[0])
product_states = [data[1] for data in sorted_data]
idxs = [data[0] for data in sorted_data]
#sort after writing, paired data sort
for idx, state in zip(idxs, product_states):
    print(f"{idx} -> {state}")
    
    
states = [qt.basis(total_truncation, idx) for idx in idxs]

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]

index_to_state = {idx: f'{state[0]}{state[1]}' for idx, state in zip(idxs, product_states)}


e_11 = evalues[hilbertspace.dressed_index((1,1))]
e_10 = evalues[hilbertspace.dressed_index((1,0))]
e_01 = evalues[hilbertspace.dressed_index((0,1))]
e_00 = evalues[hilbertspace.dressed_index((0,0))]
e_20 = evalues[hilbertspace.dressed_index((2,0))]
e_30 = evalues[hilbertspace.dressed_index((3,0))]
e_02 = evalues[hilbertspace.dressed_index((0,2))]


Static_ZZ = e_11-e_10-e_01+e_00


print('Static_ZZ(MHz)=',Static_ZZ, 'bare_F_01 = ',bare_states_a[1],'bare_F_12 =',bare_states_a[2]-bare_states_a[1], \
      'bare_T_01=',bare_states_b[1],'bare_F_03=',bare_states_a[3],'bare_T_12 =',bare_states_b[2]-bare_states_b[1])


drive_freq = e_11-e_10
A=.22*.3
def cosine_drive(t: float, args: dict) -> float:
    return A *np.cos(6.28*drive_freq* t)

n_a_00_01 = n_a[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]
n_b_00_01 = n_b[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]

eta = -n_a_00_01/n_b_00_01

print('Static_ZZ(MHz)= ',(e_11-e_10-e_01+e_00)*1e3)
print('dressed_F_01(GHz)= ',(e_10-e_00)*1)
print('dressed_F_12(GHz)= ',(e_20-e_10)*1)
print('dressed_F_03(GHz)= ',(e_30-e_00)*1)
print('dressed_T_01(GHz)= ',(e_01-e_00)*1)
print('dressed_T_12(GHz)= ',(e_02-e_01)*1)
print('Transmon alpha= ',(2*e_01-e_02)*1)



tlist = np.linspace(0, 400, 400)
H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a+eta*n_b), cosine_drive],  
]


result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((1,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((0,0))),
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


#%% AC stark shift calculation

# e_11 = evalues[hilbertspace.dressed_index((1,1))]
# e_12 = evalues[hilbertspace.dressed_index((1,2))]
# e_10 = evalues[hilbertspace.dressed_index((1,0))]
# e_01 = evalues[hilbertspace.dressed_index((0,1))]
# e_00 = evalues[hilbertspace.dressed_index((0,0))]
# e_20 = evalues[hilbertspace.dressed_index((2,0))]
# e_30 = evalues[hilbertspace.dressed_index((3,0))]
# e_02 = evalues[hilbertspace.dressed_index((0,2))]
# e_21 = evalues[hilbertspace.dressed_index((2,1))]


# Delta_1 = (e_12-e_11) - (e_02-e_01)

# Δ = evalues[d(i, j)] - evalues[d(k, l)]
# print(f"|12>-|11>: {e_12-e_11:.3f} |02>-|01>: {e_02-e_01:.3f}")
# print(f"Delta_1 = {Delta_1*1000:.3f} MHz")


def stark_shift(Omega, delta):

    return (np.sqrt(Omega**2 + delta**2) - delta) / 2
    # return Omega**2/delta

# Omega_11_12 = 

def d(i,j):
    return hilbertspace.dressed_index((i,j))
def b(i):
    return hilbertspace.bare_index(i)


# Δ = (evalues[d(1, 2)] - evalues[d(1, 1)]) - (evalues[d(0, 2)] - evalues[d(0, 1)])



x=1
y=100

def Omega(i,j,k,l):
    Omega = 5
    eps_a =x*Omega 
    eps_b = y*Omega
    # print('n_b:',abs(n_b[d(i,j),d(k,l)]))
    return abs(eps_a*n_a[d(i,j),d(k,l)]+ eps_b*n_b[d(i,j),d(k,l)])

delta = 50.0  # Example detuning in MHz

Δ3 = (evalues[d(3, 1)] - evalues[d(0, 1)]) - (evalues[d(3, 0)] - evalues[d(0, 0)])

# Assuming stark_shift and Omega are defined functions, and delta, Δ3 are defined variables
result = stark_shift(Omega(3, 1, 0, 1), delta)
print(f"Stark Shift |31>- |01>: {result:.3f} MHz")

result2 = stark_shift(Omega(3, 0, 0, 0), delta + Δ3 * 1e3)
print(f"Stark Shift |30>- |00>: {result2:.3f} MHz")

print(f"Omega_31_01: {Omega(3, 1, 0, 1):.3f} Omega_30_00: {Omega(3, 0, 0, 0):.3f}")
print(f"induced ZZ: {result - result2:.3f}")

#%%
Δ2 = (evalues[d(1, 2)] - evalues[d(1, 1)]) - (evalues[d(0, 2)] - evalues[d(0, 1)])

# Assuming stark_shift and Omega are defined functions, and delta, Δ3 are defined variables
result = stark_shift(Omega(1, 2, 1, 1), delta)
print(f"Stark Shift |31>- |01>: {result:.3f} MHz")

result2 = stark_shift(Omega(0, 2, 0, 1), delta + Δ2 * 1e3)
print(f"Stark Shift |30>- |00>: {result2:.3f} MHz")

print(f"Omega_31_01: {Omega(3, 1, 0, 1):.3f} Omega_30_00: {Omega(3, 0, 0, 0):.3f}")
print(f"induced ZZ: {result - result2:.3f}")

#%% not working
import numpy as np
import matplotlib.pyplot as plt

# Assuming stark_shift, Omega, delta, and Δ3 are already defined as in your provided code

# Define ranges for x and y
x_values = np.linspace(0, 50, 100)
y_values = np.linspace(0, 50, 100)
X, Y = np.meshgrid(x_values, y_values)

# Calculate induced ZZ for varying x and y
induced_ZZ = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        Omega_31_01 = Omega(3, 1, 0, 1)
        Omega_30_00 = Omega(3, 0, 0, 0)
        result = stark_shift(Omega_31_01, delta)
        result2 = stark_shift(Omega_30_00, delta + Δ3 * 1e3)
        induced_ZZ[i, j] = result - result2

# Create contour plot for induced ZZ
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, induced_ZZ, levels=50, cmap="viridis")
plt.colorbar(contour, label="Induced ZZ (MHz)")
plt.xlabel("x= eps_a/50")
plt.ylabel("y= eps_b/50")
plt.title("Contour Plot of Induced ZZ")
plt.show()



#%% Write all transitions



indices = range(4)  # {0, 1, 2, 3}
computed_pairs = set()  # To track processed pairs and avoid redundancies
omega_values = []  # To store Omega values with their identifiers

# Loop to calculate unique Omega_ij_kl without i >= j condition
for i in indices:
    for j in indices:
        for k in indices:
            for l in indices:
                # Skip if i~k=2 or j~l=2 (not allowed transitions)
                if abs(i - k) == 2 or abs(j - l) == 2:
                    continue
                if (i, j) != (k, l):  # Ensure (i, j) != (k, l)
                    # Create sorted pair to avoid redundancies
                    pair = tuple(sorted(((i, j), (k, l))))
                    if pair not in computed_pairs:
                        # Calculate Omega and take the absolute value
                        Omega = abs(evalues[d(i, j)] - evalues[d(k, l)])
                        
                        # Calculate n_b and n_a values (absolute values)
                        n_b_value = abs(n_b[d(i, j), d(k, l)])
                        n_a_value = abs(n_a[d(i, j), d(k, l)])
                        
                        # Append the result to the list
                        omega_values.append((f"Ω_{i}{j}_{k}{l}", Omega, n_b_value, n_a_value))
                        
                        # Add the pair to the set
                        computed_pairs.add(pair)

# Sort the Omega values by their magnitudes
omega_values.sort(key=lambda x: x[1])

# Print the sorted Omega values along with absolute n_b and n_a values (formatted to 4 decimal places)
for name, value, n_b_val, n_a_val in omega_values:
    print(f"{name} = {value:.4f}, n_a = {n_a_val:.4f}, n_b = {n_b_val:.4f}")


#%%


Δ2 = (evalues[d(1, 2)] - evalues[d(1, 1)]) - (evalues[d(0, 2)] - evalues[d(0, 1)])
Δ1 = (evalues[d(2, 1)] - evalues[d(1, 1)]) - (evalues[d(2, 0)] - evalues[d(1, 0)])

Δ3 = (evalues[d(3, 1)] - evalues[d(0, 1)]) - (evalues[d(3, 0)] - evalues[d(0, 0)])
Δ4 = (evalues[d(1, 3)] - evalues[d(1, 0)]) - (evalues[d(0, 3)] - evalues[d(0, 0)])

#Δ4, Δ3 not good enough

#%%
def Ω(i, j, k, l, eps_a, eps_b):
    Omega = 50
    # Use provided eps_a and eps_b
    return abs(eps_a * n_a[d(i, j), d(k, l)] + eps_b * n_b[d(i, j), d(k, l)])

eps_a =1
eps_b =1

#%%

#|21>-|11>, |20>, |10>  extending drive

x=10
y=4

def Omega(i,j,k,l):
    Omega = 20
    eps_a =42 
    eps_b = 9
    # print('n_b:',abs(n_b[d(i,j),d(k,l)]))
    return abs(eps_a*n_a[d(i,j),d(k,l)]+ eps_b*n_b[d(i,j),d(k,l)])

# n_a[d(i,j),d(k,l)]
print(f"n_a|21>- |11>: {abs(n_a[d(2,1),d(1,1)]):.4f}")


delta1 = -50-44  # Example detuning in MHz

Δ1 = (evalues[d(2, 1)] - evalues[d(1, 1)]) - (evalues[d(2, 0)] - evalues[d(1, 0)])

# Assuming stark_shift and Omega are defined functions, and delta, Δ3 are defined variables
result1 = stark_shift(Omega(2, 1, 1, 1), delta1)
# print(f"Rabi rate |21>- |11>: {Omega(2, 1, 1, 1):.3f} MHz")

print(f"Stark Shift |21>- |11>: {result1:.3f} MHz")

result2 = stark_shift(Omega(2, 0, 1, 0), delta1 - Δ1 * 1e3)

print(f"Stark Shift |20>- |10>: {result2:.3f} MHz")

print(f"Omega_21_11: {Omega(2, 1, 1, 1):.3f} Omega_20_10: {Omega(2, 0, 1, 0):.3f}")
print(f"induced ZZ: {result1 - result2:.3f}")



#|01>-|02>, |11>, |12>

# x=1
# y=-1

# def Omega(i,j,k,l):
#     Omega = 100
#     eps_a =x*Omega 
#     eps_b = y*Omega
#     # print('n_b:',abs(n_b[d(i,j),d(k,l)]))
#     return abs(eps_a*n_a[d(i,j),d(k,l)]+ eps_b*n_b[d(i,j),d(k,l)])

delta2 = -50  # Example detuning in MHz

Δ2 = (evalues[d(1, 2)] - evalues[d(1, 1)]) - (evalues[d(0, 2)] - evalues[d(0, 1)])

# Assuming stark_shift and Omega are defined functions, and delta, Δ3 are defined variables
result3 = stark_shift(Omega(1, 2, 1, 1), delta2)
# print(f"Rabi rate |21>- |11>: {Omega(2, 1, 1, 1):.3f} MHz")

print(f"Stark Shift |12>- |11>: {result3:.3f} MHz")

result4 = stark_shift(Omega(0, 2, 0, 1), delta2 - Δ2 * 1e3)

print(f"Stark Shift |02>- |01>: {result4:.3f} MHz")

print(f"Omega_12_11: {Omega(1, 2, 1, 1):.3f} Omega_02_01: {Omega(0, 2, 0, 1):.3f}")
print(f"induced ZZ: {result3 - result4:.3f}")


print(f"combined induced ZZ: {(result3 - result4) + (result1 - result2):.3f}")


#%% Stark shift calculation for single freqeuncy drive


# Define the range of x, y, and Omega
x_values = np.linspace(1, 36 , 20)/2  # x varies from 1 to 10
y_values = np.linspace(-18, 18, 20)/2  # y varies from -10 to 10
Omega_values = [1,1]#, 10, 15, 20]  # Different Omega values
detuning = 20

def stark_shift(Omega, delta):

    # return (np.sqrt(Omega**2 + delta**2) - delta) / 2
    return Omega**2/delta

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
# Initialize results for combined induced ZZ for each Omega
results_zz = {Omega: np.zeros((len(x_values), len(y_values))) for Omega in Omega_values}
sign = 1 # w_q-w_d
# Loop through x, y, and Omega values to calculate combined induced ZZ
for Omega in Omega_values:
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            eps_a = x
            eps_b = y
            
            # Calculate Omega for transitions
            omega_21_11 = abs(eps_a * n_a[d(2, 1), d(1, 1)] + eps_b * n_b[d(2, 1), d(1, 1)])
            omega_20_10 = abs(eps_a * n_a[d(2, 0), d(1, 0)] + eps_b * n_b[d(2, 0), d(1, 0)])
            omega_12_11 = abs(eps_a * n_a[d(1, 2), d(1, 1)] + eps_b * n_b[d(1, 2), d(1, 1)])
            omega_02_01 = abs(eps_a * n_a[d(0, 2), d(0, 1)] + eps_b * n_b[d(0, 2), d(0, 1)])
            
            # Calculate detunings
            delta1 = detuning + (1-sign)/2*42  # MHz
            delta2 = detuning + (1+sign)/2*42
            Δ1 = (evalues[d(2, 1)] - evalues[d(1, 1)]) - (evalues[d(2, 0)] - evalues[d(1, 0)])
            Δ2 = (evalues[d(1, 2)] - evalues[d(1, 1)]) - (evalues[d(0, 2)] - evalues[d(0, 1)])
            
            # Calculate Stark shifts
            stark_21_11 = stark_shift(omega_21_11, delta1)
            stark_20_10 = stark_shift(omega_20_10, delta1 - sign* Δ1 * 1e3)
            stark_12_11 = stark_shift(omega_12_11, delta2)
            stark_02_01 = stark_shift(omega_02_01, delta2 - sign*Δ2 * 1e3)
            
            # Combined induced ZZ
            combined_zz = (stark_21_11 - stark_20_10) + (stark_12_11 - stark_02_01)
            results_zz[Omega][i, j] = combined_zz

# Create contour plots for combined induced ZZ for each Omega value
plt.figure()
fig, axes = plt.subplots(1, len(Omega_values), figsize=(20, 5), constrained_layout=True)

for ax, Omega in zip(axes, Omega_values):
    X, Y = np.meshgrid(y_values, x_values)
    contour = ax.contourf(X, Y, results_zz[Omega], levels=20, cmap='viridis')
    ax.set_title(f"eps_a =1 : {eps_a} eps_b =1 : |{eps_b}|")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    fig.colorbar(contour, ax=ax, shrink=0.6)

plt.suptitle("Contour Plots of Combined Induced ZZ for Different Omega Values at detuning: {delta1:.3f}")
plt.show()



#%%
#|21>-|11>, |20>, |10>  squeezing drive

x=10
y=4

def Omega(i,j,k,l):
    Omega = 10
    eps_a =21 
    eps_b = 9
    # print('n_b:',abs(n_b[d(i,j),d(k,l)]))
    return abs(eps_a*n_a[d(i,j),d(k,l)]+ eps_b*n_b[d(i,j),d(k,l)])

# n_a[d(i,j),d(k,l)]
print(f"n_a|21>- |11>: {abs(n_a[d(2,1),d(1,1)]):.4f}")

# wd = 
delta1 = 44+50  # Example detuning in MHz

# sign = 

Δ1 = (evalues[d(2, 1)] - evalues[d(1, 1)]) - (evalues[d(2, 0)] - evalues[d(1, 0)])

# Assuming stark_shift and Omega are defined functions, and delta, Δ3 are defined variables
result1 = stark_shift(Omega(2, 1, 1, 1), delta1)
# print(f"Rabi rate |21>- |11>: {Omega(2, 1, 1, 1):.3f} MHz")

print(f"Stark Shift |21>- |11>: {result1:.3f} MHz")

result2 = stark_shift(Omega(2, 0, 1, 0), delta1 + Δ1 * 1e3)

print(f"Stark Shift |20>- |10>: {result2:.3f} MHz")

print(f"Omega_21_11: {Omega(2, 1, 1, 1):.3f} Omega_20_10: {Omega(2, 0, 1, 0):.3f}")
print(f"induced ZZ: {result1 - result2:.3f}")



#|01>-|02>, |11>, |12>

# x=1
# y=-1

# def Omega(i,j,k,l):
#     Omega = 100
#     eps_a =x*Omega 
#     eps_b = y*Omega
#     # print('n_b:',abs(n_b[d(i,j),d(k,l)]))
#     return abs(eps_a*n_a[d(i,j),d(k,l)]+ eps_b*n_b[d(i,j),d(k,l)])

delta2 = 50  # Example detuning in MHz

Δ2 = (evalues[d(1, 2)] - evalues[d(1, 1)]) - (evalues[d(0, 2)] - evalues[d(0, 1)])

# Assuming stark_shift and Omega are defined functions, and delta, Δ3 are defined variables
result3 = stark_shift(Omega(1, 2, 1, 1), delta2)
# print(f"Rabi rate |21>- |11>: {Omega(2, 1, 1, 1):.3f} MHz")

print(f"Stark Shift |12>- |11>: {result3:.3f} MHz")

result4 = stark_shift(Omega(0, 2, 0, 1), delta2 + Δ2 * 1e3)

print(f"Stark Shift |02>- |01>: {result4:.3f} MHz")

print(f"Omega_12_11: {Omega(1, 2, 1, 1):.3f} Omega_02_01: {Omega(0, 2, 0, 1):.3f}")
print(f"induced ZZ: {result3 - result4:.3f}")


print(f"combined induced ZZ: {(result3 - result4) + (result1 - result2):.3f}")


#%% 
import numpy as np
eta = 0.2e3
Omega = 10
Delta = 50

stark1 = (eta*Omega**2)/(2*Delta*(Delta+eta))
stark2 =  (np.sqrt(Omega**2+Delta**2)-Delta)/2
stark3 = Omega**2/Delta


#%% Same as previous code but with Gaussian drive

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt

# Define the qubits
qbta = scq.Fluxonium(
    EC=0.91,
    EJ=3.955,
    EL=0.36,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)

qbtb = scq.Transmon(
    EJ=11,
    EC=0.1952,
    ng=0,
    ncut=110,
    truncated_dim=10,
)

# Define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb])

# Add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.024,
    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)

# Generate spectrum lookup table
hilbertspace.generate_lookup()

# Hamiltonian in dressed eigenbasis
(evals,) = hilbertspace["evals"]
diag_dressed_hamiltonian = (
    2 * np.pi * qt.Qobj(np.diag(evals), dims=[hilbertspace.subsystem_dims] * 2)
)

# Truncate operators
total_truncation = 50

def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

# Define DRAG pulse parameters
alpha_DRAG = 0.1  # DRAG coefficient
A = 0.22 * 0.3  # Amplitude scaling
t_rise = 50  # Time rise in ns
sigma = 20  # Width of Gaussian in ns

# Define Gaussian envelope and DRAG derivative
def gaussian_envelope(t, t_rise, sigma, drive_freq):
    return A * np.cos(2 * np.pi * drive_freq * t) * np.exp(-(t - t_rise) ** 2 / (2 * sigma ** 2))

def drag_derivative(t, t_rise, sigma, drive_freq):
    return -A * (t - t_rise) / (sigma ** 2) * np.cos(2 * np.pi * drive_freq * t) * np.exp(-(t - t_rise) ** 2 / (2 * sigma ** 2))

# Combine the DRAG pulse
def drag_pulse(t, drive_freq, t_rise, sigma):
    g_x = gaussian_envelope(t, t_rise, sigma, drive_freq)
    g_y = alpha_DRAG * drag_derivative(t, t_rise, sigma, drive_freq)
    return g_x + 1j * g_y

# Add the driving Hamiltonian with the DRAG pulse
n_a = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbta.n_operator)
n_b = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtb.n_operator)
n_a = truncate(n_a, total_truncation)
n_b = truncate(n_b, total_truncation)

drive_freq = (evals[1] - evals[0])  # Drive frequency matching qubit transition
H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a + n_b), lambda t, args: drag_pulse(t, drive_freq, t_rise, sigma).real],
]

# Simulate the evolution
tlist = np.linspace(0, 400, 400)
result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((1, 0))),
    tlist,
    e_ops=[state * state.dag() for state in [qt.basis(total_truncation, i) for i in range(5)]],
)
result2 = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((0, 0))),
    tlist,
    e_ops=[state * state.dag() for state in [qt.basis(total_truncation, i) for i in range(5)]],
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

#%% with DRAG pulse

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from qutip.qip.operations import rz, cnot


qbta = scq.Fluxonium(
    EC=0.91,
    EJ = 3.955 ,
    EL=0.36,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ=11,
     EC=0.1952,
     ng=0,
     ncut=110,
     truncated_dim=10)


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
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation = 50

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
product_states_unsorted = [(0, 0), (1, 0), (0, 1),(2,0), (1, 1),(0,3) , (2,1),(0,2),(3,0),(1,2)]#,(4,0),(1,2),(3,1),(2,2),(5,0),(4,1),(3,2),(0,4),(1,4),(2,3),(1,3)]

idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]

paired_data = list(zip(idxs_unsorted, product_states_unsorted))
sorted_data = sorted(paired_data, key=lambda x: x[0])
product_states = [data[1] for data in sorted_data]
idxs = [data[0] for data in sorted_data]
#sort after writing, paired data sort
for idx, state in zip(idxs, product_states):
    print(f"{idx} -> {state}")
    
    
states = [qt.basis(total_truncation, idx) for idx in idxs]

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]

index_to_state = {idx: f'{state[0]}{state[1]}' for idx, state in zip(idxs, product_states)}


e_11 = evalues[hilbertspace.dressed_index((1,1))]
e_10 = evalues[hilbertspace.dressed_index((1,0))]
e_01 = evalues[hilbertspace.dressed_index((0,1))]
e_00 = evalues[hilbertspace.dressed_index((0,0))]
e_20 = evalues[hilbertspace.dressed_index((2,0))]
e_30 = evalues[hilbertspace.dressed_index((3,0))]
e_02 = evalues[hilbertspace.dressed_index((0,2))]


Static_ZZ = e_11-e_10-e_01+e_00


print('Static_ZZ(MHz)=',Static_ZZ, 'bare_F_01 = ',bare_states_a[1],'bare_F_12 =',bare_states_a[2]-bare_states_a[1], \
      'bare_T_01=',bare_states_b[1],'bare_F_03=',bare_states_a[3],'bare_T_12 =',bare_states_b[2]-bare_states_b[1])




# Define DRAG pulse parameters
alpha_DRAG = 0.1  # DRAG coefficient
A = 0.22   # Amplitude scaling
t_rise = 50  # Time rise in ns
sigma = 20*6  # Width of Gaussian in ns

# Define Gaussian envelope and DRAG derivative
def gaussian_envelope(t, t_rise, sigma, drive_freq):
    return A * np.cos(2 * np.pi * drive_freq * t) * np.exp(-(t - t_rise) ** 2 / (2 * sigma ** 2))

def drag_derivative(t, t_rise, sigma, drive_freq):
    return -A * (t - t_rise) / (sigma ** 2) * np.cos(2 * np.pi * drive_freq * t) * np.exp(-(t - t_rise) ** 2 / (2 * sigma ** 2))

# Combine the DRAG pulse
def drag_pulse(t, drive_freq, t_rise, sigma):
    g_x = gaussian_envelope(t, t_rise, sigma, drive_freq)
    g_y = alpha_DRAG * drag_derivative(t, t_rise, sigma, drive_freq)
    return g_x + 1j * g_y
    
    
    
    
    
    
drive_freq = e_01-e_00
A=.22*.3
def cosine_drive(t: float, args: dict) -> float:
    return A *np.cos(6.28*drive_freq* t)

n_a_00_01 = n_a[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]
n_b_00_01 = n_b[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]

eta = -n_a_00_01/n_b_00_01

print('Static_ZZ(MHz)= ',(e_11-e_10-e_01+e_00)*1e3)
print('dressed_F_01(GHz)= ',(e_10-e_00)*1)
print('dressed_F_12(GHz)= ',(e_20-e_10)*1)
print('dressed_F_03(GHz)= ',(e_30-e_00)*1)
print('dressed_T_01(GHz)= ',(e_01-e_00)*1)
print('dressed_T_12(GHz)= ',(e_02-e_01)*1)
print('Transmon alpha= ',(2*e_01-e_02)*1)



tlist = np.linspace(0, 400, 400)
H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a+eta*n_b), lambda t, args: drag_pulse(t, drive_freq, t_rise, sigma).real],  
]


result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((1,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((0,0))),
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


computational_indices = [0, 1, 3, 4]
computational_subspace = [states[i] for i in computational_indices]


# get the propagator at the final time step
prop = qt.propagator(H_qbt_drive, tlist)[-1]  

# truncate the propagator to the computational subspace
Uc = qt.Qobj(
    [
        [prop.matrix_element(s1, s2) for s1 in computational_subspace]
        for s2 in computational_subspace
    ]
)

# Factor global phase so that upper-left corner of matrix is real
def remove_global_phase(op):
    return op * np.exp(-1j * cmath.phase(op[0, 0]))

# The process for obtaining the Z rotations is taken from page 3 of Nesterov et al., at the
# bottom of the paragraph beginning, "To model gate operation..."
def dphi(state):
    return -np.angle(prop.matrix_element(state, state)) + np.angle(
        prop.matrix_element(states[0], states[0])
    )

# product of single-qubit Z-gates
Uz = remove_global_phase(qt.tensor(rz(dphi(states[2])), rz(dphi(states[1]))))
Uc_reshaped = qt.Qobj(Uc.data, dims=[[2, 2], [2, 2]])
Ucprime = remove_global_phase(Uz * Uc_reshaped)
Ucprime  # result should be close to diag(1,1,1,-1)

#fidelity measure given on page 3 of Nesterov et al.
((Ucprime.dag() * Ucprime).tr() + np.abs((Ucprime.dag() * cnot()).tr()) ** 2) / 20


#%%

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from qutip.qip.operations import rz, cz_gate
import cmath

# experimental values borrowed from
# [https://arxiv.org/abs/1802.03095]
# define fluxonium A
qbta = scq.Fluxonium(
    EC=1.5,
    EJ=5.5,
    EL=1.0,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)

# define fluxonium B
qbtb = scq.Fluxonium(
    EC=1.2,
    EJ=5.7,
    EL=1.0,
    flux=0.5,
    cutoff=110,
    truncated_dim=10,
)

# define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb])

# add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=0.15,
    op1=qbta.n_operator,
    op2=qbtb.n_operator,
)

# generate spectrum lookup table
hilbertspace.generate_lookup()

# get the transition frequency between two states specified by dressed indices
def transition_frequency(s0: int, s1: int) -> float:
    return (
        (
            hilbertspace.energy_by_dressed_index(s1)
            - hilbertspace.energy_by_dressed_index(s0)
        )
        * 2
        * np.pi
    )

# The matrix representations can be truncated further for the simulation
total_truncation = 20

# truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

# get the representation of the n_b operator in the dressed eigenbasis of the composite system
n_b = hilbertspace.op_in_dressed_eigenbasis((qbtb.n_operator, qbtb))
# truncate the operator after expressing in the dressed basis to speed up the simulation
n_b = truncate(n_b, total_truncation)

# convert the product states to the closes eigenstates of the dressed system
product_states = [(0, 0), (1, 0), (0,1), (1, 1), (2, 1)]
idxs = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states]
states = [qt.basis(total_truncation, idx) for idx in idxs]

# The computational subspace is spanned by the first 4 states
computational_subspace = states[:4]


# get dressed state 11 to 12 transition frequency
omega_1112 = transition_frequency(idxs[3], idxs[4])

# Gaussian pulse parameters optimized by hand
A = 0.02417 # GHz
tg = 100 # ns

#Gaussian pulse envelope
def drive_coeff(t: float, args: dict) -> float:
    return A * np.exp(-8 * t * (t - tg) / tg**2) * np.cos(omega_1112 * t)

# Hamiltonian in dressed eigenbasis
(evals,) = hilbertspace["evals"]
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)
diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

# time-dependent drive Hamiltonian
H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * n_b, drive_coeff],  # driving through the resonator
]




# array of time list
tlist = np.linspace(0, 90, 200)  # total time

# This simulation is just for viewing the affect of the pulse
result = qt.sesolve(
    H_qbt_drive,
    qt.basis(20, hilbertspace.dressed_index(product_states[3])),
    tlist,
    e_ops=[state * state.dag() for state in states]
)


for idx, res in zip(idxs, result.expect):
    plt.plot(tlist, res, label=r"$|%u\rangle$" % (idx))


plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")

# get the propagator at the final time step
prop = qt.propagator(H_qbt_drive, tlist)[-1]  

# truncate the propagator to the computational subspace
Uc = qt.Qobj(
    [
        [prop.matrix_element(s1, s2) for s1 in computational_subspace]
        for s2 in computational_subspace
    ]
)

# Factor global phase so that upper-left corner of matrix is real
def remove_global_phase(op):
    return op * np.exp(-1j * cmath.phase(op[0, 0]))

# The process for obtaining the Z rotations is taken from page 3 of Nesterov et al., at the
# bottom of the paragraph beginning, "To model gate operation..."
def dphi(state):
    return -np.angle(prop.matrix_element(state, state)) + np.angle(
        prop.matrix_element(states[0], states[0])
    )

# product of single-qubit Z-gates
Uz = remove_global_phase(qt.tensor(rz(dphi(states[2])), rz(dphi(states[1]))))
Uc_reshaped = qt.Qobj(Uc.data, dims=[[2, 2], [2, 2]])
Ucprime = remove_global_phase(Uz * Uc_reshaped)
Ucprime  # result should be close to diag(1,1,1,-1)

#fidelity measure given on page 3 of Nesterov et al.
((Ucprime.dag() * Ucprime).tr() + np.abs((Ucprime.dag() * cz_gate()).tr()) ** 2) / 20

#%% trying Fidelity calculation for F-T system


import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


qbta = scq.Fluxonium(
    EC=0.91,
    EJ = 3.955 ,
    EL=0.36,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ=11,
     EC=0.1952,
     ng=0,
     ncut=110,
     truncated_dim=10)


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
# The factor of 2pi converts the energy to GHz so that the time is in units of ns
diag_dressed_hamiltonian = (
        2 * np.pi * qt.Qobj(np.diag(evals),
        dims=[hilbertspace.subsystem_dims] * 2)
)

# The matrix representations can be truncated further for the simulation
total_truncation = 20

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
product_states_unsorted = [(0, 0), (1, 0), (0, 1),(2,0), (1, 1),(0,3) , (2,1),(0,2),(3,0),(1,2)]#,(4,0),(1,2),(3,1),(2,2),(5,0),(4,1),(3,2),(0,4),(1,4),(2,3),(1,3)]

idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]

paired_data = list(zip(idxs_unsorted, product_states_unsorted))
sorted_data = sorted(paired_data, key=lambda x: x[0])
product_states = [data[1] for data in sorted_data]
idxs = [data[0] for data in sorted_data]
#sort after writing, paired data sort
for idx, state in zip(idxs, product_states):
    print(f"{idx} -> {state}")
    
    
states = [qt.basis(total_truncation, idx) for idx in idxs]

bare_states_a = qbta.eigenvals()-qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals()-qbtb.eigenvals()[0]

index_to_state = {idx: f'{state[0]}{state[1]}' for idx, state in zip(idxs, product_states)}

computational_indices = [0, 1, 3, 4]
computational_subspace = [states[i] for i in computational_indices]


e_11 = evalues[hilbertspace.dressed_index((1,1))]
e_10 = evalues[hilbertspace.dressed_index((1,0))]
e_01 = evalues[hilbertspace.dressed_index((0,1))]
e_00 = evalues[hilbertspace.dressed_index((0,0))]
e_20 = evalues[hilbertspace.dressed_index((2,0))]
e_30 = evalues[hilbertspace.dressed_index((3,0))]
e_02 = evalues[hilbertspace.dressed_index((0,2))]


drive_freq = e_01-e_00

# Gaussian pulse parameters optimized by hand
A = 0.02417*10 # GHz
tg = 300 # ns

#Gaussian pulse envelope
def drive_coeff(t: float, args: dict) -> float:
    return A * np.exp(-8 * t * (t - tg) / tg**2) * np.cos(drive_freq * t)

n_a_00_01 = n_a[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]
n_b_00_01 = n_b[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]

eta = -n_a_00_01/n_b_00_01


tlist = np.linspace(0, 400, 400)
# H_qbt_drive = [
#     diag_dressed_hamiltonian_trunc,
#     [2 * np.pi * (n_a+eta*n_b), lambda t, args: drag_pulse(t, drive_freq, t_rise, sigma).real],  
# ]

H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi *(n_a+eta*n_b*0), drive_coeff],  # driving through the resonator
]


result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((1,0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
#%%

A = 0.1 # Amplitude
drive_freq = e_01-e_00  # Example drive frequency in GHz
t_cos = 500  # Cosine drive active time in ns

# Define the cosine drive with a time-dependent envelope
def cosine_drive(t: float, args: dict) -> float:
    if t <= t_cos:
        return A * np.cos(2 * np.pi * drive_freq * t)
    else:
        return 0.0  # No drive after t_cos

# Parameters and Hamiltonian setup (assuming hilbertspace, n_a, n_b, etc. are defined)
n_a_00_01 = n_a[hilbertspace.dressed_index((0, 0)), hilbertspace.dressed_index((0, 1))]
n_b_00_01 = n_b[hilbertspace.dressed_index((0, 0)), hilbertspace.dressed_index((0, 1))]
eta = -n_a_00_01 / n_b_00_01

# Time list for the simulation
tlist = np.linspace(0, 600, 700)

# Define the time-dependent Hamiltonian
H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a + eta * n_b*0), cosine_drive],  # Time-dependent drive
]

# Solve the Schrödinger equation
result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((1, 0))),
    tlist,
    e_ops=[state * state.dag() for state in states],
)


plt.figure()
for idx, res in zip(idxs[:5], result.expect[:5]):
    plt.plot(tlist, res, label=f"|{index_to_state[idx]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium) in state |1>")