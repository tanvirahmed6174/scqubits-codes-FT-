# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:13:02 2025

@author: WANGLAB
"""

import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


qbta = scq.Fluxonium(
    EC=0.9,
    EJ = 6 ,
    EL=.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)

qbtb = scq.Transmon(
     EJ=4,
     EC=0.05,
     ng=0,
     ncut=110,
     truncated_dim=10,
     )

qbtc = scq.Fluxonium(
    EC=1,
    EJ = 4.5 ,
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
    
    
idxs = list(range(15))

# Generate product_states by applying hilbertspace.bare_index to each index
product_states = [hilbertspace.bare_index(i) for i in idxs]

# Print the results
for idx, state in zip(idxs, product_states):
    print(f"{idx} -> {state}")
    
    
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


#condition 1


n_a_000_010 = n_a[hilbertspace.dressed_index((0,0,0)),hilbertspace.dressed_index((0,1,0))]
n_b_000_010 = n_b[hilbertspace.dressed_index((0,0,0)),hilbertspace.dressed_index((0,1,0))]

eta = -n_a_000_010/n_b_000_010*0


drive_freq_A =evalues[hilbertspace.dressed_index((1,1,0))]- evalues[hilbertspace.dressed_index((1,0,0))]
A_A=.22*3
def cosine_drive(t: float, args: dict) -> float:
    return A_A *np.cos(6.28*drive_freq_A* t)
tlist = np.linspace(0, 400, 400)
H_qbt_drive_A = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a+eta*n_b), cosine_drive],  
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
for idx, res in zip(idxs[:10], result.expect[:10]):
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


#condition 2


drive_freq_B =evalues[hilbertspace.dressed_index((0,1,1))]- evalues[hilbertspace.dressed_index((0,0,1))]
A_B=.22*1.5
def cosine_drive(t: float, args: dict) -> float:
    return A_B *np.cos(6.28*drive_freq_B* t)
tlist = np.linspace(0, 400, 400)
H_qbt_drive_B = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_c+eta*n_b), cosine_drive],  
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


#condtion 3

drive_freq_B =evalues[hilbertspace.dressed_index((0,1,1))]- evalues[hilbertspace.dressed_index((0,0,1))]
A_B=.22*5
def cosine_drive(t: float, args: dict) -> float:
    return A_B *np.cos(6.28*drive_freq_B* t)
tlist = np.linspace(0, 400, 400)
H_qbt_drive_B = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_c+eta*n_b), cosine_drive],  
]

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


#%%


# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 12:13:02 2025

@author: WANGLAB
"""
import json

# Use a raw string or forward slashes to avoid path issues
file_path = r"Z:\Tanvir\Tanvir Files and codes\Codes_11_14_2024\ftf_parameters.json"

# Safely load the JSON file
try:
    with open(file_path, "r") as file:
        params = json.load(file)
        print("Parameters loaded successfully!")
        print(params)
except json.JSONDecodeError as e:
    print(f"Error loading JSON: {e}")
except OSError as e:
    print(f"Error accessing file: {e}")

#%%
import pandas as pd

# Creating a table similar to the provided image with values from the given code
data = {
    "Qubit Type": ["Fluxonium (A)", "Transmon (B)", "Fluxonium (C)"],
    "EC (GHz)": [0.9, 0.2, 1],
    "EJ (GHz)": [6, 15, 4.5],
    "EL (GHz)": [0.8, "N/A", 0.8],
    "Flux (Φ/Φ0)": [0.5, "N/A", 0.5],
    "Cutoff": [110, 110, 110],
    "Truncated Dim": [10, 10, 10],
    "Interaction Strength (g, MHz)": [20, 20, 20]
}

# Converting to a DataFrame
df = pd.DataFrame(data)

# Displaying the DataFrame to the user
import ace_tools as tools
tools.display_dataframe_to_user(name="Qubit Parameters Table", dataframe=df)

#%% shot noise dephasing

import numpy as np
import matplotlib.pyplot as plt

# Define constants
n_th = 1  # thermal photon number
chi = .25  # fixed chi value in MHz

# Kappa/Chi ratio values
kappa_chi_ratios = np.linspace(0.1, 10, 500)
kappa_values = kappa_chi_ratios * chi

# Calculate dephasing rate
gamma_phi_th = n_th * kappa_values * chi**2 / (kappa_values**2 + chi**2)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(kappa_chi_ratios, gamma_phi_th, label=r'$\Gamma_\phi^{th}$')
plt.axvline(1, color='r', linestyle='--', label='Threshold ($\kappa/\chi=1$)')
plt.xlabel(r'$\kappa/\chi$ Ratio')
plt.ylabel(r'Thermal Dephasing Rate $\Gamma_\phi^{th}$ (arbitrary units)')
plt.title(r'Photon Shot Noise-Induced Dephasing vs $\kappa/\chi$ Ratio')
plt.legend()
plt.grid(True)
plt.show()


#%%


#optimizer


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate
import time
import qutip as qtp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pysqkit
from pysqkit import QubitSystem, Qubit
from pysqkit.drives.pulse_shapes import gaussian_top
from pysqkit.util.metrics import average_process_fidelity, average_gate_fidelity
from pysqkit.util.phys import temperature_to_thermalenergy
import pysqkit.util.transformations as trf
from pysqkit.util.linalg import get_mat_elem
from pysqkit.solvers.solvkit import integrate
from pysqkit.util.hsbasis import pauli_by_index
from pysqkit.solvers import solvkit
from pysqkit.drives.pulse_shapes import gaussian_top
from typing import List, Dict, Callable
import matplotlib
import copy
import json
import cmath
import util_tf_cr

matplotlib.rcParams['mathtext.fontset'] = 'cm'

from IPython.display import Latex  # Unused, but imported in original

#############################################
# Cross-resonance gate between a transmon and a fluxonium
#############################################

# In this tutorial code, we study the cross-resonance two-qubit gate
# between a transmon and a fluxonium.

# The gate is based on the following driven Hamiltonian of a fluxonium
# and a transmon capacitively coupled:
#
#  H = H(0)_t + H(0)_f + V + H_drive
#

# We also consider the case in which we have relaxation due to dielectric loss.

# We import parameters from a json file:
with open('flx_transm_params.txt') as param_file:
    parameters_set = json.load(param_file)
print(parameters_set.keys())

temperature = 0.020  # K
thermal_energy = temperature_to_thermalenergy(temperature)  # kb T/h in GHz
d_comp = 4

p_set = "CR_Tanvir_type3"  # This parameter set corresponds to omega_t/2 pi = 4.7
# Other parameter sets: ['CR_1','CR_3','CR_4'] for omega_t/2 pi = 4.3, 5.3, 5.7

# Transmon
levels_t = 3
transm = pysqkit.qubits.SimpleTransmon(
    label='T',
    max_freq=parameters_set[p_set]["max_freq_t"],
    anharm=parameters_set[p_set]["anharm_t"],
    diel_loss_tan=parameters_set[p_set]["diel_loss_tan_t"],
    env_thermal_energy=thermal_energy,
    dim_hilbert=levels_t,
    dephasing_times=None
)

# Fluxonium
levels_f = 5
flx = pysqkit.qubits.Fluxonium(
    label='F',
    charge_energy=parameters_set[p_set]["charge_energy_f"],
    induct_energy=parameters_set[p_set]["induct_energy_f"],
    joseph_energy=parameters_set[p_set]["joseph_energy_f"],
    diel_loss_tan=parameters_set[p_set]["diel_loss_tan_f"],
    env_thermal_energy=thermal_energy,
    dephasing_times=None
)
flx.diagonalize_basis(levels_f)

# We add the drive on the fluxonium
flx.add_drive(
    pysqkit.drives.microwave_drive,
    label='cr_drive_f',
    pulse=pysqkit.drives.pulses.cos_modulation,
    pulse_shape=pysqkit.drives.pulse_shapes.gaussian_top
)

d_leak = levels_t * levels_f - d_comp

jc = parameters_set[p_set]["jc"]
coupled_sys = transm.couple_to(flx, coupling=pysqkit.couplers.capacitive_coupling, strength=jc)
bare_system = transm.couple_to(flx, coupling=pysqkit.couplers.capacitive_coupling, strength=0.0)

states_label = coupled_sys.all_state_labels()
states_dict = coupled_sys.states_as_dict(as_qobj=True)
flx_freq = flx.eig_energies(2)[1] - flx.eig_energies(2)[0]
flx_freq_03 = flx.eig_energies(4)[3] - flx.eig_energies(4)[0]
flx_freq_12 = flx.eig_energies(4)[2] - flx.eig_energies(4)[1]

state_label = ["00", "01", "10", "11"]
comp_states = {}
for label in state_label:
    state_tmp = coupled_sys.state(label)[1]
    loc = np.argmax(np.abs(state_tmp))
    phase = cmath.phase(state_tmp[loc])
    state_tmp = np.exp(-1j*phase) * state_tmp
    comp_states[label] = state_tmp

level_list = ['00', '01', '10', '11', '02', '20', '12','21', '03', '13', '04']
util_tf_cr.energy_levels_diagram(bare_system, level_list, show_drive=False)

#######################################################
# We look at the following quantities:
#   ζ_ZZ/2π  = (E00 + E11 - E01 - E10)/h
#   μ_YZ     = (|<10|qF|00> - <11|qF|01>|)*εd/2
#   μ_Y      = (|<10|qF|00> + <11|qF|01>|)*εd/2
#######################################################

def zz(system: QubitSystem) -> float:
    xi_zz = system.state('00')[0] + system.state('11')[0] \
            - system.state('01')[0] - system.state('10')[0]
    return xi_zz

def mu_yz_flx(
    comp_states: dict,
    op: np.ndarray,
    eps: float
) -> float:
    """
    Evaluates the CR coefficient numerically in the dressed basis
    when driving the fluxonium.
    """
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'])
    return (np.imag(yz0 - yz1))/2 * eps/2

def mu_zy_transm(
    comp_states: dict,
    op: np.ndarray,
    eps: float
) -> float:
    """
    Evaluates the CR coefficient numerically in the dressed basis
    when driving the transmon.
    """
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['01'])
    yz1 = get_mat_elem(op, comp_states['10'], comp_states['11'])
    return (np.imag(yz0 - yz1))/2

def mu_yi_flx(
    comp_states: dict,
    op: np.ndarray,
    eps: float
) -> float:
    """
    Evaluates the direct drive on the transmon numerically in the dressed basis
    when driving the fluxonium.
    """
    yz0 = get_mat_elem(op, comp_states['00'], comp_states['10'])
    yz1 = get_mat_elem(op, comp_states['01'], comp_states['11'])
    return (np.imag(yz0 + yz1))/2 * eps/2

def mu_yz_flx_sw(
    transm: Qubit,
    flx: Qubit,
    jc: float,
    eps: float
) -> float:
    """
    Evaluates the CR coefficient via the second-order Schrieffer-Wolff
    transformation.
    """
    q_zpf = transm.charge_zpf
    omega_t = transm.freq
    omega_flx, states_flx = flx.eig_states(4)
    omega_flx = omega_flx - omega_flx[0]
    q_10 = np.imag(get_mat_elem(flx.charge_op(), states_flx[1], states_flx[0]))
    q_21 = np.imag(get_mat_elem(flx.charge_op(), states_flx[2], states_flx[1]))
    q_30 = np.imag(get_mat_elem(flx.charge_op(), states_flx[3], states_flx[0]))

    coeff = q_21**2/(omega_flx[2] - (omega_t + omega_flx[1]))
    coeff += -q_30**2/(omega_flx[3] - omega_t)
    coeff += q_10**2/(omega_t - omega_flx[1])

    mu_yz = jc*q_zpf*coeff/2 * eps/2
    return mu_yz

eps_test = 0.3
q_op = coupled_sys["F"].charge_op()

#################################################
# Functions for time integration and gate times
#################################################

def func_to_minimize(
    pulse_time: list,
    t_rise: float,
    cr_coeff: float
) -> float:
    """
    Computes the difference between the time integral
    of the CR coefficient and the target phase π/4.
    """
    step = 1e-3
    n_points = int(pulse_time[0]/step)
    times = np.linspace(0, pulse_time[0], n_points)
    pulse = gaussian_top(times, t_rise, pulse_time[0])
    integral = scipy.integrate.simpson(2*np.pi*cr_coeff*pulse, times)
    return np.abs(integral - np.pi/4)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

eps = 0.5
omega_flx, states_flx = flx.eig_states(4)
op = coupled_sys["F"].charge_op()
freq_drive = transm.max_freq
omega_drive = np.abs(get_mat_elem(op, coupled_sys.state("01")[1], coupled_sys.state("11")[1]))
delta_drive = freq_drive - transm.max_freq
rabi_period = 1/np.sqrt(omega_drive**2 + delta_drive**2)
t_rise = 10.0  # [ns]

cr_coeff = np.abs(util_tf_cr.mu_yz_flx(comp_states, op, eps))
t_gate_0 = [util_tf_cr.cr_gate_time(cr_coeff)]
args_to_pass = (t_rise, cr_coeff)

start = time.time()
minimization_result = minimize(func_to_minimize, t_gate_0, args=args_to_pass)
end = time.time()

t_gate = minimization_result['x'][0]
print(minimization_result)
print("t_gate: {} ns".format(t_gate))

pts_per_drive_period = 10
nb_points = int(t_gate * freq_drive * pts_per_drive_period)
tlist = np.linspace(0, t_gate, nb_points)

coupled_sys['F'].drives['cr_drive_f'].set_params(
    phase=0, 
    time=tlist, 
    rise_time=t_rise, 
    pulse_time=t_gate,
    amp=eps, 
    freq=freq_drive
)

simu_opt = qtp.solver.Options()
simu_opt.atol = 1e-14
simu_opt.rtol = 1e-12

env_syst = pysqkit.tomography.TomoEnv(
    system=coupled_sys, 
    time=2*np.pi*tlist, 
    options=simu_opt, 
    with_noise=False, 
    dressed_noise=False
)

#########################################################
# Obtain the superoperator in the computational subspace
#########################################################

comp_states_list = []
for key in comp_states.keys():
    comp_states_list.append(comp_states[key])

n_process = 4
my_hs_basis = pauli_by_index

start = time.time()
sup_op = env_syst.to_super(comp_states_list, my_hs_basis, n_process, speed_up=True)
end = time.time()
print("Computational time = " + str(end - start) + ' s')

#########################################################
# Single-qubit Z corrections
#########################################################

sq_corr = util_tf_cr.single_qubit_corrections(sup_op, my_hs_basis)
sq_corr_sup = trf.kraus_to_super(sq_corr, my_hs_basis)
total_sup_op = sq_corr_sup.dot(sup_op)

#########################################################
# Build target CR gate
#########################################################

def cry(theta):
    ide = np.identity(4)
    yz = np.kron(np.array([[0, -1j], [1j, 0]]),
                  np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*yz

def crx(theta):
    ide = np.identity(4)
    zx = np.kron(np.array([[0, 1], [1, 0]]),
                  np.array([[1, 0], [0, -1]]))
    return np.cos(theta/2)*ide - 1j*np.sin(theta/2)*zx

#########################################################
# Process fidelity (compare to CR(-pi/2))
#########################################################

cr_super_target = trf.kraus_to_super(cry(-np.pi/2), my_hs_basis)
f_pro = average_process_fidelity(cr_super_target, total_sup_op)
print("Process fidelity =", f_pro)

#########################################################
# Virtual correction for the extra Y rotation on transmon
#########################################################

def ry_t(theta):
    rot_y = (np.cos(theta/2)*np.identity(2) -
             1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]]))
    return np.kron(rot_y, np.identity(2))

def ry_f(theta):
    rot_y = (np.cos(theta/2)*np.identity(2) -
             1j*np.sin(theta/2)*np.array([[0, -1j], [1j, 0]]))
    return np.kron(np.identity(2), rot_y)

theta_list = list(np.linspace(0, 2*np.pi, 100))
fid_list_ry = []

for theta in theta_list:
    rot_y_super = trf.kraus_to_super(ry_t(theta), my_hs_basis)
    fid_list_ry.append(
        average_process_fidelity(cr_super_target, rot_y_super.dot(total_sup_op))
    )

fid_ry = np.array(fid_list_ry)
max_fid = np.max(fid_ry)
max_index = np.argmax(fid_ry)
sup_rot_y_opt = trf.kraus_to_super(ry_t(theta_list[max_index]), my_hs_basis)
print("Maximum fidelity after ideal Y rotation =", max_fid)

#########################################################
# Leakage and seepage
#########################################################

avg_leakage = env_syst.leakage(comp_states_list)
print("Average leakage =", avg_leakage)

avg_seepage = env_syst.seepage(comp_states_list)
print("Average seepage =", avg_seepage)

print("d_1 * L_1(E) =", d_comp*avg_leakage)
print("d_2 * L_2(E) =", d_leak*avg_seepage)

#########################################################
# Average gate fidelity
#########################################################

total_sup_op_ry = sup_rot_y_opt.dot(total_sup_op)
f_gate = average_gate_fidelity(cr_super_target, total_sup_op_ry, avg_leakage)
print("Average gate fidelity =", f_gate)

