import scqubits as scq
import qutip as qt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# === Define Fluxonium Qubit ===
qbtA = scq.Fluxonium(
    EJ=4.5, EC=1.09, EL=0.8,
    flux=0.50,  # Flux frustration point
    cutoff=110,  # Higher cutoff to model leakage states
    truncated_dim=3  # Include leakage to |2>
)

# Define computational basis states
ket_0 = qt.basis(3, 0)  # |0⟩
ket_1 = qt.basis(3, 1)  # |1⟩
ket_2 = qt.basis(3, 2)  # |2⟩ (leakage state)

# Define Pauli matrices in 3-level space
sigma_x = qt.basis(3, 0) * qt.basis(3, 1).dag() + qt.basis(3, 1) * qt.basis(3, 0).dag()
sigma_y = -1j * (qt.basis(3, 0) * qt.basis(3, 1).dag() - qt.basis(3, 1) * qt.basis(3, 0).dag())
sigma_z = qt.basis(3, 0) * qt.basis(3, 0).dag() - qt.basis(3, 1) * qt.basis(3, 1).dag()

# === Define Noise Parameters ===
T1 = 30e3  # Relaxation time in ns
T2 = 15e3  # Dephasing time in ns
leakage_rate = 0.01  # Probability of leakage to |2⟩
gate_error_strength = 0.002  # Strength of unitary gate noise
spam_error_prob = 0.01  # SPAM error probability

# === Define Collapse Operators for Lindblad Master Equation ===
collapse_ops = [
    np.sqrt(1/T1) * qt.sigmam(),  # T1 relaxation (decay |1⟩ → |0⟩)
    np.sqrt(1/T2) * sigma_z       # T2 dephasing
]

# Leakage operator (causes transitions from |1⟩ → |2⟩)
leakage_op = leakage_rate * (qt.basis(3, 2) * qt.basis(3, 1).dag())

# === Define Noisy Clifford Gates ===
def apply_noisy_gate(state, gate, noise_strength=gate_error_strength):
    """Apply a Clifford gate with a small random error"""
    random_hamiltonian = noise_strength * qt.rand_herm(3)  # Small random perturbation
    error_operator = (-1j * random_hamiltonian).expm()  # Convert to unitary error
    noisy_gate = error_operator * gate  # Combine with ideal gate
    return noisy_gate * state

# Define random Clifford gates (identity, X, Y, Z)
clifford_gates = [
    qt.qeye(3),  # Identity
    sigma_x,     # X
    sigma_y,     # Y
    sigma_z      # Z
]

# === Define SPAM Error Model ===
def add_spam_error(state, p_error=spam_error_prob):
    """Simulate SPAM (State Preparation And Measurement) errors"""
    if np.random.rand() < p_error:
        return ket_1 if state == ket_0 else ket_0  # Flip measurement result
    return state

# === Define Randomized Benchmarking Simulation ===
def simulate_rb_with_noise(sequence_length):
    """Simulate randomized benchmarking with gate errors and decoherence"""
    state = ket_0  # Start in |0⟩

    for _ in range(sequence_length):
        random_gate = np.random.choice(clifford_gates)  # Pick a random Clifford gate
        state = apply_noisy_gate(state, random_gate)  # Apply noisy gate
    
    # Apply Lindblad equation for decoherence
    H = qt.qeye(3)  # Identity Hamiltonian (assuming gates are unitary)

    final_state = qt.mesolve(H, state, tlist=[0, 10], c_ops=collapse_ops).states[-1]

    # Apply SPAM error in measurement
    measured_state = add_spam_error(final_state)

    # Compute probability of remaining in |0⟩
    return abs(qt.expect(qt.basis(3, 0) * qt.basis(3, 0).dag(), measured_state))

# === Run the RB Experiment ===
sequence_lengths = np.arange(1, 100, 5)  # Sequence lengths from 1 to 100
num_experiments = 30  # Number of random experiments per sequence length
survival_probabilities = [np.mean([simulate_rb_with_noise(length) for _ in range(num_experiments)]) for length in sequence_lengths]

# === Fit the Exponential Decay Curve ===
def exp_decay(x, A, p, B):
    return A * p**x + B

popt, _ = curve_fit(exp_decay, sequence_lengths, survival_probabilities, p0=[1, 0.98, 0.5])
gate_fidelity = popt[1] * 100

# === Plot RB Curve ===
plt.figure(figsize=(8, 5))
plt.plot(sequence_lengths, survival_probabilities, 'o', label="Simulated Data")
plt.plot(sequence_lengths, exp_decay(sequence_lengths, *popt), label="Fit: Fidelity={:.2f}%".format(gate_fidelity))
plt.xlabel("Sequence Length")
plt.ylabel("Survival Probability")
plt.legend()
plt.title("Single-Qubit Randomized Benchmarking with Noise")
plt.grid()
plt.show()

print(f"Extracted gate fidelity: {gate_fidelity:.6f}%")
