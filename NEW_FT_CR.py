

#%% code for best set of parameters

import numpy as np
import scqubits as scq
import qutip as qt
from matplotlib import pyplot as plt

# ----------------------------
# Fixed transmon parameters
# ----------------------------
qbtb = scq.Transmon(
    EJ=4.5,
    EC=0.04,
    ng=0,
    ncut=110,
    truncated_dim=10,
)

# ----------------------------
# Parameter ranges for fluxonium (in GHz)
# ----------------------------
EC_range = np.linspace(0.92, 1.2, 1)    # Example: 6 points between 0.7 and 1.2
EJ_range = np.linspace(5.6, 8.0, 1)      # Example: 7 points between 5.0 and 8.0
EL_range = np.linspace(0.56, 1.1, 1)      # Example: 5 points between 0.7 and 1.1

# Desired fluxonium f_01 range (in GHz)
target_f01_min = 0.25  # 250 MHz
target_f01_max = 0.40  # 400 MHz

# A list to store candidate parameter sets and their metrics
results = []

# For truncating dressed operators/matrices
total_truncation = 50

def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    """
    Truncate a Qobj (e.g. representing the dressed Hamiltonian or an operator)
    to the specified matrix dimension.
    """
    return qt.Qobj(operator.full()[:dimension, :dimension])

# -------------------------------------------------
# Grid search over fluxonium parameters
# -------------------------------------------------
print("Searching for fluxonium parameters that yield a fluxonium 0-1 frequency in the"
      " range 250–400 MHz while minimizing static ZZ and maximizing CR rate...\n")

# Loop over the grid for fluxonium parameters
for Ec in EC_range:
    for EJ in EJ_range:
        for EL in EL_range:
            
            # Build the fluxonium qubit at half-flux bias
            qbta = scq.Fluxonium(
                EC=Ec,
                EJ=EJ,
                EL=EL,
                flux=0.5,      # flux frustration point
                cutoff=110,
                truncated_dim=10,
            )
            
            # Construct the composite Hilbert space (order: [fluxonium, transmon])
            hilbertspace = scq.HilbertSpace([qbta, qbtb])
            
            # Add the capacitive interaction between the two qubits
            hilbertspace.add_interaction(
                g_strength=0.024,
                op1=qbta.n_operator,
                op2=qbtb.n_operator,
            )
            
            # Build the lookup for the dressed basis of the composite system
            hilbertspace.generate_lookup()
            
            # -------------------------
            # Construct the dressed Hamiltonian
            # -------------------------
            # Obtain the eigenvalue lookup table from the HilbertSpace.
            # The eigenvalues (in GHz) are stored in the lookup.
            (evals_lookup,) = hilbertspace["evals"]
            # Multiply by 2π to get units compatible with qutip (so time is in ns)
            H_dressed = 2 * np.pi * qt.Qobj(np.diag(evals_lookup),
                                            dims=[hilbertspace.subsystem_dims] * 2)
            # Truncate the dressed Hamiltonian to a manageable dimension:
            H_dressed_trunc = truncate(H_dressed, total_truncation)
            # Compute the eigenenergies and subtract the ground state energy.
            # Dividing by 6.28 roughly recovers energies in GHz.
            dressed_evals = (H_dressed_trunc.eigenenergies() - H_dressed_trunc.eigenenergies()[0]) / 6.28
            
            # ---------------
            # Identify dressed basis indices for the product states:
            # For a composite system [fluxonium, transmon]:
            # - (0,0): both in ground state
            # - (1,0): fluxonium in first excited state, transmon ground state
            # - (0,1): fluxonium ground state, transmon first excited state
            # - (1,1): both excited (used for ZZ calculation)
            # ----------------
            try:
                idx00 = hilbertspace.dressed_index((0, 0))
                idx10 = hilbertspace.dressed_index((1, 0))
                idx01 = hilbertspace.dressed_index((0, 1))
                idx11 = hilbertspace.dressed_index((1, 1))
            except Exception as err:
                print(f"Error obtaining dressed indices: {err}")
                continue
            
            # Compute fluxonium 0-1 transition frequency from the dressed spectrum:
            f01_flux = dressed_evals[idx10] - dressed_evals[idx00]
            
            # Only consider if fluxonium f01 is in the desired range:
            if target_f01_min <= f01_flux <= target_f01_max:
                
                # Compute static ZZ (in GHz):
                static_ZZ = dressed_evals[idx11] - dressed_evals[idx10] - dressed_evals[idx01] + dressed_evals[idx00]
                
                # Compute the representation of the fluxonium number operator in the dressed basis:
                n_a = hilbertspace.op_in_dressed_eigenbasis(qbta.n_operator)
                n_a = truncate(n_a, total_truncation)
                
                # Compute the CR rate as the absolute difference between the following matrix elements:
                #   n_a[|0,0> -> |0,1>] and n_a[|1,0> -> |1,1>].
                n_a_00_to_01 = n_a.full()[idx00, hilbertspace.dressed_index((0, 1))]
                n_a_10_to_11 = n_a.full()[hilbertspace.dressed_index((1, 0)), hilbertspace.dressed_index((1, 1))]
                CR_coeff = abs(n_a_00_to_01 - n_a_10_to_11)
                
                # Save the candidate parameters and metrics.
                results.append({
                    'Ec (GHz)': Ec,
                    'EJ (GHz)': EJ,
                    'EL (GHz)': EL,
                    'Fluxonium f01 (GHz)': f01_flux,
                    'static_ZZ (GHz)': static_ZZ,
                    'CR_coeff (arb.)': CR_coeff,
                })
                
                print(f"Candidate found: Ec={Ec:.3f}, EJ={EJ:.3f}, EL={EL:.3f} --> "
                      f"f01_flux={f01_flux:.3f} GHz, static_ZZ={static_ZZ:.3f} GHz, CR_coeff={CR_coeff:.3e}")
                
# -----------------------------
# Post-Processing Results
# -----------------------------
if not results:
    print("\nNo candidates found with the fluxonium 0-1 frequency in the target range. "
          "Consider adjusting the parameter ranges or grid resolution.")
else:
    # Here, you might want to choose candidates that have minimal |static_ZZ| and a large CR rate.
    # For example, one simple ranking is to sort first by small absolute static_ZZ and then by descending CR_coeff.
    results_sorted = sorted(results,
                            key=lambda r: (abs(r['static_ZZ (GHz)']), -r['CR_coeff (arb.)']))
    
    print("\nSorted candidate parameters (priority: minimal static_ZZ and high CR_coeff):")
    for cand in results_sorted:
        print(cand)
        
    # Optionally, plot the tradeoff between static ZZ and CR rate for visualization.
    static_ZZ_vals = [abs(c['static_ZZ (GHz)']) for c in results_sorted]
    CR_coeff_vals = [c['CR_coeff (arb.)'] for c in results_sorted]
    
    plt.figure(figsize=(6,4))
    plt.scatter(static_ZZ_vals, CR_coeff_vals, s=80, c='blue')
    plt.xlabel(r'|static ZZ| (GHz)')
    plt.ylabel('CR rate (arb. units)')
    plt.title('Trade-off between static ZZ and CR rate in candidate designs')
    plt.grid(True)
    plt.show()

#%% parameters checking

import numpy as np
import scqubits as scq
import qutip as qt
from matplotlib import pyplot as plt

# -----------------------------------------------------------
# Fixed transmon parameters (held constant throughout)
# -----------------------------------------------------------
qbtb = scq.Transmon(
    EJ=5,
    EC=0.032,
    ng=0,
    ncut=110,
    truncated_dim=10,
)

# -----------------------------------------------------------
# Define fluxonium parameter ranges (in GHz)
# -----------------------------------------------------------
EC_range = np.linspace(0.7, 1.2, 6)    # Charging energy: 6 points between 0.7 and 1.2 GHz
EJ_range = np.linspace(5.0, 8.0, 7)      # Josephson energy: 7 points between 5.0 and 8.0 GHz
EL_range = np.linspace(0.7, 1.1, 5)      # Inductive energy: 5 points between 0.7 and 1.1 GHz

# Desired fluxonium 0-1 frequency range (in GHz): 250-400 MHz
target_f01_min = 0.25  # GHz
target_f01_max = 0.50  # GHz

# Candidate results will be stored in a list of dictionaries
results = []

# For truncating dressed operators/matrices
total_truncation = 50

def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    """
    Truncate a Qobj (for example, representing an operator or Hamiltonian)
    to the specified dimension.
    """
    return qt.Qobj(operator.full()[:dimension, :dimension])

print("Starting grid search for candidate fluxonium parameter sets...\n")

# -----------------------------------------------------------
# Grid search: loop over fluxonium parameters
# -----------------------------------------------------------
for Ec in EC_range:
    for EJ in EJ_range:
        for EL in EL_range:
            
            # Create a fluxonium qubit with the current parameters at half flux bias.
            qbta = scq.Fluxonium(
                EC=Ec,
                EJ=EJ,
                EL=EL,
                flux=0.5,      # half-flux bias (flux frustration point)
                cutoff=110,
                truncated_dim=10,
            )
            
            # Combine the fluxonium and transmon into a composite Hilbert space.
            hilbertspace = scq.HilbertSpace([qbta, qbtb])
            hilbertspace.add_interaction(
                g_strength=0.024,
                op1=qbta.n_operator,
                op2=qbtb.n_operator,
            )
            hilbertspace.generate_lookup()
            
            # -----------------------------------------------------------
            # Construct the composite dressed Hamiltonian.
            # -----------------------------------------------------------
            (evals_lookup,) = hilbertspace["evals"]
            # Convert eigenvalues to a Qobj with a 2pi factor (units: GHz -> ns system)
            H_dressed = 2 * np.pi * qt.Qobj(np.diag(evals_lookup),
                                            dims=[hilbertspace.subsystem_dims] * 2)
            H_dressed_trunc = truncate(H_dressed, total_truncation)
            dressed_evals = (H_dressed_trunc.eigenenergies() - H_dressed_trunc.eigenenergies()[0]) / 6.28
            
            # -----------------------------------------------------------
            # Obtain dressed basis indices corresponding to product states.
            # For the composite system [fluxonium, transmon]:
            # (0,0): both qubits in ground state
            # (1,0): fluxonium excited, transmon ground state
            # (0,1): fluxonium ground state, transmon excited
            # (1,1): both excited (for ZZ calculation)
            # -----------------------------------------------------------
            try:
                idx00 = hilbertspace.dressed_index((0, 0))
                idx10 = hilbertspace.dressed_index((1, 0))
                idx01 = hilbertspace.dressed_index((0, 1))
                idx11 = hilbertspace.dressed_index((1, 1))
            except Exception as err:
                print(f"Error obtaining dressed indices: {err}")
                continue
            
            # Compute fluxonium 0-1 transition frequency (f01) taken as the difference
            # between the dressed energies corresponding to states (1,0) and (0,0)
            f01_flux = dressed_evals[idx10] - dressed_evals[idx00]
            
            # Only consider candidates in the desired range
            if target_f01_min <= f01_flux <= target_f01_max:
                
                # Compute static ZZ (in GHz)
                static_ZZ = 1e3*(dressed_evals[idx11] - dressed_evals[idx10] - dressed_evals[idx01] + dressed_evals[idx00])
                
                # Compute the representation of fluxonium's number operator in the dressed basis.
                n_a = hilbertspace.op_in_dressed_eigenbasis(qbta.n_operator)
                n_a = truncate(n_a, total_truncation)
                
                # Compute the CR rate as the absolute difference of the matrix elements:
                #   <0,0|n_a|0,1>  versus  <1,0|n_a|1,1>
                n_a_00_to_01 = n_a.full()[idx00, hilbertspace.dressed_index((0, 1))]
                n_a_10_to_11 = n_a.full()[hilbertspace.dressed_index((1, 0)), hilbertspace.dressed_index((1, 1))]
                CR_coeff = abs(n_a_00_to_01 - n_a_10_to_11)*1e3
                
                # Record candidate results:
                results.append({
                    'Ec (GHz)': Ec,
                    'EJ (GHz)': EJ,
                    'EL (GHz)': EL,
                    'Fluxonium f01 (GHz)': f01_flux,
                    'static_ZZ (GHz)': static_ZZ,
                    'CR_coeff (arb.)': CR_coeff,
                })
                print(f"Candidate: Ec={Ec:.3f}, EJ={EJ:.3f}, EL={EL:.3f} --> f01={f01_flux:.3f} GHz, "
                      f"static_ZZ={static_ZZ:.3f} GHz, CR_coeff={CR_coeff:.3e}")

# -----------------------------------------------------------
# Post-Processing and Plotting
# -----------------------------------------------------------
if not results:
    print("\nNo candidates found with the desired fluxonium 0-1 frequency range. "
          "Adjust the parameter ranges or grid resolution.")
else:
    # Sort the candidates by preferring minimal |static_ZZ| and higher CR_coeff.
    results_sorted = sorted(results,
                            key=lambda r: (abs(r['static_ZZ (GHz)']), -r['CR_coeff (arb.)']))
    
    # For example, the best candidate is chosen as follows:
    best_candidate = results_sorted[0]
    
    print("\nBest candidate parameters:")
    for key, value in best_candidate.items():
        print(f"  {key}: {value}")
    
    # -----------------------
    # Scatter Plot: trade-off between |static_ZZ| and CR_coeff.
    # -----------------------
    # This plot shows each candidate as a point with:
    #   x-axis: |static_ZZ| (lower is better)
    #   y-axis: CR_coeff (higher is better)
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot each candidate design.
    for cand in results_sorted:
        ax.scatter(abs(cand['static_ZZ (GHz)']), cand['CR_coeff (arb.)'],
                   color='blue', s=50)
    
    # Highlight the best candidate.
    best_x = abs(best_candidate['static_ZZ (GHz)'])
    best_y = best_candidate['CR_coeff (arb.)']
    ax.scatter(best_x, best_y, color='red', s=120, label='Best Candidate')
    
    ax.set_xlabel(r'$|\mathrm{static\ ZZ}|$ (GHz)', fontsize=12)
    ax.set_ylabel('CR Rate (arb. units)', fontsize=12)
    ax.set_title('Candidate Trade-off: Static ZZ vs CR Rate', fontsize=14)
    ax.legend()
    ax.grid(True)
    
    # Annotate the best candidate with its parameters
    annotation = (f"Ec={best_candidate['Ec (GHz)']:.2f}\n"
                  f"EJ={best_candidate['EJ (GHz)']:.2f}\n"
                  f"EL={best_candidate['EL (GHz)']:.2f}\n"
                  f"f01={best_candidate['Fluxonium f01 (GHz)']*1e3:.1f} MHz")
    ax.annotate(annotation, xy=(best_x, best_y), xytext=(best_x*1.1, best_y*1.1),
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                fontsize=10, backgroundcolor='white')
    
    plt.show()

    # Optionally, you could also create a parallel coordinates plot to display
    # all three fluxonium parameters for each candidate.
    try:
        import pandas as pd
        from pandas.plotting import parallel_coordinates

        # Create a DataFrame from the candidate results for the selected parameters.
        df = pd.DataFrame(results_sorted)
        # Choose columns to display.
        cols = ['Ec (GHz)', 'EJ (GHz)', 'EL (GHz)', 'Fluxonium f01 (GHz)',
                'static_ZZ (GHz)', 'CR_coeff (arb.)']
        df = df[cols]
        plt.figure(figsize=(10, 6))
        parallel_coordinates(df, class_column='Fluxonium f01 (GHz)', colormap=plt.get_cmap("Set1"))
        plt.title("Parallel Coordinates Plot for Candidate Designs")
        plt.show()
    except ImportError:
        print("pandas is not installed so skipping the parallel coordinates plot.")


#%% best parameters choice 


import numpy as np
import scqubits as scq
import qutip as qt
from matplotlib import pyplot as plt
from adjustText import adjust_text  # Make sure to install adjustText using `pip install adjustText`

# -----------------------------------------------------------
# Fixed transmon parameters (held constant throughout)
# -----------------------------------------------------------
qbtb = scq.Transmon(
    EJ=5,
    EC=0.032,
    ng=0,
    ncut=110,
    truncated_dim=10,
)

# -----------------------------------------------------------
# Define fluxonium parameter ranges (in GHz)
# -----------------------------------------------------------
EC_range = np.linspace(1.07, 1.1, 1)    # Charging energy: 6 points between 0.7 and 1.2 GHz
EJ_range = np.linspace(5, 7.5, 20)    # Josephson energy: 7 points between 5.0 and 8.0 GHz
EL_range = np.linspace(.8, 1.5,10)    # Inductive energy: 5 points between 0.7 and 1.1 GHz

# Desired fluxonium 0-1 frequency range (in GHz): 250-400 MHz
target_f01_min = 0.2  # GHz
target_f01_max = 0.5  # GHz

# Candidate results will be stored in a list of dictionaries
results = []

# For truncating dressed operators/matrices
total_truncation = 50

def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    """
    Truncate a Qobj (for example, representing an operator or Hamiltonian)
    to the specified dimension.
    """
    return qt.Qobj(operator.full()[:dimension, :dimension])

print("Starting grid search for candidate fluxonium parameter sets...\n")

# -----------------------------------------------------------
# Grid search: loop over fluxonium parameters
# -----------------------------------------------------------
for Ec in EC_range:
    for EJ in EJ_range:
        for EL in EL_range:
            
            # Create a fluxonium qubit with the current parameters at half flux bias.
            qbta = scq.Fluxonium(
                EC=Ec,
                EJ=EJ,
                EL=EL,
                flux=0.5,      # half-flux bias (flux frustration point)
                cutoff=110,
                truncated_dim=10,
            )
            
            # Combine the fluxonium and transmon into a composite Hilbert space.
            hilbertspace = scq.HilbertSpace([qbta, qbtb])
            hilbertspace.add_interaction(
                g_strength=0.024,
                op1=qbta.n_operator,
                op2=qbtb.n_operator,
            )
            hilbertspace.generate_lookup()
            
            # -----------------------------------------------------------
            # Construct the composite dressed Hamiltonian.
            # -----------------------------------------------------------
            (evals_lookup,) = hilbertspace["evals"]
            # Convert eigenvalues to a Qobj with a 2pi factor (units: GHz -> ns system)
            H_dressed = 2 * np.pi * qt.Qobj(np.diag(evals_lookup),
                                            dims=[hilbertspace.subsystem_dims] * 2)
            H_dressed_trunc = truncate(H_dressed, total_truncation)
            dressed_evals = (H_dressed_trunc.eigenenergies() - H_dressed_trunc.eigenenergies()[0]) / 6.28
            
            # -----------------------------------------------------------
            # Obtain dressed basis indices corresponding to product states.
            # -----------------------------------------------------------
            try:
                idx00 = hilbertspace.dressed_index((0, 0))
                idx10 = hilbertspace.dressed_index((1, 0))
                idx01 = hilbertspace.dressed_index((0, 1))
                idx11 = hilbertspace.dressed_index((1, 1))
            except Exception as err:
                print(f"Error obtaining dressed indices: {err}")
                continue
            
            # Compute fluxonium 0-1 transition frequency (f01) in GHz
            f01_flux = dressed_evals[idx10] - dressed_evals[idx00]
            
            # Only consider candidates in the desired range
            if target_f01_min <= f01_flux <= target_f01_max:
                
                # Compute static ZZ (in MHz)
                static_ZZ = abs(dressed_evals[idx11] - dressed_evals[idx10] - dressed_evals[idx01] + dressed_evals[idx00]) * 1e3
                
                # Compute the CR rate (multiplied by 1000 for clarity)
                n_a = hilbertspace.op_in_dressed_eigenbasis(qbta.n_operator)
                n_a = truncate(n_a, total_truncation)
                n_a_00_to_01 = n_a.full()[idx00, hilbertspace.dressed_index((0, 1))]
                n_a_10_to_11 = n_a.full()[hilbertspace.dressed_index((1, 0)), hilbertspace.dressed_index((1, 1))]
                CR_coeff = abs(n_a_00_to_01 - n_a_10_to_11) * 1e3
                
                # Record candidate results
                results.append({
                    'Ec (GHz)': Ec,
                    'EJ (GHz)': EJ,
                    'EL (GHz)': EL,
                    'Fluxonium f01 (GHz)': f01_flux,
                    'static_ZZ (MHz)': static_ZZ,
                    'CR_coeff (MHz)': CR_coeff,
                })
                print(f"Candidate: Ec={Ec:.3f}, EJ={EJ:.3f}, EL={EL:.3f} --> f01={f01_flux:.3f} GHz, "
                      f"static_ZZ={static_ZZ:.3f} MHz, CR_coeff={CR_coeff:.3f} MHz")

# -----------------------------------------------------------
# Plotting: Static ZZ vs CR Rate
# -----------------------------------------------------------
if results:
    # Extract data for plotting
    static_ZZ_vals = [r['static_ZZ (MHz)'] for r in results]
    CR_coeff_vals = [r['CR_coeff (MHz)'] for r in results]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all candidates in blue
    ax.scatter(static_ZZ_vals, CR_coeff_vals, color='blue', s=50)

    # Annotate all points with parameters including f01
    texts = []
    for r in results:
        annotation = f"Ec={r['Ec (GHz)']:.2f}, EJ={r['EJ (GHz)']:.2f}, EL={r['EL (GHz)']:.2f}, f01={r['Fluxonium f01 (GHz)']*1e3:.1f} MHz"
        texts.append(ax.text(r['static_ZZ (MHz)'], r['CR_coeff (MHz)'], annotation, fontsize=10))

    # Use adjustText to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    # Labels and title
    ax.set_xlabel('Static ZZ (MHz)', fontsize=12)
    ax.set_ylabel('CR Rate (MHz)', fontsize=12)
    ax.set_title('Static ZZ vs CR Rate with Parameters and f01 Frequency Fluxonium + Low Freq Transmon', fontsize=14)
    ax.grid(True)

    # Show the plot
    plt.show()
#%% 
import os

# Create the folder if it doesn't exist
folder_path = r"Z:\Tanvir\Tanvir Files and codes\Codes_11_14_2024\qubit_results"
os.makedirs(folder_path, exist_ok=True)

# Define the header format for the files
header = f"{'Flux_EC (GHz)':<12}{'Flux_EJ (GHz)':<12}{'Flux_EL (GHz)':<12}{'Trans_EC (GHz)':<14}{'Trans_EJ (GHz)':<14}{'f01 (GHz)':<12}{'CR (MHz)':<10}{'ZZ (MHz)':<10}\n"

# -----------------------------------------------------------
# Save ZZ values in ascending order
# -----------------------------------------------------------
results_sorted_by_ZZ = sorted(results, key=lambda x: x['static_ZZ (MHz)'])
with open(os.path.join(folder_path, "ZZ_ascending.txt"), "w") as f:
    f.write(header)
    for r in results_sorted_by_ZZ:
        f.write(f"{r['Flux_EC (GHz)']:<12.2f}{r['Flux_EJ (GHz)']:<12.2f}{r['Flux_EL (GHz)']:<12.2f}"
                f"{r['Trans_EC (GHz)']:<14.4f}{r['Trans_EJ (GHz)']:<14.4f}{r['f01 (GHz)']:<12.6f}"
                f"{r['CR_coeff (MHz)']:<10.2f}{r['static_ZZ (MHz)']:<10.2f}\n")

# -----------------------------------------------------------
# Save CR rate values in descending order
# -----------------------------------------------------------
results_sorted_by_CR = sorted(results, key=lambda x: -x['CR_coeff (MHz)'])
with open(os.path.join(folder_path, "CR_descending.txt"), "w") as f:
    f.write(header)
    for r in results_sorted_by_CR:
        f.write(f"{r['Flux_EC (GHz)']:<12.2f}{r['Flux_EJ (GHz)']:<12.2f}{r['Flux_EL (GHz)']:<12.2f}"
                f"{r['Trans_EC (GHz)']:<14.4f}{r['Trans_EJ (GHz)']:<14.4f}{r['f01 (GHz)']:<12.6f}"
                f"{r['CR_coeff (MHz)']:<10.2f}{r['static_ZZ (MHz)']:<10.2f}\n")

# -----------------------------------------------------------
# Confirm completion
# -----------------------------------------------------------
print("Text files saved successfully.")

import os
import csv

# Create the folder if it doesn't exist
folder_path = r"Z:\Tanvir\Tanvir Files and codes\Codes_11_14_2024\qubit_results"
os.makedirs(folder_path, exist_ok=True)

# Define the header format for the CSV files
header = ["Flux_EC (GHz)", "Flux_EJ (GHz)", "Flux_EL (GHz)", "Trans_EC (GHz)", "Trans_EJ (GHz)", "f01 (GHz)", "CR_coeff (MHz)", "static_ZZ (MHz)"]

# -----------------------------------------------------------
# Save ZZ values in ascending order (as CSV)
# -----------------------------------------------------------
results_sorted_by_ZZ = sorted(results, key=lambda x: x['static_ZZ (MHz)'])
zz_csv_path = os.path.join(folder_path, "ZZ_ascending.csv")

with open(zz_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(header)
    # Write the data
    for r in results_sorted_by_ZZ:
        writer.writerow([
            r['Flux_EC (GHz)'], r['Flux_EJ (GHz)'], r['Flux_EL (GHz)'],
            r['Trans_EC (GHz)'], r['Trans_EJ (GHz)'], r['f01 (GHz)'],
            r['CR_coeff (MHz)'], r['static_ZZ (MHz)']
        ])

# -----------------------------------------------------------
# Save CR rate values in descending order (as CSV)
# -----------------------------------------------------------
results_sorted_by_CR = sorted(results, key=lambda x: -x['CR_coeff (MHz)'])
cr_csv_path = os.path.join(folder_path, "CR_descending_Jc_25.csv")

with open(cr_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    # Write the header
    writer.writerow(header)
    # Write the data
    for r in results_sorted_by_CR:
        writer.writerow([
            r['Flux_EC (GHz)'], r['Flux_EJ (GHz)'], r['Flux_EL (GHz)'],
            r['Trans_EC (GHz)'], r['Trans_EJ (GHz)'], r['f01 (GHz)'],
            r['CR_coeff (MHz)'], r['static_ZZ (MHz)']
        ])

# -----------------------------------------------------------
# Confirm completion
# -----------------------------------------------------------
print("CSV files saved successfully.")


#%% Best parmeter choice vary all five params

import numpy as np
import scqubits as scq
import qutip as qt
from matplotlib import pyplot as plt
from adjustText import adjust_text  # Make sure to install adjustText using `pip install adjustText`

# -----------------------------------------------------------
# Define parameter ranges for Fluxonium and Transmon (in GHz)
# -----------------------------------------------------------
# Fluxonium parameter ranges
flux_EC_range = np.linspace(1.05, 1.1, 1)    # Fluxonium charging energy
flux_EJ_range = np.linspace(7,8.5, 6 )      # Fluxonium Josephson energy
flux_EL_range = np.linspace(1.27, 1.6, 1)    # Fluxonium inductive energy

# Transmon parameter ranges
trans_EC_range = np.linspace(0.04, 0.04, 1)  # Transmon charging energy
trans_EJ_range = np.linspace(6, 6, 1)     # Transmon Josephson energy

# flux_EC_range = np.linspace(1.05, 1.1, 1)    # Fluxonium charging energy
# flux_EJ_range = np.linspace(5,9, 13 )      # Fluxonium Josephson energy
# flux_EL_range = np.linspace(.9 ,1.5, 11)    # Fluxonium inductive energy

# # Transmon parameter ranges
# trans_EC_range = np.linspace(0.04, 0.04, 1)  # Transmon charging energy
# trans_EJ_range = np.linspace(5, 7, 5)     # Transmon Josephson energy

# Fluxonium parameter ranges
# flux_EC_range = np.linspace(1.05, 1.1, 1)    # Fluxonium charging energy
# flux_EJ_range = np.linspace(6.88,7.5, 1 )      # Fluxonium Josephson energy
# flux_EL_range = np.linspace(1.5, 1.6, 1)    # Fluxonium inductive energy

# # Transmon parameter ranges
# trans_EC_range = np.linspace(0.04, 0.04, 1)  # Transmon charging energy
# trans_EJ_range = np.linspace(5, 5.5, 1)     # Transmon Josephson energy

Coupling_Jc = 0.015

# Desired fluxonium 0-1 frequency range (in GHz): 250-500 MHz
target_f01_min = 0.2  # GHz
target_f01_max = 0.5  # GHz

# Candidate results will be stored in a list of dictionaries
results = []

# For truncating dressed operators/matrices
total_truncation = 20

# -----------------------------------------------------------
# Helper function: Truncate operators to desired dimension
# -----------------------------------------------------------
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator.full()[:dimension, :dimension])

print("Starting grid search for candidate parameter sets...\n")

# -----------------------------------------------------------
# Grid search: loop over both Fluxonium and Transmon parameters
# -----------------------------------------------------------
for flux_EC in flux_EC_range:
    for flux_EJ in flux_EJ_range:
        for flux_EL in flux_EL_range:
            for trans_EC in trans_EC_range:
                for trans_EJ in trans_EJ_range:
                    # -----------------------------------------------------------
                    # Create Fluxonium and Transmon qubits with current parameters
                    # -----------------------------------------------------------
                    qbta = scq.Fluxonium(
                        EC=flux_EC,
                        EJ=flux_EJ,
                        EL=flux_EL,
                        flux=0.5,  # half-flux bias (flux frustration point)
                        cutoff=110,
                        truncated_dim=10,
                    )
                    qbtb = scq.Transmon(
                        EJ=trans_EJ,
                        EC=trans_EC,
                        ng=0,
                        ncut=110,
                        truncated_dim=10,
                    )

                    # -----------------------------------------------------------
                    # Combine Fluxonium and Transmon into a composite Hilbert space
                    # -----------------------------------------------------------
                    hilbertspace = scq.HilbertSpace([qbta, qbtb])
                    hilbertspace.add_interaction(
                        g_strength=Coupling_Jc,
                        op1=qbta.n_operator,
                        op2=qbtb.n_operator,
                    )
                    hilbertspace.generate_lookup()

                    # -----------------------------------------------------------
                    # Construct the composite dressed Hamiltonian
                    # -----------------------------------------------------------
                    (evals_lookup,) = hilbertspace["evals"]
                    # Convert eigenvalues to a Qobj with a 2pi factor (GHz -> ns system)
                    H_dressed = 2 * np.pi * qt.Qobj(np.diag(evals_lookup), dims=[hilbertspace.subsystem_dims] * 2)
                    H_dressed_trunc = truncate(H_dressed, total_truncation)
                    dressed_evals = (H_dressed_trunc.eigenenergies() - H_dressed_trunc.eigenenergies()[0]) / 6.28

                    # -----------------------------------------------------------
                    # Obtain dressed basis indices corresponding to product states
                    # -----------------------------------------------------------
                    try:
                        idx00 = hilbertspace.dressed_index((0, 0))
                        idx10 = hilbertspace.dressed_index((1, 0))
                        idx01 = hilbertspace.dressed_index((0, 1))
                        idx11 = hilbertspace.dressed_index((1, 1))
                    except Exception as err:
                        print(f"Error obtaining dressed indices: {err}")
                        continue
                    static_ZZ = abs(dressed_evals[idx11] - dressed_evals[idx10] - dressed_evals[idx01] + dressed_evals[idx00]) * 1e3

                    # -----------------------------------------------------------
                    # Compute fluxonium 0-1 transition frequency (f01) in GHz
                    # -----------------------------------------------------------
                    f01_flux = dressed_evals[idx10] - dressed_evals[idx00]
                    f01_trans = dressed_evals[idx01] - dressed_evals[idx00]
                    n_a = hilbertspace.op_in_dressed_eigenbasis(qbta.n_operator)
                    n_a = truncate(n_a, total_truncation)
                    n_a_00_to_01 = n_a.full()[idx00, hilbertspace.dressed_index((0, 1))]
                    n_a_10_to_11 = n_a.full()[hilbertspace.dressed_index((1, 0)), hilbertspace.dressed_index((1, 1))]
                    CR_coeff = abs(n_a_00_to_01 - n_a_10_to_11) * 1e3

                    # Only consider candidates in the desired range
                    if target_f01_min <= f01_flux <= target_f01_max  :
                    # if target_f01_min <= f01_flux <= target_f01_max and static_ZZ <= 0.12 and CR_coeff >= 4 :
                        # Compute static ZZ (in MHz)
                        static_ZZ = abs(dressed_evals[idx11] - dressed_evals[idx10] - dressed_evals[idx01] + dressed_evals[idx00]) * 1e3

                        # Compute the CR rate (multiplied by 1000 for clarity) and
                        n_a = hilbertspace.op_in_dressed_eigenbasis(qbta.n_operator)
                        n_a = truncate(n_a, total_truncation)
                        n_a_00_to_01 = n_a.full()[idx00, hilbertspace.dressed_index((0, 1))]
                        n_a_10_to_11 = n_a.full()[hilbertspace.dressed_index((1, 0)), hilbertspace.dressed_index((1, 1))]
                        CR_coeff = abs(n_a_00_to_01 - n_a_10_to_11) * 1e3

                        # Record candidate results
                        results.append({
                            'Flux_EC (GHz)': flux_EC,
                            'Flux_EJ (GHz)': flux_EJ,
                            'Flux_EL (GHz)': flux_EL,
                            'Trans_EC (GHz)': trans_EC,
                            'Trans_EJ (GHz)': trans_EJ,
                            'f01 (GHz)': f01_flux,
                            'static_ZZ (MHz)': static_ZZ,
                            'CR_coeff (MHz)': CR_coeff,
                            'f01_T (GHz)': f01_trans,
                        })
                        print(f"Candidate: EC={flux_EC:.3f}, EJ={flux_EJ:.3f}, EL={flux_EL:.3f}, "
                              
                              f"f01={f01_flux:.3f} GHz, f01_T={f01_trans:.3f} GHz, static_ZZ={static_ZZ:.3f} MHz, CR_coeff={CR_coeff:.3f} MHz")

# -----------------------------------------------------------
# Plotting: Static ZZ vs CR Rate
# -----------------------------------------------------------
if results:
    # Extract data for plotting
    static_ZZ_vals = [r['static_ZZ (MHz)'] for r in results]
    CR_coeff_vals = [r['CR_coeff (MHz)'] for r in results]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all candidates in blue
    ax.scatter(static_ZZ_vals, CR_coeff_vals, color='blue', s=50)

    # Annotate all points with parameters including f01
    texts = []
    for r in results:
        annotation = (f"EC={r['Flux_EC (GHz)']:.2f}, EJ={r['Flux_EJ (GHz)']:.2f}, "
                      f"EL={r['Flux_EL (GHz)']:.2f}, f01={r['f01 (GHz)']*1e3:.1f},f01_T={r['f01_T (GHz)']*1e3:.1f} ")
        texts.append(ax.text(r['static_ZZ (MHz)'], r['CR_coeff (MHz)'], annotation, fontsize=10))

    # Use adjustText to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray'))

    # Labels and title
    ax.set_xlabel('Static ZZ (MHz)', fontsize=12)
    ax.set_ylabel('CR Coefficient (MHz)', fontsize=12)
    ax.set_title('Static ZZ vs CR Rate with Varying Fluxonium and Transmon Parameters', fontsize=14)
    ax.grid(True)

    # Show the plot
    plt.show()


#%%

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file_path = r'Z:\Tanvir\Tanvir Files and codes\Codes_11_14_2024\qubit_results\CR_descending_Jc_25.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Step 2: Extract the columns you want to plot
f01_values = data['f01 (GHz)']
CR_values = data['CR_coeff (MHz)']
ZZ_values = data['static_ZZ (MHz)']

# Step 3: Create Subplots
plt.figure(figsize=(15, 6))

# Subplot 1: f01
plt.subplot(1, 3, 1)
plt.boxplot(f01_values, vert=True, patch_artist=True)
plt.title('f01 (GHz)')
plt.ylabel('Values')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 2: CR Coefficient
plt.subplot(1, 3, 2)
plt.boxplot(CR_values, vert=True, patch_artist=True)
plt.title('CR Coeff (MHz)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Subplot 3: Static ZZ
plt.subplot(1, 3, 3)
plt.boxplot(ZZ_values, vert=True, patch_artist=True)
plt.title('Static ZZ (MHz)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()
plt.show()


#%%
import scqubits as scq
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt


qbta = scq.Fluxonium(
    EC=1,
    EJ = 5 ,
    EL=.8,
    flux=0.5,  # flux frustration point
    cutoff=110,
    truncated_dim=10,
)


qbtb = scq.Transmon(
     EJ=4.5,
     EC=0.04,
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
A=.22*4
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

#%%
# new code with a json file for the differet types of devices 


import scqubits as scq
import qutip as qt
import numpy as np
import json
from matplotlib import pyplot as plt

# Load parameters from JSON
with open("Z:\Tanvir\Tanvir Files and codes\Codes_11_14_2024\qubit_params.json", "r") as file:
    params_all = json.load(file)

# Select parameter set
selected_set = "CR_type_FUN10"  # Change this to "CR_type_2" or others as needed
params = params_all[selected_set]

# Define Fluxonium
qbta = scq.Fluxonium(
    EC=params["fluxonium"]["EC"],
    EJ=params["fluxonium"]["EJ"],
    EL=params["fluxonium"]["EL"],
    flux=params["fluxonium"]["flux"],
    cutoff=params["fluxonium"]["cutoff"],
    truncated_dim=params["fluxonium"]["truncated_dim"]
)

# Define Transmon
qbtb = scq.Transmon(
    EJ=params["transmon"]["EJ"],
    EC=params["transmon"]["EC"],
    ng=params["transmon"]["ng"],
    ncut=params["transmon"]["ncut"],
    truncated_dim=params["transmon"]["truncated_dim"]
)

# Define the common Hilbert space
hilbertspace = scq.HilbertSpace([qbta, qbtb])

# Add interaction between two qubits
hilbertspace.add_interaction(
    g_strength=params["interaction"]["g_strength"],
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

# Truncate operators to desired dimension
def truncate(operator: qt.Qobj, dimension: int) -> qt.Qobj:
    return qt.Qobj(operator[:dimension, :dimension])

total_truncation = params["simulation"]["total_truncation"]
diag_dressed_hamiltonian_trunc = truncate(diag_dressed_hamiltonian, total_truncation)

evalues = (diag_dressed_hamiltonian_trunc.eigenenergies() - diag_dressed_hamiltonian_trunc.eigenenergies()[0]) / 6.28

n_a = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbta.n_operator)
n_b = hilbertspace.op_in_dressed_eigenbasis(op_callable_or_tuple=qbtb.n_operator)
n_a = truncate(n_a, total_truncation)
n_b = truncate(n_b, total_truncation)

product_states_unsorted = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 3), (2, 1), (0, 2), (3, 0), (1, 2)]
idxs_unsorted = [hilbertspace.dressed_index((s1, s2)) for (s1, s2) in product_states_unsorted]
paired_data = list(zip(idxs_unsorted, product_states_unsorted))
sorted_data = sorted(paired_data, key=lambda x: x[0])
product_states = [data[1] for data in sorted_data]
idxs = [data[0] for data in sorted_data]

states = [qt.basis(total_truncation, idx) for idx in idxs]

# Bare and dressed energy calculations
e_11 = evalues[hilbertspace.dressed_index((1, 1))]
e_10 = evalues[hilbertspace.dressed_index((1, 0))]
e_01 = evalues[hilbertspace.dressed_index((0, 1))]
e_00 = evalues[hilbertspace.dressed_index((0, 0))]
e_20 = evalues[hilbertspace.dressed_index((2, 0))]
e_30 = evalues[hilbertspace.dressed_index((3, 0))]
e_02 = evalues[hilbertspace.dressed_index((0, 2))]



Static_ZZ = e_11 - e_10 - e_01 + e_00

bare_states_a = qbta.eigenvals() - qbta.eigenvals()[0]
bare_states_b = qbtb.eigenvals() - qbtb.eigenvals()[0]

print('Static_ZZ(MHz)= ', Static_ZZ * 1e3)
print('bare_F_01 = ', bare_states_a[1])
print('bare_F_12 =', bare_states_a[2] - bare_states_a[1])
print('bare_T_01=', bare_states_b[1])
print('bare_F_03=', bare_states_a[3])
print('bare_T_12 =', bare_states_b[2] - bare_states_b[1])

print('dressed_F_01(GHz)= ', (e_10 - e_00) * 1)
print('dressed_F_12(GHz)= ', (e_20 - e_10) * 1)
print('dressed_F_03(GHz)= ', (e_30 - e_00) * 1)
print('dressed_T_01(GHz)= ', (e_01 - e_00) * 1)
print('dressed_T_12(GHz)= ', (e_02 - e_01) * 1)
print('Transmon alpha= ', (2 * e_01 - e_02) * 1)



n_a_00_01 = n_a[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]
n_b_00_01 = n_b[hilbertspace.dressed_index((0,0)),hilbertspace.dressed_index((0,1))]

eta = -n_a_00_01/n_b_00_01

# Drive parameters
drive_freq = evalues[hilbertspace.dressed_index((1, 1))] - evalues[hilbertspace.dressed_index((1, 0))]
A = params["drive"]["A"]

def cosine_drive(t: float, args: dict) -> float:  
    return A * np.cos(6.28 * drive_freq * t)

tlist = np.linspace(0, params["drive"]["duration"], int(params["drive"]["duration"]))

H_qbt_drive = [
    diag_dressed_hamiltonian_trunc,
    [2 * np.pi * (n_a +eta* n_b), cosine_drive],
]

result = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((1, 0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)
result2 = qt.sesolve(
    H_qbt_drive,
    qt.basis(total_truncation, hilbertspace.dressed_index((0, 0))),
    tlist,
    e_ops=[state * state.dag() for state in states]
)

# Plot results
plt.figure()
for idx, res in zip(idxs[:6], result.expect[:6]):
    plt.plot(tlist, res, label=f"|{product_states[idx][0]}{product_states[idx][1]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium) in state |1>")

plt.figure()
for idx, res in zip(idxs[:7], result2.expect[:7]):
    plt.plot(tlist, res, label=f"|{product_states[idx][0]}{product_states[idx][1]}>")

plt.legend()
plt.ylabel("population")
plt.xlabel("t (ns)")
plt.title("Control (Fluxonium) in state |0>")


#%%

#integer Fluxonium


import math

# Each device’s EJ, EC, EL in GHz (from your table):
devices = {
    'A':  (4.12, 1.64, 0.18),
    'B':  (3.84, 1.75, 0.14),
    'C':  (7.20, 2.04, 0.18),
    'D':  (6.78, 1.47, 0.22),
    'Fig1': (5.00, 1.50, 0.20),
    'E':  (3.955, .916, 0.36),
    'F':  (4.5, 1, 0.8),
    'G':  (6.4, .92, 0.55)
    
}

print("Checking condition: sqrt(8 * EJ * EC) >> 2 * pi^2 * EL\n")
for label, (EJ, EC, EL) in devices.items():
    # Left Hand Side
    lhs = math.sqrt(8.0 * EJ * EC)
    # Right Hand Side
    rhs = 2.0 * (math.pi**2) * EL
    
    ratio = lhs / rhs
    
    print(f"Device {label}:")
    print(f"  EJ = {EJ:.2f} GHz, EC = {EC:.2f} GHz, EL = {EL:.2f} GHz")
    print(f"  LHS = sqrt(8*EJ*EC)     = {lhs:.4f}")
    print(f"  RHS = 2*pi^2*EL         = {rhs:.4f}")
    print(f"  LHS / RHS = {ratio:.4f}")
    if ratio > 5:  # or whichever threshold you consider "much greater than"
        print("  --> Condition appears satisfied (LHS >> RHS)\n")
    else:
        print("  --> Condition is NOT strongly satisfied\n")
        
#%% 
