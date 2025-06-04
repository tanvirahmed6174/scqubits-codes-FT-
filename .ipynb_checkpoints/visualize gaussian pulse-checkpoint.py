# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:38:14 2024

@author: WANGLAB
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Gaussian pulse
A = 0.02417  # GHz
tg = 100  # ns
omega_1112 = 2 * np.pi * 5.0  # Example frequency in GHz (set to 5 GHz for visualization)

# Define the Gaussian pulse envelope
def drive_coeff(t, tg, A, omega_1112):
    return A * np.exp(-8 * t * (t - tg) / tg**2) * np.cos(omega_1112 * t)

# Time points for visualization
t = np.linspace(0, 1000, 1000)  # 0 to 2*tg with 1000 points

# Compute the drive coefficient values
drive_values = drive_coeff(t, tg, A, omega_1112)

# Plot the Gaussian drive
plt.figure(figsize=(8, 5))
plt.plot(t, drive_values, label="Gaussian Drive", color='blue')
plt.title("Gaussian Drive Pulse")
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude (GHz)")
plt.grid(True)
plt.legend()
plt.show()

#%%

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Function to calculate and plot the Gaussian drive pulse
def plot_gaussian_drive():
    # Get user inputs
    try:
        A = float(amplitude_entry.get())
        tg = float(tg_entry.get())
        omega_1112 = float(frequency_entry.get()) * 2 * np.pi
        total_time = float(total_time_entry.get())
    except ValueError:
        status_label.config(text="Please enter valid numbers.", fg="red")
        return

    # Time points for visualization
    t = np.linspace(0, total_time, 1000)

    # Compute the drive coefficient values
    drive_values = A * np.exp(-8 * t * (t - tg) / tg**2) * np.cos(omega_1112 * t)

    # Plot on the canvas
    ax.clear()
    ax.plot(t, drive_values, label="Gaussian Drive", color="blue")
    ax.set_title("Gaussian Drive Pulse")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (GHz)")
    ax.grid(True)
    ax.legend()
    canvas.draw()
    status_label.config(text="Plot updated successfully.", fg="green")

# Create the main application window
root = tk.Tk()
root.title("Gaussian Drive Visualizer")

# Create input fields and labels
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

ttk.Label(frame, text="Amplitude (A):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
amplitude_entry = ttk.Entry(frame, width=10)
amplitude_entry.grid(row=0, column=1, padx=5, pady=5)
amplitude_entry.insert(0, "0.02417")  # Default value

ttk.Label(frame, text="Pulse Duration (tg, ns):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
tg_entry = ttk.Entry(frame, width=10)
tg_entry.grid(row=1, column=1, padx=5, pady=5)
tg_entry.insert(0, "100")  # Default value

ttk.Label(frame, text="Frequency (GHz):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
frequency_entry = ttk.Entry(frame, width=10)
frequency_entry.grid(row=2, column=1, padx=5, pady=5)
frequency_entry.insert(0, "5.0")  # Default value

ttk.Label(frame, text="Total Time (ns):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
total_time_entry = ttk.Entry(frame, width=10)
total_time_entry.grid(row=3, column=1, padx=5, pady=5)
total_time_entry.insert(0, "200")  # Default value

# Button to generate the plot
plot_button = ttk.Button(frame, text="Generate Plot", command=plot_gaussian_drive)
plot_button.grid(row=4, column=0, columnspan=2, pady=10)

# Status label
status_label = ttk.Label(frame, text="", foreground="green")
status_label.grid(row=5, column=0, columnspan=2, pady=5)

# Matplotlib figure and canvas for the plot
fig, ax = plt.subplots(figsize=(8, 5))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=1, column=0)

# Start the Tkinter event loop
root.mainloop()

#%%

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
t = 0.99  # Transmission coefficient
r = np.sqrt(1 - t**2)  # Reflection coefficient
squeezing_param = 1.0  # Squeezing parameter

# Create squeezed vacuum state and vacuum state
N = 30  # Truncation for Fock space
squeezed_vacuum = squeeze(N, squeezing_param) * basis(N, 0)  # Squeezed vacuum
vacuum_state = basis(N, 0)  # Vacuum state

# Combine into a two-mode input state
state_after_bs = tensor(squeezed_vacuum, vacuum_state)

# Define beamsplitter operators for two modes
BS = tensor(create(N) * t + destroy(N) * r, destroy(N) * t - create(N) * r)  # Full beamsplitter operator

# Apply beamsplitter transformation
transformed_state = BS * state_after_bs

# Post-measurement state
# Assuming a single photon is detected in the reflected mode
reflected_mode = fock(N, 1)  # Single-photon detection in reflected mode
post_measurement_state = (tensor(reflected_mode.dag(), qeye(N)) * transformed_state).unit()

# Wigner function for the transmitted mode
rho_transmitted = ptrace(post_measurement_state, 1)  # Partial trace for transmitted mode
x = np.linspace(-5, 5, 200)
p = np.linspace(-5, 5, 200)
W = wigner(rho_transmitted, x, p)

# Plot the Wigner function
plt.figure(figsize=(8, 6))
plt.contourf(x, p, W, 100, cmap="RdBu")
plt.colorbar(label="Wigner function value")
plt.xlabel(r"$x$")
plt.ylabel(r"$p$")
plt.title("Wigner Function of the Post-Measurement State")
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters for the beamsplitter and input states
t = 0.99  # Transmission coefficient
r = np.sqrt(1 - t**2)  # Reflection coefficient
squeezing_param = 1.0  # Squeezing parameter
alpha = 2.0  # Coherent state amplitude
N = 30  # Dimension of Fock space

# Define the input states
squeezed_vacuum = squeeze(N, squeezing_param) * basis(N, 0)  # Squeezed vacuum state
coherent_state = coherent(N, alpha)  # Coherent state

# Define the two-mode input state
input_state = tensor(squeezed_vacuum, coherent_state)

# Define the beamsplitter operator
BS_a = t * create(N) + r * destroy(N)  # Reflected mode operator
BS_b = t * destroy(N) - r * create(N)  # Transmitted mode operator
BS = tensor(BS_a, BS_b)  # Full beamsplitter operator

# Apply the beamsplitter transformation
output_state = BS * input_state

# Partial trace to verify each mode (optional)
rho_a = ptrace(output_state, 0)  # Density matrix of reflected mode
rho_b = ptrace(output_state, 1)  # Density matrix of transmitted mode

# Print the state description
print("State after the beamsplitter transformation:")
print(output_state)
