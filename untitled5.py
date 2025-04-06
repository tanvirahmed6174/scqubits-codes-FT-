# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:15:14 2025


Circuit Quantization of the FTF circuit

@author: Tanvir
"""

import sympy as sp

# Define symbolic variables for the updated capacitance matrix C
C1, C2, C3, C4, C5 = sp.symbols('C1 C2 C3 C4 C5')
C12, C13, C14, C15, C23, C24, C25, C34, C35, C45 = sp.symbols('C12 C13 C14 C15 C23 C24 C25 C34 C35 C45')

# Define the updated capacitance matrix C
C = sp.Matrix([
    [C1 + C12 + C13 + C14 + C15, -C12, -C13, -C14, -C15],
    [-C12, C2 + C12 + C23 + C24 + C25, -C23, -C24, -C25],
    [-C13, -C23, C3 + C13 + C23 + C34 + C35, -C34, -C35],
    [-C14, -C24, -C34, C4 + C14 + C24 + C34 + C45, -C45],
    [-C15, -C25, -C35, -C45, C5 + C15 + C25 + C35 + C45]
])

# Define the M matrix
M = sp.Matrix([
    [1, 1, 0, 0, 0],
    [1, -1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, -1]
])

# Calculate M inverse
M_inv = M.inv()

# Calculate the transformed capacitance matrix C_tilde
C_tilde = M_inv * C * M_inv

# Discard the 1st and 4th rows and columns to get the 3x3 matrix
tilde_C_3x3 = C_tilde.extract([1, 2, 4], [1, 2, 4])  # Keeping only the differential modes

# Calculate the inverse of the 3x3 matrix
tilde_C_3x3_inv = tilde_C_3x3.inv()

# Display the result
sp.pprint(tilde_C_3x3_inv, use_unicode=True)

