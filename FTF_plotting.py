# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:58:19 2025

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

# Calculate bare energy levels
bare_levels_A = qbta.eigenvals()[:5]  # First 5 energy levels for Fluxonium A
bare_levels_B = qbtc.eigenvals()[:5]  # First 5 energy levels for Fluxonium B
bare_levels_T = qbtb.eigenvals()[:5]  # First 5 energy levels for Transmon

# Print the energy levels for verification
print("Bare levels for Fluxonium A:", bare_levels_A)
print("Bare levels for Transmon:", bare_levels_T)
print("Bare levels for Fluxonium B:", bare_levels_B)
