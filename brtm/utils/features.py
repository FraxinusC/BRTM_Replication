"""Feature construction helpers combining θ vectors & structured profiles."""
import numpy as np
import pandas as pd
from tqdm import tqdm

def feat(r, θD, θA, θB, guest_p, host_p, order_D, order_A, order_B):
    """Construct a feature vector from a single row"""
    g, l, h = int(r.guest_id), int(r.listing_id), int(r.host_id)
    
    # Concatenate features
    f = np.concatenate([
        θD.get(l, np.zeros(len(order_D))),  # Listing topic distribution
        θA.get(g, np.zeros(len(order_A))),  # Guest topic distribution
        θB.get(h, np.zeros(len(order_B))),  # Host topic distribution
        guest_p.loc[g].to_numpy() if g in guest_p.index else np.zeros(guest_p.shape[1]),  # Guest profile
        host_p.loc[h].to_numpy() if h in host_p.index else np.zeros(host_p.shape[1])      # Host profile
    ])
    return f

def XY(df, θD, θA, θB, guest_p, host_p, order_D, order_A, order_B):
    """Construct X and y from a DataFrame"""
    X = np.vstack([feat(r, θD, θA, θB, guest_p, host_p, order_D, order_A, order_B) for _, r in tqdm(df.iterrows(), total=len(df), desc="build X")])
    y = df.label.to_numpy(float)
    return X, y