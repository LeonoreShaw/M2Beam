import numpy as np
import pandas as pd
import math
from pathlib import Path
path = Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_per_position.csv')
df = pd.read_csv(path)
lam_um = 10.6

def compute(z_mm, d4sigma_um, div):
    z_um = z_mm * 1000.0
    rho_um = d4sigma_um / div
    coeff = np.polyfit(z_um, rho_um**2, 2)
    a, b, c = coeff
    z0_um = -b / (2*a)
    rho0_sq = c - b*b/(4*a)
    rho0_um = math.sqrt(rho0_sq)
    theta_rad = math.sqrt(a)
    m2 = math.pi * rho0_um * theta_rad / lam_um
    return m2, coeff, z0_um, rho0_um, theta_rad

for axis in ['x','y']:
    col = 'csv_d4' + axis + '_um'
    for div in [2,4]:
        m2, coeff, z0, rho0, theta = compute(df['z_mm'].values, df[col].values, div)
        print(f'axis={axis}, div={div}, M2={m2:.4f}, z0={z0:.1f}, rho0={rho0:.1f}, theta={theta:.4e}')
