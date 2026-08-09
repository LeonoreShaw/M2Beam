import pandas as pd
import numpy as np
import math
from pathlib import Path
path = Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_per_position.csv')
df = pd.read_csv(path)
lam_um = 10.6

def compute(z_mm, d4sigma_um):
    z_um = z_mm * 1000.0
    rho_um = d4sigma_um / 2.0
    coeff = np.polyfit(z_um, rho_um**2, 2)
    a, b, c = coeff
    z0_um = -b / (2*a)
    rho0_sq = c - b*b/(4*a)
    rho0_um = math.sqrt(rho0_sq)
    theta_rad = math.sqrt(a)
    m2 = math.pi * rho0_um * theta_rad / lam_um
    return a, b, c, z0_um, rho0_um, theta_rad, m2

for axis in ['x', 'y']:
    col = f'csv_d4{axis}_um'
    a,b,c,z0,rho0,theta,m2 = compute(df['z_mm'].values, df[col].values)
    print(axis)
    print('  a=', a)
    print('  b=', b)
    print('  c=', c)
    print('  z0_um=', z0)
    print('  rho0_um=', rho0)
    print('  theta_rad=', theta)
    print('  M2=', m2)
