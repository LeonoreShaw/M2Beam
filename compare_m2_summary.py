import pandas as pd
import numpy as np
import math
from pathlib import Path

path1 = Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_per_position.csv')
path2 = Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_fit_summary.csv')

df = pd.read_csv(path1)
fit = pd.read_csv(path2)
fit.columns = fit.columns.str.strip()
lam_um = 10.6

def notebook_m2(z_mm, d4sigma_um):
    z_um = z_mm * 1000.0
    rho_um = d4sigma_um / 2.0
    a, b, c = np.polyfit(z_um, rho_um**2, 2)
    rho0 = math.sqrt(c - b*b/(4*a))
    theta = math.sqrt(a)
    return np.pi * rho0 * theta / lam_um, rho0, theta, a, b, c

print('notebook calc:')
for axis in ['x', 'y']:
    col = f'csv_d4{axis}_um'
    m2, rho0, theta, a, b, c = notebook_m2(df['z_mm'].values, df[col].values)
    print(axis, 'm2=', m2, 'rho0=', rho0, 'theta=', theta, 'a=', a)

print('\nsummary file:')
for axis in ['x', 'y']:
    row = fit[(fit.dataset.str.strip() == 'csv_original') & (fit.direction.str.strip() == axis)].iloc[0]
    theta_rad = row.theta_um_per_mm / 1000.0
    rho0 = row.d0_um / 2.0
    m2_formula = np.pi * rho0 * theta_rad / lam_um
    print(axis, 'summary m2=', row.m2, 'd0=', row.d0_um, 'theta_um_per_mm=', row.theta_um_per_mm, 'formula=', m2_formula)
