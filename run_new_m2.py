import numpy as np
import pandas as pd
from pathlib import Path

lam_um = 10.6

df = pd.read_csv(Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_per_position.csv'))
cols = [c for c in df.columns if 'd4' in c and c.endswith('_um')]
for col in cols:
    z_um = df['z_mm'].values * 1000.0
    rho_um = df[col].values / 2.0
    coeff = np.polyfit(z_um, rho_um**2, 2)
    a,b,c = coeff
    z0_um = -b/(2*a)
    rho0_sq = c - b*b/(4*a)
    rho0_um = np.sqrt(rho0_sq)
    theta_rad = np.sqrt(a)
    m2 = np.pi * rho0_um * theta_rad / lam_um
    print(f'{col}: M2={m2:.6f}, z0={z0_um:.3f} um, rho0={rho0_um:.3f} um')

fit_df = pd.read_csv(Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_fit_summary.csv'))
fit_df.columns = fit_df.columns.str.strip()
print('\nsummary file calculation:')
def compute_m2_from_summary(row):
    rho0_um = row.d0_um / 2.0
    theta_rad = row.theta_um_per_mm / 2000.0
    m2_calc = np.pi * rho0_um * theta_rad / lam_um
    return m2_calc
fit_df['m2_calc'] = fit_df.apply(compute_m2_from_summary, axis=1)
print(fit_df[['dataset','direction','m2','m2_calc','d0_um','theta_um_per_mm','z0_mm']].to_string(index=False))
