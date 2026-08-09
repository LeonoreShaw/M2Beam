import pandas as pd
import numpy as np
import math
from pathlib import Path
path = Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_per_position.csv')
df = pd.read_csv(path)
lam_um = 10.6
cols = [c for c in df.columns if 'd4' in c and c.endswith('_um')]
print('cols', cols)
def compute(z_mm,d4sigma_um,div):
    z_um=z_mm*1000.0
    rho_um=d4sigma_um/div
    coeff=np.polyfit(z_um, rho_um**2, 2)
    a,b,c=coeff
    z0=-b/(2*a)
    rho0_sq=c-b*b/(4*a)
    rho0=math.sqrt(rho0_sq)
    theta=math.sqrt(a)
    return coeff,z0,rho0,theta,math.pi*rho0*theta/lam_um
for div in [2,4]:
    print('--- div',div)
    for col in cols:
        coeff,z0,rho0,theta,m2 = compute(df['z_mm'].values, df[col].values, div)
        print(col, 'M2=', m2, 'a=',coeff[0], 'rho0=',rho0, 'theta=',theta)
