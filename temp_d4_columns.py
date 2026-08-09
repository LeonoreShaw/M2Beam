import pandas as pd
from pathlib import Path
path = Path(r'F:/CodexProjects/光斑图片/DataBase/analysis/beam_m2_per_position.csv')
df = pd.read_csv(path)
cols = [c for c in df.columns if 'd4' in c and c.endswith('_um')]
print(cols)
for c in cols:
    print(c, df[c].dtype, df[c].isna().sum(), df[c].min(), df[c].max())
