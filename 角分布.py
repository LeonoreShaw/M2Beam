import time
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# 模拟一个 DataFrame
df = pd.DataFrame({'A': range(1000)})

def sub_process(val):
    time.sleep(0.01)
    return val * 2

# 假设你把 df 分成了 10 个批次
batch_size = 10
num_batches = len(df) // batch_size

# 用来收集所有批次结果的列表
all_results = []

# 外层用 tqdm 显示大进度
for i in tqdm(range(num_batches), desc="数据处理中", unit="items"):
    # 获取当前批次的数据
    batch_data = df['A'].iloc[i * batch_size : (i + 1) * batch_size]
    
    # 内部使用 Parallel 并行处理这一个批次
    results = Parallel(n_jobs=6)(
        delayed(sub_process)(val) for val in batch_data
    )
    
    # 将当前批次的结果存起来
    all_results.extend(results)

print("处理完毕，总结果数：", len(all_results))