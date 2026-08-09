import time
import pandas as pd
from tqdm.contrib.concurrent import process_map

# 1. 定义你的处理函数（必须在外面，顶层定义）
def sub_process(val):
    time.sleep(0.1)
    return val * 2

# 2. 关键：必须加上这行保护入口！
if __name__ == '__main__':
    # 模拟 DataFrame
    df = pd.DataFrame({'A': range(100)})

    # 3. 多进程处理逻辑放在这里面
    results = process_map(
        sub_process, 
        df['A'].tolist(), 
        max_workers=6, 
        desc="数据处理中"
    )

    print("处理完成，结果：", results[:10])