import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import pandas as pd

# LaTeX向けフォント設定（必要に応じて調整）
#matplotlib.rcParams.update({
#    "text.usetex": False,  # TrueにするとLaTeXが必要
#    "font.family": "serif",
#    "font.size": 12
#})

data_file=['data/baseline']


for n in range(1):
    # CSVファイルの読み込み
    df = pd.read_csv(data_file[n]+'.csv', index_col=1)

    # グラフの作成
    plt.figure(figsize=(6, 4))
    plt.plot(df["Value"], label="Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(data_file[n]+'.png')
    plt.pause(1)

q = input('Hit enter to finish')
plt.close()
