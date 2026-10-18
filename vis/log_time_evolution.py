import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "../data"

df_CUDA = pd.read_csv(DATA / "CUDA_data.csv")
df_MT = pd.read_csv(DATA / "MT_data.csv")
df_ST = pd.read_csv(DATA / "ST_data.csv")

N_CUDA = np.log(df_CUDA["N"])
N_MT = np.log(df_MT["N"])
N_ST = np.log(df_ST["N"])

t_CUDA = np.log(df_CUDA["Time"])
t_MT = np.log(df_MT["Time"])
t_ST = np.log(df_ST["Time"])

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#0d1117",
    "savefig.facecolor": "#0d1117",
    "axes.edgecolor": "#8b949e",
    "grid.color": "#30363d",
    "text.color": "#e6edf3",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "font.size": 12,
})

plt.scatter(N_CUDA, t_CUDA, label="CUDA", marker="x")
plt.scatter(N_MT, t_MT, label="Multithreading (Threads = 22)", marker="x")
plt.scatter(N_ST, t_ST, label="Single Threaded", marker="x")

plt.xlabel(r"$\ln(N_{sim})$")
plt.ylabel(r"$\ln(t)$ (ms)")
plt.title(r"$\ln(t)$ vs. $\ln(N_{sim})$")
plt.legend()

plt.show()
