import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "../data"

df_CUDA = pd.read_csv(DATA / "CUDA_data.csv")
df_MT = pd.read_csv(DATA / "MT_data.csv")
df_ST = pd.read_csv(DATA / "ST_data.csv")

N_CUDA = (df_CUDA["N"])
N_MT = (df_MT["N"])
N_ST = (df_ST["N"])

t_CUDA = (df_CUDA["Time"])
t_MT = (df_MT["Time"])
t_ST = (df_ST["Time"])

m_CUDA, b_CUDA = np.polyfit(N_CUDA, t_CUDA, 1)
m_MT, b_MT = np.polyfit(N_MT, t_MT, 1)
m_ST, b_ST = np.polyfit(N_ST, t_ST, 1)

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

plt.plot(N_CUDA, m_CUDA * N_CUDA + b_CUDA, label=f"CUDA, y = {m_CUDA:.2e} x + {b_CUDA:.2e}")
plt.plot(N_MT, m_MT * N_MT + b_MT, label=f"Multithreading, y = {m_MT:.2e} x + {b_MT:.2e}")
plt.plot(N_ST, m_ST * N_ST + b_ST, label=f"Single Threaded, y = {m_ST:.2e} x + {b_ST:.2e}")

plt.xlabel(r"$N_{sim}$")
plt.ylabel(r"$t$ (ms)")
plt.title(r"$t$ vs. $N_{sim}$")
plt.legend()

plt.show()
