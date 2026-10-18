import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def black_scholes(S,K,r,sigma,T):
    d1 = (np.log(S/K) + (r+sigma**2) / 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

V = black_scholes(100,100,0.01,0.3,1)
BASE = Path(__file__).resolve().parent
DATA = BASE / "../data"

df_CUDA = pd.read_csv(DATA / "CUDA_data.csv")
df_MT = pd.read_csv(DATA / "MT_data.csv")
df_ST = pd.read_csv(DATA / "ST_data.csv")

t_CUDA = (df_CUDA["Time"])
t_MT = (df_MT["Time"])
t_ST = (df_ST["Time"])

c_CUDA = (df_CUDA["Call"])
c_MT = (df_MT["Call"])
c_ST = (df_ST["Call"])

abs_error_CUDA = np.abs(c_CUDA - V)
abs_error_MT = np.abs(c_MT - V)
abs_error_ST = np.abs(c_ST - V)

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

plt.scatter(t_CUDA, abs_error_CUDA, label="CUDA", marker="x")
plt.scatter(t_MT, abs_error_MT, label="Multithreading (Threads = 22)", marker="x")
plt.scatter(t_ST, abs_error_ST, label="Single Threaded", marker="x")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$t$")
plt.ylabel(r"$|V_{MC} - M_{BS}|$")
plt.title(r"$|V_{MC} - M_{BS}|$ vs. $t$ for call options")

plt.legend()

plt.show()
