import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import torch
import scipy.spatial as scpspatial

plt.rcParams['font.sans-serif']=['Microsoft YaHei']

def plot_SimpleRegret(result_list,n_calls,true_minimum=None, yscale=None, title="Simple regret plot"): 
    
    plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"Simple regret")
    ax.grid()
    if yscale is not None:
        ax.set_yscale(yscale)
    colors = cm.hsv(np.linspace(0.25, 1.0, len(result_list)))

    for results, color in zip(result_list, colors):
        name, results = results
        iterations = range(1, n_calls + 1)
        regrets = [[np.min(r[:i])-true_minimum for i in iterations]
                for r in results]
        mu = np.mean(regrets, axis=0)
        #std = np.std(regrets, axis=0)
        plt.plot(iterations, mu, c=color, lw=1, label=name)
        #plt.fill_between(iterations, mu + std, mu - std, alpha=0.2, color='#9FAEB2', lw=0)
    ax.legend(loc="best")
    plt.savefig(title+'_sim.png')
    return ax

def plot_InstantRegret(result_list,n_calls,true_minimum=None, yscale=None, title="Instant regret plot"): 

    plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel(r"Instant regret")
    ax.grid()
    if yscale is not None:
        ax.set_yscale(yscale)
    colors = cm.hsv(np.linspace(0.25, 1.0, len(result_list)))

    minindex = []

    for results, color in zip(result_list, colors):
        name, results = results
        iterations = range(0, n_calls)
        regrets = [[r[i]-true_minimum for i in iterations]
                for r in results]
        mu = np.mean(regrets, axis=0)
        plt.plot(iterations, mu, c=color, lw=1, label=name)
        minindex.append((np.argmin(mu),np.min(mu)+true_minimum))
    ax.legend(loc="best")
    plt.savefig(title+'_ins.png')
    return minindex

def visualize_2d_contour(name,x1_range,x2_range,data,train_x,suggest=None,labelx="Training Points"):

    X1, X2 = torch.meshgrid(x1_range, x2_range, indexing="ij")
    data_reshaped = data.reshape(X1.shape)
    labelmin = ' '

    if suggest is None:
        min_value = data_reshaped.min()
        min_idx = np.unravel_index(data_reshaped.argmin(), data_reshaped.shape)
        min_x1 = X1[min_idx]
        min_x2 = X2[min_idx]
        labelmin = "Min Value"
    else:
        min_value = suggest[1]
        min_x1 = suggest[0][0]
        min_x2 = suggest[0][1]
        labelmin = "Suggest point"

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X1, X2, data_reshaped, cmap="coolwarm", levels=50)
    plt.colorbar(contour, ax=ax, label="object value")

    ax.scatter(train_x[:, 0], train_x[:, 1], color="black", marker="o", label=labelx)

    ax.scatter(min_x1, min_x2, color="red", marker="*", s=200, label=labelmin)
    ax.text(
        min_x1, min_x2, 
        f"loc: ({min_x1:.2f}, {min_x2:.2f})\nvalue: {min_value:.2f}", 
        color="red", fontsize=10, ha="left", va="bottom"
    )

    ax.set_title(name, fontsize=16)
    ax.set_xlabel("X1", fontsize=12)
    ax.set_ylabel("X2", fontsize=12)
    ax.legend()
    plt.tight_layout()

    plt.show()
    plt.close(fig)  