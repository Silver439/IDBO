import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import torch
import scipy.spatial as scpspatial
import plotly.graph_objects as go

plt.rcParams['font.sans-serif']=['Microsoft YaHei']

import plotly.graph_objects as go
import numpy as np

def plot_curves(name, obs, obs_M, total, random):
    fig = go.Figure()
    N = len(obs_M)
    x = np.arange(1, N+1, 1)

    fig.add_trace(go.Scatter(
        x=x,
        y=obs,
        mode='lines',
        name='obs',
        line=dict(color='rgba(138,43,226, 1)', width=3, dash='dash')  # 设置线条颜色、宽度和样式
    ))
    
    # 绘制 'obs_M' 曲线
    fig.add_trace(go.Scatter(
        x=x,
        y=obs_M,
        mode='lines',
        name='obs_M',
        line=dict(color='blue', width=3, dash='dashdot')  # 设置线条颜色和宽度
    ))
    
    # 绘制 'total' 曲线
    fig.add_trace(go.Scatter(
        x=x,
        y=total,
        mode='lines',
        name='total',
        line=dict(color='rgba(255, 0, 0, 0.7)', width=3)  # 设置线条颜色、宽度和样式
    ))

    fig.add_trace(go.Scatter(
        x=x,
        y=random,
        mode='lines',
        name='random',
        line=dict(color='rgba(34,139,34, 1)', width=3,dash='dot')  # 设置线条颜色、宽度和样式
    ))
    
    # 设置图表布局
    fig.update_layout(
        title=name,  # 设置标题
        xaxis_title='Iterations',  # 设置 x 轴标题
        title_x=0.5,  # 标题居中
        title_font_size=24,  # 设置标题字体大小
        yaxis_title='Value',  # 设置 y 轴标题
        template='plotly',  # 设置主题
        legend_title=None,  # 设置图例标题
        showlegend=True,  # 显示图例
        legend=dict(font=dict(size=16)),
        autosize=True
    )
    
    # 显示图表
    fig.show()


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