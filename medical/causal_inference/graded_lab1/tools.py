import numpy as np
from scipy.stats import norm
import networkx as nx
import matplotlib.pyplot as plt


def fisherz_test(df, x, y, cond_list=None):
    """
    Test if X is independent of Y given a list of conditioning variables
    :param df: pandas Dataframe
    :param x: Column name for X
    :param y: Column name for Y
    :param cond_list: List of column names of conditioning variables
    :return: p value
    """
    if cond_list is None:
        cond_list = []
    else:
        cond_list = cond_list
    list_nodes = [x, y] + cond_list
    df = df[list_nodes]
    a = df.values.T

    if len(cond_list) > 0:
        cond_list_int = [i + 2 for i in range(len(cond_list))]
    else:
        cond_list_int = []

    correlation_matrix = np.corrcoef(a)
    var = list((0, 1) + tuple(cond_list_int))
    sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
    if np.linalg.det(sub_corr_matrix) == 0:
        r = 1
    else:
        inv = np.linalg.inv(sub_corr_matrix)
        r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])

    z = 0.5 * np.log((1 + r) / (1 - r))
    pval = np.sqrt(df.shape[0] - len(cond_list) - 3) * abs(z)
    pval = 2 * (1 - norm.cdf(abs(pval)))
    return pval


def plot_graph(g, title='', unmeasured=None):
    """
    :param g: Networkx DiGraph
    :param title: Title of the Figure
    :param unmeasured: list of node names to display as gray circles
    """
    pos = nx.circular_layout(g)
    plt.title(title)
    
    node_colors = []
    edge_colors = []
    for n in g.nodes:
        if unmeasured and n in unmeasured:
            node_colors.append('lightgray')
            edge_colors.append('black') 
        else:
            node_colors.append('white')
            edge_colors.append('none')  
    nx.draw(g, pos, with_labels=True, font_weight='bold',
            node_color=node_colors, node_shape='o', edgecolors=edge_colors, node_size=1000)
    
    plt.show()