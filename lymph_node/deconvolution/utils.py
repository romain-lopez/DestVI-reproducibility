import hotspot
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import pandas as pd
from sklearn.mixture import GaussianMixture


#build the 2D colormap
import cmap2d 
tri_coords = [[-1,-1], [-1,1], [1, 0]]
tri_colors = [(1,0,0), (0,1,0), (0,0,1)]

import seaborn as sns
from matplotlib import pyplot as plt
sns.reset_orig()
sc.settings._vector_friendly = True
# p9.theme_set(p9.theme_classic)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.transparent"] = True
plt.rcParams["figure.figsize"] = (4, 4)

plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = 500
plt.rcParams["axes.titlepad"] = 8.0
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.labelweight"] = 500
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.labelpad"] = 6.0
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams["font.size"] = 11
# plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', "Computer Modern Sans Serif", "DejaVU Sans"]
plt.rcParams['font.weight'] = 500

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['xtick.minor.size'] = 1.375
plt.rcParams['xtick.major.size'] = 2.75
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['xtick.minor.pad'] = 2

plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['ytick.minor.size'] = 1.375
plt.rcParams['ytick.major.size'] = 2.75
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['ytick.minor.pad'] = 2

plt.rcParams["legend.fontsize"] = 12
plt.rcParams['legend.handlelength'] = 1.4
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.scatterpoints'] = 3

plt.rcParams['lines.linewidth'] = 1.7
DPI = 300

def flatten(x, threshold):
    return (x > threshold) * x

def form_stacked_quantiles(data, N=100):
    quantiles = np.quantile(data, np.linspace(0, 1, N, endpoint=False))
    return quantiles, np.vstack([flatten(data, q) for q in quantiles])

def get_autocorrelations(st_adata, stacked_quantiles, quantiles):
    # form dataframes
    loc = pd.DataFrame(data=st_adata.obsm["location"], index=st_adata.obs.index)
    df = pd.DataFrame(data=stacked_quantiles, columns=st_adata.obs.index, index=quantiles)
    # run hotspot
    hs = hotspot.Hotspot(df, model='none', latent=loc,)
    hs.create_knn_graph(
        weighted_graph=True, n_neighbors=10,
    )
    hs_results = hs.compute_autocorrelations(jobs=1)
    return hs_results.index.values, hs_results["Z"].values

def smooth_get_critical_points(x, noisy_data, k=5, s=0.1):
    f = splrep(x, noisy_data,k=5, s=1)
    smoothed = splev(x,f)
    derivative = splev(x,f,der=1)
    sign_2nd = splev(x,f,der=2) > 0
    curvature = splev(x,f,der=3)
    return noisy_data, smoothed, derivative, sign_2nd, curvature

def prettify_axis(ax, all_=False):
    if not all_:
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
def get_laplacian(s, pi):
    N = s.shape[0]
    dist_table = pdist(s)
    bandwidth = np.median(dist_table)
    sigma=(0.5*bandwidth**2)
    
    l2_square = squareform(dist_table)**2
    D = np.exp(- l2_square/ sigma) * np.dot(pi, pi.T)
    L = -D
    sum_D = np.sum(D, axis=1)
    for i in range(N):
            L[i, i] = sum_D[i]
    return L

def get_spatial_components(locations, proportions, data):
    # find top two spatial principal vectors
    # form laplacian
    L = get_laplacian(locations, proportions)
    # center data
    transla_ = data.copy()
    transla_ -= np.mean(transla_, axis=0)
    # get eigenvectors
    A = np.dot(transla_.T, np.dot(L, transla_))
    w, v = np.linalg.eig(A)
    # don't forget to sort them...
    idx = np.argsort(w)[::-1]
    vec = v[:, idx][:, :2]
    return vec

def vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def get_delta(lfc):
    return np.max(np.abs(GaussianMixture(n_components=3).fit(lfc.reshape(-1, 1)).means_))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list


def local_correlation_plot(
            local_correlation_z, modules, linkage,
            mod_cmap='tab10', vmin=-8, vmax=8,
            z_cmap='RdBu_r', gene_modules={}
):
    """
    Patched code from https://github.com/YosefLab/Hotspot/blob/master/hotspot/plots.py
    """

    row_colors = None
    colors = list(plt.get_cmap(mod_cmap).colors)
    module_colors = {i: colors[(i-1) % len(colors)] for i in modules.unique()}
    module_colors[-1] = '#ffffff'

    row_colors1 = pd.Series(
        [module_colors[i] for i in modules],
        index=local_correlation_z.index,
    )

    row_colors = pd.DataFrame({
        "Modules": row_colors1,
    })

    cm = sns.clustermap(
        local_correlation_z,
        row_linkage=linkage,
        col_linkage=linkage,
        vmin=vmin,
        vmax=vmax,
        cmap=z_cmap,
        xticklabels=False,
        yticklabels=False,
        row_colors=row_colors,
        rasterized=True,
    )
    
    fig = plt.gcf()
    plt.sca(cm.ax_heatmap)
    plt.ylabel("")
    plt.xlabel("")

    cm.ax_row_dendrogram.remove()

    # Add 'module X' annotations
    ii = leaves_list(linkage)

    mod_reordered = modules.iloc[ii]

    mod_map = {}
    y = np.arange(modules.size)

    for x in mod_reordered.unique():
        if x == -1:
            continue

        mod_map[x] = y[mod_reordered == x].mean()

    plt.sca(cm.ax_row_colors)
    for mod, mod_y in mod_map.items():
        plt.text(-.5, y=mod_y, s="Module {}".format(mod),
                 horizontalalignment='right',
                 verticalalignment='center')
    plt.xticks([])

    plt.sca(cm.ax_heatmap)
    for mod, mod_y in mod_map.items():
        if mod in gene_modules:
#             plt.arrow(local_correlation_z.shape[0],mod_y,100,0, width=1) 
#             plt.text(local_correlation_z.shape[0]+100, y=mod_y, s="\n".join(gene_modules[mod]),
#                      horizontalalignment='left',
#                      verticalalignment='center',
#                      fontsize="small",
#                      bbox= dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.annotate(text="\n".join(gene_modules[mod]), 
                         xy=(local_correlation_z.shape[0], mod_y), 
                         xytext=(local_correlation_z.shape[0] + 100, mod_y),
                     arrowprops=dict(arrowstyle="-[, widthB=1.5"),
                     horizontalalignment='left',
                     verticalalignment='center',
                     fontsize="small",
                     bbox= dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Find the colorbar 'child' and modify
    min_delta = 1e99
    min_aa = None
    for aa in fig.get_children():
        try:
            bbox = aa.get_position()
            delta = (0-bbox.xmin)**2 + (1-bbox.ymax)**2
            if delta < min_delta:
                delta = min_delta
                min_aa = aa
        except AttributeError:
            pass

    min_aa.set_ylabel('Z-Scores')
    min_aa.yaxis.set_label_position("left")