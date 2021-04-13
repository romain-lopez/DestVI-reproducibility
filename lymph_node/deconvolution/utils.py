import hotspot
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import pandas as pd


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