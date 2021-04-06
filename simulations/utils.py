from numba import jit
import numpy as np
from scipy.stats import spearmanr, pearsonr


#util functions for sampling
def categorical(p, n_samples):
    size = list(p.shape[:-1])
    size.insert(0, n_samples)
    return (p.cumsum(-1) >= np.random.uniform(size=size)[..., None]).argmax(-1).T

@jit(nopython=True)
def get_mean_normal(cell_types, gamma, mean_, components_):
    """
    Util for preparing the mean of the normal distribution.

    cell_types: (n_spots, n_cells)
    gamma: (n_spots, n_cells, n_latent)
    
    return: samples: (n_spots, n_cells, n_genes)
    """
    # extract shapes
    n_spots = gamma.shape[0]
    n_cells = gamma.shape[1]
    n_genes = components_[0].shape[1]
    
    mean_normal = np.zeros((n_spots, n_cells, n_genes))
    for spot in range(n_spots):
        for cell in range(n_cells):
            mean_normal[spot, cell] = mean_[cell_types[spot, cell]]
            c = components_[cell_types[spot, cell]]
            g = np.expand_dims(gamma[spot, cell], 0)
            mean_normal[spot, cell] += np.dot(g, c)[0]
    return mean_normal


def metrics_vector(groundtruth, predicted, scaling=1, feature_shortlist=None):
    res = {}
    if feature_shortlist is not None:
        # shortlist_features
        groundtruth = groundtruth[:, feature_shortlist].copy()
        predicted = predicted[:, feature_shortlist].copy()
    n = predicted.shape[0]
    g = predicted.shape[1]       
    # correlations metrics
    res["avg_spearman"] = np.nan_to_num(np.mean([spearmanr(groundtruth[:, i], predicted[:, i] ).correlation for i in range(g)]))
    res["avg_pearson"] = np.nan_to_num(np.mean([pearsonr(groundtruth[:, i], predicted[:, i])[0] for i in range(g)]))
    # error metrics
    res["median_l1"] = np.median(np.abs(scaling * groundtruth - scaling * predicted))
    res["mse"] = np.sqrt(np.mean((scaling * groundtruth - scaling * predicted)**2))
    return res

@jit(nopython=True)
def find_location_index_cell_type(locations, cell_type, loc_ref, ct_ref):
    """Return the indices for locations in query only if cell type matches."""
    out_a = [0]
    out_b = [0]
    for i in range(locations.shape[0]):
        for j in range(loc_ref.shape[0]):
            if np.all(locations[i] == loc_ref[j]):
                if cell_type == ct_ref[j]:
                    out_a += [i]
                    out_b += [j]
    return np.array(out_a[1:]), np.array(out_b[1:])

@jit(nopython=True)
def discrete_histogram(data, size):
    """
    Fast histogram in jit, looking at cell type abundance.

    data of shape (n_cells, n_neighbors), must be an integer
    """
    res = np.zeros((data.shape[0], size))
    for n in range(data.shape[0]):
        for k in range(data.shape[1]):
            res[n, data[n, k]] += 1
    return res / data.shape[1]