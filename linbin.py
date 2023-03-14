# designed to be interchangeable with logbin.py

import numpy as np


def linbin(data, bin_width: int = 10):
    counts = np.bincount(data).astype('float')
    total = np.sum(counts)
    data_min = min(data)
    data_max = max(data)
    num_bins = np.ceil((data_max - data_min)/bin_width)
    bin_edges = np.arange(data_min, data_min + (num_bins+2) *
                          bin_width, bin_width).astype('uint64')
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    y = np.zeros_like(x)
    y_err = np.zeros_like(x)
    for i in range(len(y)):
        y_counts = counts[bin_edges[i]:bin_edges[i+1]]
        y[i] = np.sum(y_counts /
                      (bin_edges[i+1] - bin_edges[i]))
        if np.sum(y_counts) == 0:
            y_err[i] = 0
        else:
            y_std = np.std(y_counts) / \
                (bin_edges[i+1] - bin_edges[i])
            y_err[i] = y_std / \
                np.sqrt(np.sum(y_counts))
    y /= total
    y_err /= total
    x = x[y != 0]
    y_err = y_err[y != 0]
    y = y[y != 0]
    return x, y, y_err
