from ra_network import ra_network
import matplotlib.pyplot as plt
import numpy as np
from logbin import logbin
from utils import data_folder, figures_folder
import pickle

n = 1000
m_values = [2, 4, 8, 16, 32, 64, 128]
repeats = 100
logbin_scale = 1.1
filename = 'ra_varying_m'


def collapse_function(p, m):
    return np.log(p*(m+1)*((m/(m+1))**m))/np.log(m/(m+1))


for m in m_values:
    data = []
    # if saved data, use that, else generate new data
    try:
        # attempt to load the data file
        with open(data_folder + filename + str('_%d' % m), "rb") as f:
            data = pickle.load(f)
    except:
        # generate the data
        for r in range(repeats):
            G = ra_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.extend(degrees)
            print(m, "rep:", r)

        # saving
        with open(data_folder + filename + str('_%d' % m), "wb") as f:
            pickle.dump(data, f)

    x, y, _ = logbin(data, logbin_scale)
    plt.scatter(x, [collapse_function(p, m) for p in y], label="m = %i" %
                m, color="C%i" % m_values.index(m), s=5)
    plt.plot(x, [collapse_function(p, m)
             for p in y], color="C%i" % m_values.index(m), alpha=0.5)

    # plt.plot(x, [predicted_function(k, m) for k in x], color="C%i" % m_values.index(m),
    #          linestyle='dotted', label="Predicted, m = %i" % m)
    print(m, "complete")

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$p(k)$")
plt.xlabel(r"$k$")

plt.savefig(figures_folder + 'ra_varying_m_data_collapse.svg',
            format='svg', bbox_inches='tight')
