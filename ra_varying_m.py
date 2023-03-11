from ra_network import ra_network
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from logbin import logbin
from utils import data_folder, figures_folder
import pickle

n = 1000
m_values = [2, 4, 8, 16, 32, 64, 128]
repeats = 100
logbin_scale = 1.1
filename = 'ra_varying_m'

fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0:2])
ax2 = fig.add_subplot(gs[2])


def predicted_function(k, m):
    return (1/(m+1))*((m/(m+1))**(k-m))


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
    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5)
    ax1.plot(x, y, color="C%i" % m_values.index(m), alpha=0.5)
    ax1.plot(x, [predicted_function(k, m) for k in x], color="C%i" % m_values.index(m),
             linestyle='dotted')

    ax2.scatter(x, [collapse_function(p, m)
                for p in y], label=m, color="C%i" % m_values.index(m), s=5)
    ax2.plot(x, [collapse_function(p, m)
             for p in y], color="C%i" % m_values.index(m), alpha=0.5)

    print(m, "complete")

# ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel(r"$f(p, m)$")
ax2.set_xlabel(r"$k$")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., title='m')

plt.savefig(figures_folder + 'ra_varying_m.svg',
            format='svg', bbox_inches='tight')
