from ba_network import ba_network
import matplotlib.pyplot as plt
import numpy as np
from logbin import logbin
from utils import data_folder, figures_folder
import pickle
import scipy as sp

n = 1000
m_values = [2, 4, 8, 16, 32]
repeats = 100
logbin_scale = 1.15
filename = 'ba_varying_m'
p_cutoff = 1e-3


def predicted_func(k):
    return 1/(k*(k+1)*(k+2))


fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

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
            G = ba_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.extend(degrees)
            print(m, "rep:", r)

        # saving
        with open(data_folder + filename + str('_%d' % m), "wb") as f:
            pickle.dump(data, f)

    x, y, yerr = logbin(data, logbin_scale)

    ax1.errorbar(x, y, yerr, color="C%i" % m_values.index(m),
                 linestyle='', elinewidth=1, alpha=0.8)
    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                marker='x', linewidths=1)  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(
        m), alpha=0.2)

    y_scaled = [i/(2*m*(m+1)) for i in y]
    yerr_scaled = [i/(2*m*(m+1)) for i in yerr]
    ax2.errorbar(x, y_scaled, yerr_scaled, color="C%i" %
                 m_values.index(m), linestyle='', elinewidth=1, alpha=0.7)
    ax2.scatter(x, y_scaled, label="m = %i" %
                m, color="C%i" % m_values.index(m), s=5, marker='x', linewidths=1)  # type:ignore
    ax2.plot(x, y_scaled, color="C%i" % m_values.index(m), alpha=0.2)
    print(m, "complete")

ax1.set_xscale('log')
ax1.set_yscale('log')
xmin1, xmax1 = ax1.get_xlim()
ax1.hlines([10**-3], xmin=xmin1, xmax=xmax1,
           color='k', linestyle='dashed', alpha=0.5)
ax1.set_xlim(xmin1, xmax1)
ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_title("(A)")

xmin2, xmax2 = ax2.get_xlim()
x_vals = np.linspace(m_values[0], xmax2, 1000)
ax2.hlines([10**-3], xmin=0, xmax=0,
           color='k', linestyle='dashed', alpha=0.5, label=r"$p(k) = 10^{-3}$")
ax2.plot(x_vals, predicted_func(x_vals),
         label=r'$\frac{1}{k(k+1)(k+2)}$', c='k')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel(r"$p(k) \, / \, 2m(m+1)$")
ax2.set_xlabel(r"$k$")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)
ax2.set_title("(B)")

plt.savefig(figures_folder + 'ba_varying_m.svg',
            format='svg', bbox_inches='tight')
