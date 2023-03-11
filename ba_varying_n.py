from ba_network import ba_network
import matplotlib.pyplot as plt
from logbin import logbin
from utils import data_folder, figures_folder
from scipy import optimize
import pickle
import numpy as np

m = 10
n_values = [1000000, 100000, 10000, 1000, 100]
repeats = 100
logbin_scale = 1.15
filename = 'ba_varying_n'
D = 0.5


def p_infinity(k, m):
    return 2*m*(m+1)/((k)*(k+1)*(k+2))


def p_infinity_collapsed(m):
    return 2*m*(m+1)


def k1(n):
    return -0.5 + np.sqrt(0.25 + n*m*(m+1))


fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for n in n_values:
    data = []
    # if saved data, use that, else generate new data
    try:
        # attempt to load the data file
        with open(data_folder + filename + str('_%d' % n), "rb") as f:
            data = pickle.load(f)
    except:
        # generate the data
        for r in range(repeats):
            G = ba_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.extend(degrees)
            print(n, 'rep:', r)

        # saving
        with open(data_folder + filename + str('_%d' % n), "wb") as f:
            pickle.dump(data, f)

    x, y, _ = logbin(data, logbin_scale)
    ax1.scatter(x, y, label=r"N = $10^{%i}$" % np.log10(
        n), color="C%i" % n_values.index(n), s=3)
    ax1.plot(x, y, color="C%i" % n_values.index(n), alpha=0.5)

    scaled_y = [y[i]*x[i]*(x[i] + 1)*(x[i] + 2) for i in range(len(y))]
    scaled_x = [x[i]/(n**D) for i in range(len(y))]
    ax2.scatter(scaled_x, scaled_y, label=r"N = $10^{%i}$" % np.log10(n), color="C%i" %
                n_values.index(n), s=3)
    ax2.plot(scaled_x, scaled_y, color="C%i" % n_values.index(n), alpha=0.5)
    print(n, "complete")

x_min1, x_max1 = ax1.get_xlim()
x_vals1 = np.linspace(m, x_max1, 1000)
ax1.plot(x_vals1, p_infinity(x_vals1, m), c='k',
         label=r'$p_\infty$', linestyle='dashed', alpha=0.3)
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel("$k$")
ax1.set_ylabel("$p(k)$")


x_min2, x_max2 = ax2.get_xlim()
x_vals2 = np.linspace(m/n_values[0]**0.5, x_max2, 1000)
ax2.plot(x_vals2, [p_infinity_collapsed(m)
         for _ in x_vals2], c='k', label=r'$p_\infty$', linestyle='dashed', alpha=0.3)
ax2.legend()
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r"$k \, / \, L^{1/2}$")
ax2.set_ylabel(r"$p(k) \, \cdot \, k(k+1)(k+2)$")

plt.savefig(figures_folder + 'ba_varying_n.svg',
            format='svg', bbox_inches='tight')
