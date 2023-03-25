from ra_network import ra_network
import matplotlib.pyplot as plt
from linbin import linbin
from utils import data_folder, figures_folder
import pickle
import numpy as np

m = 10
n_values = [100000, 10000, 1000, 100]
repeats = 1000
num_bins = 30
filename = 'ra_varying_n'


def p_infinity(k, m):
    return (1/(m+1))*(m/(m+1))**(k-m)


def scale_y(p, m, k):
    return p/((1/(m+1))*(m/(m+1))**(k-m))


def scale_x(k, n, a, b):
    return k/(11.5*np.log(n)-13.3)
    # return k/np.log(n)


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
            G = ra_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.extend(degrees)
            print(n, 'rep:', r)

        # saving
        with open(data_folder + filename + str('_%d' % n), "wb") as f:
            pickle.dump(data, f)

    binwidth = (max(data) - min(data))/num_bins
    x, y, yerr = linbin(data, binwidth)
    ax1.errorbar(x, y, yerr, color="C%i" %
                 n_values.index(n), linestyle='', alpha=0.7)
    ax1.scatter(x, y, label=r"N = $10^{%i}$" % np.log10(
        n), color="C%i" % n_values.index(n), s=5, marker='x', linewidths=1)  # type:ignore
    ax1.plot(x, y, color="C%i" % n_values.index(n), alpha=0.2)

    scaled_y = [scale_y(y[i], m, x[i]) for i in range(len(y))]
    scaled_yerr = [scale_y(yerr[i], m, x[i]) for i in range(len(y))]
    scaled_x = [scale_x(x[i], n, 1.5, 0.3) for i in range(len(y))]
    ax2.errorbar(scaled_x, scaled_y, scaled_yerr, color="C%i" %
                 n_values.index(n), linestyle='', alpha=0.7)  # type:ignore
    ax2.scatter(scaled_x, scaled_y, label=r"N = $10^{%i}$" % np.log10(n), color="C%i" %
                n_values.index(n), s=5, marker='x', linewidths=1)  # type:ignore
    ax2.plot(scaled_x, scaled_y, color="C%i" % n_values.index(n), alpha=0.2)
    print(n, "complete")


ax1.set_yscale('log')
x_min1, x_max1 = ax1.get_xlim()
x_vals1 = np.linspace(x_min1, x_max1, 1000)
ax1.plot(x_vals1, p_infinity(x_vals1, m), c='k',
         label=r'$p_\infty$', linestyle='dashed', alpha=0.3, zorder=-10)
ax1.set_xlim(x_min1, x_max1)
ax1.set_xlabel("$k$")
ax1.set_ylabel("$p(k)$")
ax1.set_title("(A)")

ax2.set_yscale('log')
x_min2, x_max2 = ax2.get_xlim()
x_vals2 = np.linspace(x_min2, x_max2, 10)
ax2.plot(x_vals2, [1
         for _ in x_vals2], c='k', label=r'$p_\infty(k)$', linestyle='dashed', alpha=0.3, zorder=-10)
ax2.set_xlim(x_min2, x_max2)
ax2.set_xlabel(r"$k \, / \, k_1$")
ax2.set_ylabel(r"$p(k) \, / \, p_\infty (k)$")
ax2.set_title("(B)")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., markerscale=2)

plt.savefig(figures_folder + 'ra_varying_n.svg',
            format='svg', bbox_inches='tight')
