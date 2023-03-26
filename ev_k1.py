from ev_network import ev_network
import matplotlib.pyplot as plt
from utils import data_folder, figures_folder
import pickle
import numpy as np
from scipy.optimize import curve_fit

m = 12
r = int(m/3)
n_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
repeats = 100
filename = 'ev_k1'

fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

k1_values = []
for n in n_values:
    data = []
    # if saved data, use that, else generate new data
    try:
        # attempt to load the data file
        with open(data_folder + filename + str('_%d' % n), "rb") as f:
            data = pickle.load(f)
    except:
        # generate the data
        for rep in range(repeats):
            G = ev_network(n, m, r)
            degrees = [d for _, d in G.degree()]
            data.append(max(degrees))
            print(n, 'rep:', rep)

        # saving
        with open(data_folder + filename + str('_%d' % n), "wb") as f:
            pickle.dump(data, f)
    k1_values.append(np.mean(data))
    print(n, "complete")


def fit(x, a, b):
    return a*x**b


popt, pcov = curve_fit(fit, n_values, k1_values, p0=(1, 1))
print("Fit: y = %.4f +/- %.4f N^%.4f +/- %.4f" %
      (popt[0], np.sqrt(pcov[0, 0]), popt[1], np.sqrt(pcov[1, 1])))

ax1.scatter(n_values, k1_values, label='Observed',
            c='C2', marker='x')  # type:ignore
ax1.set_xscale('log')
xmin1, xmax1 = ax1.get_xlim()
xvals1 = np.linspace(xmin1, xmax1, 1000)
ax1.plot(xvals1, [fit(x, *popt) for x in xvals1],
         label=r'$\alpha N^{\beta}$ Fit', c='C2', linestyle='dashed', alpha=0.5)
ax1.set_xlim(xmin1, xmax1)
ax1.set_xlabel(r"$N$")
ax1.set_ylabel("$k_1$")
ax1.set_title("(A)")

ax2.scatter(n_values, k1_values, label='Observed',
            c='C2', marker='x')  # type:ignore
ax2.set_xscale('log')
ax2.set_yscale('log')
xmin2, xmax2 = ax2.get_xlim()
xvals2 = np.linspace(xmin2, xmax2, 1000)
ax2.plot(xvals2, [fit(x, *popt) for x in xvals2],
         label=r'$\alpha N^{\beta}$ Fit', c='C2', linestyle='dashed', alpha=0.5)
ax2.set_xlim(xmin2, xmax2)
ax2.set_xlabel(r"$N$")
ax2.set_ylabel("$k_1$")
ax2.set_title("(B)")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)

plt.savefig(figures_folder + 'ev_k1.svg',
            format='svg', bbox_inches='tight')
