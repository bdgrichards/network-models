from ra_network import ra_network
import matplotlib.pyplot as plt
from utils import data_folder, figures_folder
import pickle
import numpy as np
from scipy.optimize import curve_fit

m = 10
n_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
repeats = 100
filename = 'ra_k1'

fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
ax1 = fig.add_subplot(111)

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
        for r in range(repeats):
            G = ra_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.append(max(degrees))
            print(n, 'rep:', r)

        # saving
        with open(data_folder + filename + str('_%d' % n), "wb") as f:
            pickle.dump(data, f)
    k1_values.append(np.mean(data))
    print(n, "complete")


def expected(N, m):
    return m + (np.log(N) / (np.log(m+1) - np.log(m)))


def fit(x, a, b):
    return a*np.log(x) + b


popt, pcov = curve_fit(fit, n_values, k1_values, p0=(1, 1))
print("Fit: y = %.4f +/- %.4f ln(N) + %.4f +/- %.4f" %
      (popt[0], np.sqrt(pcov[0, 0]), popt[1], np.sqrt(pcov[1, 1])))
ax1.plot(n_values, [expected(n, m) for n in n_values],
         label='Expected', c='k', linestyle='dashed')
ax1.scatter(n_values, k1_values, label='Observed',
            c='C0', marker='x')  # type:ignore
ax1.plot(n_values, [fit(n, *popt) for n in n_values],
         label=r'$\alpha \ln N + \beta$ Fit', c='C0', linestyle='dashed', alpha=0.3)

ax1.set_xscale('log')
ax1.set_xlabel("N")
ax1.set_ylabel("$k_1$")
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)

plt.savefig(figures_folder + 'ra_k1.svg',
            format='svg', bbox_inches='tight')
