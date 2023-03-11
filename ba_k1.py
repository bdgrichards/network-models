from ba_network import ba_network
import matplotlib.pyplot as plt
from utils import data_folder, figures_folder
import pickle
import numpy as np
from scipy import optimize

m = 10
n_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
repeats = 100
logbin_scale = 1.15
filename = 'ba_k1'

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
            G = ba_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.append(max(degrees))
            print(n, 'rep:', r)

        # saving
        with open(data_folder + filename + str('_%d' % n), "wb") as f:
            pickle.dump(data, f)
    k1_values.append(np.mean(data))
    print(n, "complete")


def expected_k1(n, m):
    return -(1/2) + np.sqrt((1/4) + n*m*(m+1))


def power_law(k, a):
    return a*np.array(k)**0.5


popt, pcov = optimize.curve_fit(power_law, n_values, k1_values)

ax1.plot(n_values, [expected_k1(n, m)
         for n in n_values], label='Expected', c='k')
ax1.scatter(n_values, k1_values, label='Observed')
ax1.plot(n_values, power_law(n_values, *popt),
         label=r'$%.2f * N^{0.5}$ Fit' % popt[0], linestyle='dashed')
ax1.set_xlabel(r"$N$")
ax1.set_ylabel("$k_1$")
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)

plt.savefig(figures_folder + 'ba_k1.svg',
            format='svg', bbox_inches='tight')
