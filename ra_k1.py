from ra_network import ra_network
import matplotlib.pyplot as plt
from utils import data_folder, figures_folder
import pickle
import numpy as np

m = 10
n_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
repeats = 100
logbin_scale = 1.15
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
    return np.log(((m+1)/N)*(m/(m+1)**m)) / np.log(m/(m+1))


# popt, pcov = curve_fit(power_law, n_values, k1_values)

ax1.scatter(n_values, k1_values, label='Measured')
ax1.scatter(n_values, [expected(n, m) for n in n_values],
            label='Expected')
ax1.legend()
ax1.set_xlabel("N")
ax1.set_ylabel("$k_1$")
ax1.set_xscale('log')
ax1.set_yscale('log')

plt.savefig(figures_folder + 'ra_k1.svg',
            format='svg', bbox_inches='tight')