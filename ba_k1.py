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
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

k1_values = []
k1_errs = []
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
    k1_errs.append(np.std(data)/np.sqrt(np.sum(data)))
    print(n, "complete")
print("Max err: %.3f min err: %.3f" % (max(k1_errs), min(k1_errs)))


def expected_k1(n, m):
    return -(1/2) + np.sqrt((1/4) + n*m*(m+1))


def power_law(k, a):
    return a*np.array(k)**0.5


popt, pcov = optimize.curve_fit(power_law, n_values, k1_values)
print("Fit: %.4f +/- %.4f" % (popt[0], pcov[0]))

ax1.set_xscale('log')
ax1.scatter(n_values, k1_values, label='Observed',
            marker='x')  # type:ignore
xmin1, xmax1 = ax1.get_xlim()
xvals1 = np.linspace(xmin1, xmax1, 1000)
ax1.plot(xvals1, [expected_k1(val, m)
         for val in xvals1], label='Expected', c='k', linestyle='dashed', zorder=-10)
ax1.plot(xvals1, power_law(xvals1, *popt),
         label=r'$\alpha N^{1/2}$ Fit', linestyle='dashed', alpha=0.3)
ax1.set_xlim(xmin1, xmax1)
ax1.set_xlabel(r"$N$")
ax1.set_ylabel("$k_1$")
ax1.set_title("(A)")

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.scatter(n_values, k1_values, label='Observed',
            marker='x')  # type:ignore
xmin2, xmax2 = ax2.get_xlim()
xvals2 = np.linspace(xmin2, xmax2, 1000)
ax2.plot(xvals2, [expected_k1(val, m)
         for val in xvals2], label='Expected', c='k', linestyle='dashed')
ax2.plot(xvals2, power_law(xvals2, *popt),
         label=r'$\alpha N^{1/2}$ Fit', linestyle='dashed', alpha=0.3)
ax2.set_xlim(xmin2, xmax2)
ax2.set_xlabel(r"$N$")
ax2.set_ylabel("$k_1$")
ax2.set_title("(B)")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)

plt.savefig(figures_folder + 'ba_k1.svg',
            format='svg', bbox_inches='tight')
