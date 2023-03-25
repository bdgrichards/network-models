from ra_network import ra_network
import matplotlib.pyplot as plt
from utils import data_folder, figures_folder
import pickle
import numpy as np
from scipy.optimize import curve_fit

# parameters
m = 10
n_values = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
repeats = 100
filename = 'ra_k1'

# create plots
fig = plt.figure(figsize=(6.4, 3), tight_layout=True)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# get data
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
            G = ra_network(n, m)
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


def expected(N, m):
    return m + (np.log(N) / (np.log(m+1) - np.log(m)))


def fit(x, a, b):
    return a*np.log(x) + b


# curve fit the data
popt, pcov = curve_fit(fit, n_values, k1_values, p0=(1, 1))
print("Fit: y = %.4f +/- %.4f ln(N) + %.4f +/- %.4f" %
      (popt[0], np.sqrt(pcov[0, 0]), popt[1], np.sqrt(pcov[1, 1])))

ax1.scatter(n_values, k1_values, label='Observed',
            c='C1', marker='x')  # type:ignore
ax1.set_xscale('log')
xmin1, xmax1 = ax1.get_xlim()
xvals1 = np.linspace(xmin1, xmax1, 1000)
ax1.plot(xvals1, [expected(x, m) for x in xvals1],
         label='Expected', c='k', linestyle='dashed')
ax1.plot(xvals1, [fit(x, *popt) for x in xvals1],
         label=r'$\alpha \ln N + \beta$ Fit', c='C1', linestyle='dashed', alpha=0.5)
ax1.set_xlim(xmin1, xmax1)
ax1.set_xlabel(r"$N$")
ax1.set_ylabel("$k_1$")
ax1.set_title("(A)")

ax2.scatter(n_values, k1_values, label='Observed',
            c='C1', marker='x')  # type:ignore
ax2.set_xscale('log')
ax2.set_yscale('log')
xmin2, xmax2 = ax2.get_xlim()
xvals2 = np.linspace(xmin2, xmax2, 1000)
ax2.plot(xvals2, [expected(x, m) for x in xvals2],
         label='Expected', c='k', linestyle='dashed')
ax2.plot(xvals2, [fit(x, *popt) for x in xvals2],
         label=r'$\alpha \ln N + \beta$ Fit', c='C1', linestyle='dashed', alpha=0.5)
ax2.set_xlim(xmin2, xmax2)
ax2.set_xlabel(r"$N$")
ax2.set_ylabel("$k_1$")
ax2.set_title("(B)")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0.)

plt.savefig(figures_folder + 'ra_k1.svg',
            format='svg', bbox_inches='tight')
