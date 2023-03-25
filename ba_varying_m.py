from ba_network import ba_network
import matplotlib.pyplot as plt
import numpy as np
from logbin import logbin
from utils import data_folder, figures_folder
import pickle

# setup parameters
n = 10000
m_values = [2, 4, 8, 16, 32]
repeats = 100
logbin_scale = 1.15
filename = 'ba_varying_m'
p_cutoff = 1e-4


def predicted_func(k):
    """
    The predicted degree distribution for a given value of k, 
    when divided by 2m(m+1)
    """
    return 1/(k*(k+1)*(k+2))


# lists to store values for chi squared testing
chi_sqr_x_vals = []
chi_sqr_y_vals = []
chi_sqr_y_errs = []

# setup subplots
fig = plt.figure(figsize=(6.4, 3), tight_layout=True)
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

    # logbin the data
    x, y, yerr = logbin(data, logbin_scale)

    # plot raw data on subplot A
    ax1.errorbar(x, y, yerr, color="C%i" % m_values.index(m),
                 linestyle='', elinewidth=1, alpha=0.8)
    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                marker='x', linewidths=1)  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(
        m), alpha=0.2)

    # collapse the data above the cutoff
    yerr_scaled = [yerr[i]/(2*m*(m+1))
                   for i in range(len(y)) if y[i] > p_cutoff]
    x_scaled = [x[i] for i in range(len(y)) if y[i] > p_cutoff]
    y_scaled = [i/(2*m*(m+1)) for i in y if i > p_cutoff]

    # saved scaled data for chi squared test
    chi_sqr_y_vals.extend(y_scaled)
    chi_sqr_x_vals.extend(x_scaled)
    chi_sqr_y_errs.extend(yerr_scaled)

    # plot the collapsed data on subplot B
    ax2.errorbar(x_scaled, y_scaled, yerr_scaled, color="C%i" %
                 m_values.index(m), linestyle='', elinewidth=1, alpha=0.7)
    ax2.scatter(x_scaled, y_scaled, label=r"$m = %i$" %
                m, color="C%i" % m_values.index(m), s=5, marker='x', linewidths=1)  # type:ignore
    print(m, "complete")

ax1.set_xscale('log')
ax1.set_yscale('log')
xmin1, xmax1 = ax1.get_xlim()
ax1.hlines([p_cutoff], xmin=xmin1, xmax=xmax1,
           color='k', linestyle='dashed', alpha=0.2)
ax1.set_xlim(xmin1, xmax1)
ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_title("(A)")

ax2.set_xscale('log')
ax2.set_yscale('log')
xmin2, xmax2 = ax2.get_xlim()
x_vals = np.linspace(xmin2, xmax2, 1000)
# plot hlines for legend only
ax2.hlines([p_cutoff], xmin=0, xmax=0,
           color='k', linestyle='dashed', alpha=0.2, label=r"$p(k) = 10^{-4}$")
ax2.plot(x_vals, predicted_func(x_vals),
         label=r'$\frac{1}{k(k+1)(k+2)}$', c='k', zorder=-1, alpha=0.2)
ax2.set_xlim(xmin2, xmax2)
ax2.set_ylabel(r"$p(k) \, / \, 2m(m+1)$")
ax2.set_xlabel(r"$k$")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., markerscale=2)
ax2.set_title("(B)")

plt.savefig(figures_folder + 'ba_varying_m.svg',
            format='svg', bbox_inches='tight')


# perform chi squared test


def chi_sqr(f, x, y, yerr):
    if len(x) != len(y) != len(yerr):
        raise Exception("Invalid arguments")
    total = 0
    for i in range(len(x)):
        total += ((y[i] - f(x[i]))/yerr[i])**2
    return total


# re-bin the data across multiple m values
binned_x = np.unique(chi_sqr_x_vals)
binned_y = []
binned_yerr = []
for x_i in binned_x:
    y_values_list = [chi_sqr_y_vals[ind] for ind in range(
        len(chi_sqr_y_vals)) if chi_sqr_x_vals[ind] == x_i]
    binned_y.append(np.mean(y_values_list))
    binned_yerr.append(np.std(y_values_list))

# calculate chi squared
print("Chi Squared: %.3f" % chi_sqr(f=predicted_func,
      x=binned_x, y=binned_y, yerr=binned_yerr))
print("DoF: %i" % len(binned_x))
