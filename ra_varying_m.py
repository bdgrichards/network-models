from ra_network import ra_network
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from linbin import linbin
from utils import data_folder, figures_folder, chi_sqr
import pickle
import numpy as np

# parameters
n = 10000
m_values = [128, 64, 32, 16, 8, 4, 2]
repeats = 100
filename = 'ra_varying_m'
num_bins = 30
chi_squared_limit = 500

# create plots
fig = plt.figure(figsize=(6.4, 3), tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0:2])
ax2 = fig.add_subplot(gs[2:])


def predicted_function(k, m):
    return (1/(m+1))*((m/(m+1))**(k-m))


# lists to store values for chi squared testing
chi_sqr_x_vals = []
chi_sqr_y_vals = []
chi_sqr_y_errs = []

# get data
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
            G = ra_network(n, m)
            degrees = [d for _, d in G.degree()]
            data.extend(degrees)
            print(m, "rep:", r)

        # saving
        with open(data_folder + filename + str('_%d' % m), "wb") as f:
            pickle.dump(data, f)

    # due to drastically different ranges, set num bins not widths
    binwidth = (max(data) - min(data))/num_bins
    x, y, yerr = linbin(data, binwidth)

    # do chi squared test for 128 only
    if m == 128:
        # plot that data for the chi squared test
        trunc_x = [val for val in x if val < chi_squared_limit]
        trunc_y = [y[i] for i in range(len(y)) if x[i] < chi_squared_limit]
        trunc_yerr = [yerr[i]
                      for i in range(len(yerr)) if x[i] < chi_squared_limit]
        ax2.set_yscale('log')
        ax2.scatter(trunc_x, trunc_y, color="C%i" % m_values.index(m), s=15,
                    marker='x', linewidths=1)  # type:ignore
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        ax2.set_xlim(xlims[0]-30, xlims[1]+30)
        ax2.set_ylim(ylims[0]-1e-4, ylims[1]+2e-3)
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        xvals = np.linspace(xlims[0], xlims[1], 100)
        ax2.plot(xvals, [predicted_function(k, m)
                 for k in xvals], color="k", alpha=0.2, zorder=-10)

        # save data for chi squared test
        chi_sqr_y_vals.extend(trunc_y)
        chi_sqr_x_vals.extend(trunc_x)
        chi_sqr_y_errs.extend(trunc_yerr)

    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                label=m, marker='x', linewidths=1)  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)
    ax1.plot(x, [predicted_function(k, m) for k in x], color="C%i" % m_values.index(m),
             linestyle='dashed', alpha=0.3)

    # invisible lines for the legend
    ax2.scatter([0, 0], [0, 0], color="C%i" % m_values.index(m), s=5,
                label=m, marker='x', linewidths=1)  # type:ignore

    print(m, "complete")

ax1.set_yscale('log')
ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_title('(A)')

ax2.set_ylabel(r"$p(k)$")
ax2.set_xlabel(r"$k$")
ax2.set_title('(B)')
# to change legend ordering
ax2.plot([0, 0], [0, 0], color="k", alpha=0.2,
         zorder=-10, label=r"$p_\infty(k)$")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., title=r'$m$', markerscale=2)

plt.savefig(figures_folder + 'ra_varying_m.svg',
            format='svg', bbox_inches='tight')

# chi squared test


# def predicted_func_wrapper(k):
#     return predicted_function(k, 128)


# # calculate chi squared
# print("Chi Squared: %.3f" % chi_sqr(f=predicted_func_wrapper,
#       x=chi_sqr_x_vals, y=chi_sqr_y_vals, yerr=chi_sqr_y_errs))
# print("DoF: %i" % len(chi_sqr_x_vals))
