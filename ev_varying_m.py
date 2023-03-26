from ev_network import ev_network
import matplotlib.pyplot as plt
from logbin import logbin
from utils import data_folder, figures_folder, chi_sqr
import matplotlib.gridspec as gridspec
import pickle
import scipy as sp
import numpy as np

# parameters
n = 10000
m_values = [81, 27, 9, 3]
r_values = [int(m/3) for m in m_values]
repeats = 100
filename = 'ev_varying_m'
logbin_scale = 1.1
chi_squared_limit = 15

# create plots
fig = plt.figure(figsize=(6.4, 3), tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0:2])
ax2 = fig.add_subplot(gs[2:])


def predicted_full(k, r):
    """
    Return the full function, or above some threshold return the Puiseux approximation
    """
    norm = (1/(1+(5/3)*r)) * sp.special.gamma((5/2)
                                              * (r + 1)) / sp.special.gamma((5/2)*r)
    if k < 100:
        # full series
        return norm * (sp.special.gamma(k + (3/2)*r)) / (sp.special.gamma(k + (3/2)*r + (5/2)))
    else:
        # puiseux approximation
        return norm * ((k + (3/2)*r)**(-(5/2)) - (15/8)*(k + (3/2)*r)**(-(7/2)))


# lists to store values for chi squared testing
chi_sqr_x_vals = []
chi_sqr_y_vals = []
chi_sqr_y_errs = []

# get data
for i in range(len(m_values)):
    m = m_values[i]
    r = r_values[i]
    data = []
    # if saved data, use that, else generate new data
    try:
        # attempt to load the data file
        with open(data_folder + filename + str('_%d' % m), "rb") as f:
            data = pickle.load(f)
    except:
        # generate the data
        for rep in range(repeats):
            G = ev_network(n, m, r)
            degrees = [d for _, d in G.degree()]
            data.extend(degrees)
            print(m, "rep:", rep)

        # saving
        with open(data_folder + filename + str('_%d' % m), "wb") as f:
            pickle.dump(data, f)

    x, y, yerr = logbin(data, logbin_scale)

    # do chi squared test for 128 only
    if m == 3:
        # plot that data for the chi squared test
        trunc_x = [val for val in x if val < chi_squared_limit]
        trunc_y = [y[i] for i in range(len(y)) if x[i] < chi_squared_limit]
        trunc_yerr = [yerr[i]
                      for i in range(len(yerr)) if x[i] < chi_squared_limit]
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.scatter(trunc_x, trunc_y, color="C%i" % m_values.index(m), s=15,
                    marker='x', linewidths=1)  # type:ignore
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        ax2.set_xlim(xlims[0]-0.2, xlims[1]+7)
        ax2.set_ylim(ylims[0]-0.5e-3, ylims[1]+0.1)
        xlims = ax2.get_xlim()
        ylims = ax2.get_ylim()
        xvals = np.linspace(xlims[0], xlims[1], 100)
        ax2.plot(xvals, [predicted_full(k, r)
                 for k in xvals], color="k", alpha=0.2, zorder=-10)
        ax2.set_xlim(xlims[0], xlims[1])
        ax2.set_ylim(ylims[0], ylims[1])

        # save data for chi squared test
        chi_sqr_y_vals.extend(trunc_y)
        chi_sqr_x_vals.extend(trunc_x)
        chi_sqr_y_errs.extend(trunc_yerr)

    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                marker='x', linewidths=1, label=m)  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)
    ax1.plot(x, [predicted_full(k, r) for k in x],
             color="C%i" % m_values.index(m), linestyle='dashed', alpha=0.3)

    # invisible lines for the legend
    ax2.scatter([0, 0], [0, 0], color="C%i" % m_values.index(m), s=5,
                label=m, marker='x', linewidths=1)  # type:ignore

    print(m, "complete")

ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title("(A)")

ax2.set_ylabel(r"$p(k)$")
ax2.set_xlabel(r"$k$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("(B)")
# to change legend ordering
ax2.plot([0, 0], [0, 0], color="k", alpha=0.2,
         zorder=-10, label=r"$p_\infty(k)$")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., markerscale=2, title=r"$m$")

plt.savefig(figures_folder + 'ev_varying_m.svg',
            format='svg', bbox_inches='tight')

# chi squared test


# def predicted_func_wrapper(k):
#     return predicted_full(k, 1)


# # calculate chi squared
# print("Chi Squared: %.3f" % chi_sqr(f=predicted_func_wrapper,
#       x=chi_sqr_x_vals, y=chi_sqr_y_vals, yerr=chi_sqr_y_errs))
# print("DoF: %i" % len(chi_sqr_x_vals))
