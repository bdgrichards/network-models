from ev_network import ev_network
import matplotlib.pyplot as plt
from logbin import logbin
from utils import data_folder, figures_folder
import pickle
import scipy as sp
import numpy as np

n = 1000
# m_values = [81]/
m_values = [81, 27, 9, 3]
r_values = [int(m/3) for m in m_values]
repeats = 100
filename = 'ev_varying_m'
logbin_scale = 1.1

fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
ax1 = fig.add_subplot(111)


def predicted(k, m, r, a):
    return a * (sp.special.gamma(k + (m*r)/(m-r))) / (sp.special.gamma(k + 1 + (m*(r+1))/(m-r)))


def predicted_wrapper(m, r):
    return lambda k, a: predicted(k, m, r, a)


def predicted_full(k, m, r):
    alpha = 2/3
    norm = (r*(1 + alpha) * sp.special.gamma(r + 2 + (r+1)/alpha)) / \
        ((1 + alpha*r + r)*(r+1+alpha*(r+1)) * sp.special.gamma(r+1+r/alpha))
    return norm * (sp.special.gamma(k + (m*r)/(m-r))) / (sp.special.gamma(k + 1 + (m*(r+1))/(m-r)))


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

    x, y, _ = logbin(data, logbin_scale)
    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                marker='x', linewidths=1, label='m=%i, r=%i' % (m, r))  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)

    # wrapped_predicted = predicted_wrapper(m, r)
    # popt, pcov = sp.optimize.curve_fit(
    #     wrapped_predicted, x[:int(np.floor(len(x)/2))], y[:int(np.floor(len(y)/2))])
    # x_vals, y_vals = [], []
    # for i in range(len(x)):
    #     try:
    #         y_vals.append(predicted(x[i], m, r, popt[0]))
    #         x_vals.append(x[i])
    #         # print(i, y_vals)
    #     except:
    #         continue
    ax1.plot(x, [predicted_full(k, m, r) for k in x],
             label=r'Predicted', color="C%i" % m_values.index(m), linestyle='dashed', alpha=0.3)

    print(m, "complete")

ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., markerscale=2)

plt.savefig(figures_folder + 'ev_varying_m.svg',
            format='svg', bbox_inches='tight')
