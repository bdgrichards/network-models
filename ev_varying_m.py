from ev_network import ev_network
import matplotlib.pyplot as plt
from logbin import logbin
from utils import data_folder, figures_folder
import matplotlib.gridspec as gridspec
import pickle
import scipy as sp

n = 10000
m_values = [81, 27, 9, 3]
r_values = [int(m/3) for m in m_values]
repeats = 100
filename = 'ev_varying_m'
logbin_scale = 1.1

fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0:1])
ax2 = fig.add_subplot(gs[1:])


def predicted_full(k, r):
    norm = (1/(1+(5/3)*r)) * sp.special.gamma((5/2)
                                              * (r + 1)) / sp.special.gamma((5/2)*r)
    if k < 100:
        # full series
        return norm * (sp.special.gamma(k + (3/2)*r)) / (sp.special.gamma(k + (3/2)*r + (5/2)))
    else:
        # puiseux approximation
        return norm * ((k + (3/2)*r)**(-(5/2)) - (15/8)*(k + (3/2)*r)**(-(7/2)))


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
                marker='x', linewidths=1)  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)

    ax2.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                marker='x', linewidths=1, label='%i' % m)  # type:ignore
    ax2.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)
    ax2.plot(x, [predicted_full(k, r) for k in x],
             color="C%i" % m_values.index(m), linestyle='dashed', alpha=0.3)

    print(m, "complete")

ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_xscale('log')
ax1.set_title("(A)")

ax2.set_ylabel(r"$p(k)$")
ax2.set_xlabel(r"$k$")
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("(B)")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., markerscale=2, title=r"$m$")

plt.savefig(figures_folder + 'ev_varying_m.svg',
            format='svg', bbox_inches='tight')
