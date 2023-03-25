from ra_network import ra_network
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from linbin import linbin
from utils import data_folder, figures_folder
import pickle

n = 10000
m_values = [128, 64, 32, 16, 8, 4, 2]
repeats = 100
filename = 'ra_varying_m'
num_bins = 30

fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0:1])
ax2 = fig.add_subplot(gs[1:])


def predicted_function(k, m):
    return (1/(m+1))*((m/(m+1))**(k-m))


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

    binwidth = (max(data) - min(data))/num_bins
    x, y, _ = linbin(data, binwidth)
    ax1.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                marker='x', linewidths=1)  # type:ignore
    ax1.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)

    ax2.scatter(x, y, color="C%i" % m_values.index(m), s=5,
                label=m, marker='x', linewidths=1)  # type:ignore
    ax2.plot(x, y, color="C%i" % m_values.index(m), alpha=0.2)
    ax2.plot(x, [predicted_function(k, m) for k in x], color="C%i" % m_values.index(m),
             linestyle='dotted')

    print(m, "complete")

ax1.set_ylabel(r"$p(k)$")
ax1.set_xlabel(r"$k$")
ax1.set_title('(A)')

ax2.set_yscale('log')
ax2.set_ylabel(r"$p(k)$")
ax2.set_xlabel(r"$k$")
ax2.set_title('(B)')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           borderaxespad=0., title=r'$m$', markerscale=2)

plt.savefig(figures_folder + 'ra_varying_m.svg',
            format='svg', bbox_inches='tight')
