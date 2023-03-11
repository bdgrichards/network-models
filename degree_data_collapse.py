from ba_network import ba_network
import matplotlib.pyplot as plt
from logbin import logbin
from utils import data_folder
import pickle

m = 10
n_values = [100000, 10000, 1000, 100]
repeats = 100
logbin_scale = 1.15
filename = 'ba_varying_n'
D = 0.5


fig = plt.figure(figsize=(6.5, 3), tight_layout=True)
ax1 = fig.add_subplot(111)

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
            data.extend(degrees)
            print(n, 'rep:', r)

        # saving
        with open(data_folder + filename + str('_%d' % n), "wb") as f:
            pickle.dump(data, f)

    x, y, _ = logbin(data, logbin_scale)
    scaled_y = [y[i]*x[i]*(x[i] + 1)*(x[i] + 2) for i in range(len(y))]
    scaled_x = [x[i]/(n**D) for i in range(len(y))]
    ax1.scatter(scaled_x, scaled_y, label=n, color="C%i" % n_values.index(n))
    ax1.plot(scaled_x, scaled_y, color="C%i" % n_values.index(n), alpha=0.5)
    print(n, "complete")

ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')

plt.show()
