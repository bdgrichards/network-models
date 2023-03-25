# folder locations
figures_folder = './figures/'
data_folder = './data/'


def chi_sqr(f, x, y, yerr):
    """
    Chi squared test, with self explanatory variables
    Returns only the chi squared value, not p value
    """
    if len(x) != len(y) != len(yerr):
        raise Exception("Invalid arguments")
    total = 0
    for i in range(len(x)):
        total += ((y[i] - f(x[i]))/yerr[i])**2
    return total
