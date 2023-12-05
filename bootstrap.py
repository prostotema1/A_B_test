import numpy as np
from tqdm import tqdm
import scipy.stats as stas
from matplotlib import pyplot as plt
import math

def bootstrap(values_ctrl, values_var, iter=1000, alpha=0.05, mde = 0):

    size_ctrl = len(values_ctrl)
    size_var = len(values_var)
    size = size_var + size_ctrl

    diff = round(np.mean(values_var) - np.mean(values_ctrl), 2)
    diffs = []

    for _ in tqdm(range(iter)):
        sample_ctrl = np.random.choice(a=values_ctrl, size=size_ctrl, replace=True)
        sample_var = np.random.choice(a=values_var, size=size_var, replace=True)
        diffs.append(np.mean(sample_var) - np.mean(sample_ctrl))

    plot_diffs = np.round(diffs, 2)
    unique, counts = np.unique(plot_diffs, return_counts=True)

    plt.bar(unique, counts)
    plt.show()

    p_value = 1 - len(list(filter(lambda x: abs(x) > mde, diffs))) / iter
    print('p-value:', p_value)

    point_est = diff
    std_diff = np.std(diffs)
    z_crit = stas.norm.ppf(1 - alpha / 2)
    conf_int = (round(point_est - z_crit * std_diff / math.sqrt(size), 2), round(point_est + z_crit * std_diff / math.sqrt(size), 2))
    print('confidence interval with alpha=', alpha, ':', conf_int)

    #has_stat_diff = not (conf_int[0] <= 0 <= conf_int[1])
    has_stat_diff = p_value <= alpha

    if (has_stat_diff):
        print('There are statistically significant changes in the control group')
    else:
        print('There are NO statistically significant changes in the control group')

    return has_stat_diff, p_value, conf_int


ctrl = [2280, 1757, 2343, 1940, 1835, 3083, 2544, 1900, 2813, 2149, 2490, 2319, 2697, 1875, 2774, 2024, 2177, 1876, 2596, 2675, 1803, 2939, 2496, 1892, 1962, 2233, 2061, 2421, 2375, 2324]
var = [3008,2542,2365,2710,2297,2458,2838,2916,2652,2790,2420,2831,1972,2537,2516,3076,1968,1979,2626,2712,3112,2899,2407,2078,2928,2311,2915,2247,2805,1977]
print(bootstrap(ctrl, var))