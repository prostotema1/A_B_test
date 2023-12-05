import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def bootstrap(values_ctrl, values_var, iter=1000, alpha=0.05, mde = 0):

    size_ctrl = len(values_ctrl)
    size_var = len(values_var)

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

    conf_int = (np.quantile(diffs, alpha / 2), np.quantile(diffs, 1 - alpha / 2))
    print('confidence interval with alpha=', alpha, ':', conf_int)

    has_stat_diff = not (conf_int[0] <= 0 <= conf_int[1])
    #has_stat_diff = p_value <= alpha

    if (has_stat_diff):
        print('There are statistically significant changes in the control group')
    else:
        print('There are NO statistically significant changes in the control group')

    return has_stat_diff, p_value, conf_int