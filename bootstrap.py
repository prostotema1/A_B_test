import numpy as np
from tqdm import tqdm
import scipy.stats as stas

def bootstrap(values_ctrl, values_var, iter=1000, alpha=0.05):

    size_ctrl = len(values_ctrl)
    size_var = len(values_var)

    ctrl_quantiles = []
    var_quantiles = []

    for _ in tqdm(range(iter)):
        sample_ctrl = np.random.choice(a=values_ctrl, size=size_ctrl, replace=True)
        sample_var = np.random.choice(a=values_var, size=size_var, replace=True)

        ctrl_quantiles.append(np.quantile(sample_ctrl, 1 - alpha))
        var_quantiles.append(np.quantile(sample_var, 1 - alpha))

    std_ctrl = np.std(ctrl_quantiles)
    std_var = np.std(var_quantiles)

    point_est = np.mean(ctrl_quantiles) - np.mean(var_quantiles)
    z_crit = stas.norm.ppf(1 - alpha / 2)
    std = (std_ctrl ** 2 / size_ctrl + std_var ** 2 / size_var) ** 0.5
    conf_int = (point_est - z_crit * std, point_est + z_crit * std)

    has_stat_diff = not (conf_int[0] <= 0 <= conf_int[1])

    if (has_stat_diff):
        print('There are statistically significant changes in the control group')
    else:
        print('There are NO statistically significant changes in the control group')

    return has_stat_diff
