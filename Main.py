import scipy.stats as sps
import numpy as np
import statsmodels.api as sm


def getSampleWithUniformDistribuition(left_bound, length, size):
    return sps.uniform(loc=left_bound, scale=length).rvs(size)


def getSampleWithNormalDistribution(size, mean, var):
    return np.random.randn(size) * var + mean


def getStandartDeviationFromVariance(sample):
    return sample.var(ddof=1)


def Mann_whitneyu(sample1, sample2, alternative='two-sided', alpha=0.05):
    def mann_whitney_u_test(sample1, sample2):
        statistic, _ = sps.mannwhitneyu(sample1, sample2, alternative=alternative)
        return statistic

    def bootstrap_mann_whitney(sample1, sample2, n_bootstrap=1000, alpha=0.05):
        u_statistic_observed = mann_whitney_u_test(sample1, sample2)

        u_statistics = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            combined_data = np.concatenate([sample1, sample2])
            np.random.shuffle(combined_data)
            u_statistics[i] = mann_whitney_u_test(combined_data[:len(sample1)], combined_data[len(sample1):])

        lower_percentile = alpha / 2 * 100
        upper_percentile = 100 - alpha / 2 * 100
        lower_bound = np.percentile(u_statistics, lower_percentile)
        upper_bound = np.percentile(u_statistics, upper_percentile)

        return u_statistic_observed, (lower_bound, upper_bound)

    observed_statistic, confidence_interval = bootstrap_mann_whitney(sample1, sample2, alpha=alpha)
    print(f"Наблюдаемая U-статистика: {observed_statistic}")
    print(f"95% Доверительный интервал: {confidence_interval}")


def tTest(sample1, sample2, size, alternative='two-sided'):
    t_statistic, p_value = sps.ttest_ind(sample1, sample2, alternative=alternative)
    print(f"T-статистика: {t_statistic}")
    print(f"P-значение: {p_value}")


def z_test_proportions(successes1, trials1, successes2, trials2,alternative='two-sided',value=0):
    z_stat, p_value = sm.stats.proportions_ztest(
        [successes1, successes2],
        [trials1, trials2],
        alternative=alternative,
        value=value
    )
    print(f"Z-статистика: {z_stat}")
    print(f"P-значение: {p_value}")
