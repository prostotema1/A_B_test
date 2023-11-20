import scipy.stats as sps
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.proportion as prop


def getSampleWithUniformDistribuition(left_bound, length, size):
    return sps.uniform(loc=left_bound, scale=length).rvs(size)


def getSampleWithNormalDistribution(size, mean, var):
    return np.random.randn(size) * var + mean


def getStandartDeviationFromVariance(sample):
    return sample.var(ddof=1)


def Mann_whitneyu(sample1, sample2, alternative='two-sided'):
    statistic, _ = sps.mannwhitneyu(sample1, sample2, alternative=alternative)
    return statistic, _


def calculate_confidence_interval(data, alpha=0.05):
    lower, upper = prop.proportion_confint(sum(data), len(data), alpha=alpha, method='wilson')
    return lower, upper


def tTest(sample1, sample2, alternative='two-sided'):
    t_statistic, p_value = sps.ttest_ind(sample1, sample2, alternative=alternative)
    return t_statistic, p_value


def z_test_proportions(successes1, trials1, successes2, trials2, alternative='two-sided', value=0):
    z_stat, p_value = sm.stats.proportions_ztest(
        [successes1, successes2],
        [trials1, trials2],
        alternative=alternative,
        value=value
    )
    return z_stat, p_value
