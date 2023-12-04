import scipy.stats as sps
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.proportion as prop


def getSampleWithUniformDistribuition(left_bound, length, size):
    return sps.uniform(loc=left_bound, scale=length).rvs(size)


def getSampleWithNormalDistribution(size, mean, var):
    return np.random.normal(mean, var, size=size)


def getStandartDeviationFromVariance(sample):
    return sample.var(ddof=1)


def Mann_whitneyu(sample1, sample2, alternative='two-sided', alpha=0.05):
    statistic, p_value = sps.mannwhitneyu(sample1, sample2, alternative=alternative)
    has_difference = p_value < alpha
    if has_difference:
        print("There is statistical differences between samples")
    else:
        print("There is no statistical differences between samples")

    return p_value


def calculate_confidence_interval(data, alpha=0.05):
    lower, upper = prop.proportion_confint(sum(data), len(data), alpha=alpha, method='wilson')
    return lower, upper


def tTest(sample1, sample2, alternative='two-sided', alpha=0.05):
    t_statistic, p_value = sps.ttest_ind(sample1, sample2, alternative=alternative)
    has_difference = p_value < alpha
    if has_difference:
        print("There is statistical differences between samples")
    else:
        print("There is no statistical differences between samples")

    return p_value


def compare_tTest_and_Mann_Whitney(counter=1000) -> bool:
    counter_t_test = 0
    counter_manna_whitney = 0
    k = counter
    while k >= 0:
        k -= 1
        test = getSampleWithUniformDistribuition(-1,2,1000)
        control = getSampleWithUniformDistribuition(-100,200,1000)
        t_Test_result = tTest(test, control)
        manna_whit_res = Mann_whitneyu(test,control)

        if t_Test_result < 0.05:
            counter_t_test += 1
        if manna_whit_res < 0.05:
            counter_manna_whitney += 1

    print(f"T-test significance level: {counter_t_test / counter}")
    print(f"Mann-whitney significance level: {counter_manna_whitney / counter}")
    return counter_t_test < counter_manna_whitney


def z_test_proportions(successes1, trials1, successes2, trials2, alternative='two-sided', value=0, alpha=0.05):
    z_stat, p_value = sm.stats.proportions_ztest(
        [successes1, successes2],
        [trials1, trials2],
        alternative=alternative,
        value=value
    )
    has_difference = p_value < alpha
    if has_difference:
        print("There is statistical differences between samples")
    else:
        print("There is no statistical differences between samples")

    return has_difference
