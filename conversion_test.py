from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Генерация случайных данных для двух групп (конверсии)
np.random.seed(42)
group_a = np.random.binomial(n=1, p=0.1, size=1000)  # группа A, конверсия 10%
group_b = np.random.binomial(n=1, p=0.12, size=1000)  # группа B, конверсия 12%

# Проведение z-теста для различий в конверсии
count_a = np.sum(group_a)
count_b = np.sum(group_b)
nobs_a = len(group_a)
nobs_b = len(group_b)

stat, p_value = proportions_ztest([count_a, count_b], [nobs_a, nobs_b])

# Вывод результатов
print(f"Z-statistic: {stat}")
print(f"P-value: {p_value}")

# Определение статистической значимости при уровне значимости 0.05
alpha = 0.05
if p_value < alpha:
    print("The differences are statistically significant. We accept an alternative hypothesis.")
else:
    print("The differences are not statistically significant. We accept the null hypothesis.")