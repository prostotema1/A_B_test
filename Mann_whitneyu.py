# Подключим библиотеки
import scipy.stats as sps
from tqdm.notebook import tqdm  # tqdm – библиотека для визуализации прогресса в цикле
from statsmodels.stats.proportion import proportion_confint

# Заводим счетчики количества отвергнутых гипотез для Манна-Уитни и для t-test
mann_bad_cnt = 0
ttest_bad_cnt = 0

# Прогоняем критерии 1000 раз
sz = 1000
for i in tqdm(range(sz)):
    # Генерируем распределение
    test = sps.uniform(loc=-1, scale=2).rvs(1000)  # U[-1, 1]
    control = sps.uniform(loc=-100, scale=200).rvs(1000)  # U[-100, 100]

    # Считаем pvalue
    mann_pvalue = sps.mannwhitneyu(control, test, alternative='two-sided').pvalue
    ttest_pvalue = sps.ttest_ind(control, test, alternative='two-sided').pvalue

    # отвергаем критерий на уровне 5%
    if mann_pvalue < 0.05:
        mann_bad_cnt += 1

    if ttest_pvalue < 0.05:
        ttest_bad_cnt += 1

# Строим доверительный интервал для уровня значимости критерия (или для FPR критерия)
left_mann_level, right_mann_level = proportion_confint(count=mann_bad_cnt, nobs=sz, alpha=0.05, method='wilson')
left_ttest_level, right_ttest_level = proportion_confint(count=ttest_bad_cnt, nobs=sz, alpha=0.05, method='wilson')
# Выводим результаты
print(
    f"Mann-whitneyu significance level: {round(mann_bad_cnt / sz, 4)}, [{round(left_mann_level, 4)}, {round(right_mann_level, 4)}]")
print(
    f"T-test significance level: {round(ttest_bad_cnt / sz, 4)}, [{round(left_ttest_level, 4)}, {round(right_ttest_level, 4)}]")