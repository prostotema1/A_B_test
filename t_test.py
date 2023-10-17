import numpy as np
from scipy import stats

N = 10
# нормально распределенный набор mean = 2 and var = 1
x = np.random.randn(N) + 2
# нормально распределенный набор mean = 0 and var = 1
y = np.random.randn(N)
# Вычисляем стандартное отклонение
# через дисперсии
var_x = x.var(ddof=1)
var_y = y.var(ddof=1)

standard_deviation = np.sqrt((var_x + var_y) / 2)
print("Standard Deviation =", standard_deviation)
# вычисляем значение тесовой статистики
t_value = (x.mean() - y.mean()) / (standard_deviation * np.sqrt(2 / N))

# число степеней свободы
dof = 2 * N - 2
# p-value
p_value = 1 - stats.t.cdf(t_value, df=dof)
print("t = " + str(t_value))
print("p = " + str(2 * p_value))
# Сравниваем со значением встроенной функции
t_value_2, p_value_2 = stats.ttest_ind(x, y, alternative='two-sided')
print("t = " + str(t_value_2))
print("p = " + str(p_value_2))
