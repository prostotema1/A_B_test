from scipy.stats import beta
from calc_prob import calc_prob_between

imps_ctrl, convs_ctrl = 16500, 30
imps_test, convs_test = 17000, 50


a_C, b_C = convs_ctrl + 1, imps_ctrl - convs_ctrl + 1
beta_C = beta(a_C, b_C)

a_T, b_T = convs_test + 1, imps_test - convs_test + 1
beta_T = beta(a_T, b_T)

lift = (beta_T.mean() - beta_C.mean()) / beta_C.mean()
prob = calc_prob_between(beta_T, beta_C)

print(f"Test option lift Conversion Rates by {lift * 100:2.2f}% with {prob * 100:2.1f}% probability.")