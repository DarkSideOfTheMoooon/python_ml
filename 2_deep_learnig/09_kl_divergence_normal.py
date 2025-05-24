import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    """
    计算两个正态分布 N(mu_p, sigma_p^2) 和 N(mu_q, sigma_q^2) 的KL散度
    """
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    kl = term1 + term2 - 0.5
    return float(kl)
