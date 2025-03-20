import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def zc_sequence(L):
    tmp = torch.linspace(0, L - 1, steps=L)
    tmp = tmp * (tmp + 1)
    q = torch.linspace(1, L - 1, steps=L - 1).reshape(-1, 1)
    theta = torch.tensor(- np.pi) * q * tmp / L
    basis = torch.polar(torch.tensor(1.00, dtype=torch.float), theta).unsqueeze(1).expand(L - 1, L, L).reshape(-1, L)
    tmp0 = torch.linspace(0, L - 1, steps=L).reshape(-1, 1)
    tmp1 = torch.linspace(0, L - 1, steps=L).reshape(1, -1)
    shift = ((tmp0 + tmp1) % L)
    shift = shift.expand(L - 1, L, L).reshape(-1, L)
    zc = basis.gather(1, shift.long())
    # zc = torch.polar(torch.tensor(1.00, dtype=torch.float), theta)[0:2, :].unsqueeze(1).expand(2, L, L).reshape(-1, L)
    # shift = shift.expand(2, L, L).reshape(-1, L)
    # zc = zc.gather(1, shift.long())
    # zc = torch.polar(torch.tensor(1.00, dtype=torch.float), theta)[0:2, :]
    return zc


if __name__ == '__main__':
    L = 41
    N = L * (L - 1)
    group_size = L
    z = zc_sequence(L)
    mode = "ZC"
    # S_r = z.real
    # S_i = z.imag
    # print(z)
    corr = torch.mm(z, z.H)
    corr = torch.sqrt(corr.real ** 2 + corr.imag ** 2)
    norm = torch.sqrt(torch.diag(corr)).unsqueeze(1) * torch.sqrt(torch.diag(corr)).unsqueeze(0)
    corr = corr / norm
    # corr = torch.sqrt(corr.real ** 2 + corr.imag ** 2)
    # corr = torch.mm(S_r, S_r.t()) + torch.mm(S_i, S_i.t())
    # corr = corr / (torch.sqrt(torch.diag(corr)).unsqueeze(1) * torch.sqrt(torch.diag(corr)).unsqueeze(0))

    corr = corr.reshape(-1)
    # corr = corr - torch.diag(torch.diag(corr))
    index = list(set(range(N * N)) - set(range(0, N * N, N + 1)))
    tmp0 = torch.linspace(0, 1 - L, steps=L).reshape(-1, 1)
    tmp1 = torch.linspace(0, L - 1, steps=L).reshape(1, -1)
    shift = ((tmp0 + tmp1) % L)
    shift = shift.triu() - shift.triu().t()
    shift = shift.expand((L - 1), L, L).reshape(L * L * (L - 1))
    # inner group correlation
    index_intra = (shift +
                   (N + 1) * torch.linspace(0, N - 1, steps=N).unsqueeze(1).expand(N, L).reshape(L * N)).int().numpy()
    index_intra = list(set(index_intra) & set(index))
    corr_intra = corr[index_intra]
    # intra group correlation
    index_inter = list(set(index) - set(index_intra))
    corr_inter = corr[index_inter]
    plt.title(mode + "intra_N" + str(N) + "_L" + str(L))
    plt.hist(corr_intra.numpy(), bins=10)
    plt.show()
    plt.title(mode + "_inter_N" + str(N) + "_L" + str(L))
    plt.hist(corr_inter.numpy(), bins=10)
    plt.show()

    print(mode + " intra mean: ", np.abs(corr_intra).mean())
    print(mode + " intra max: ", np.abs(corr_intra).max())
    print(mode + " inter mean: ", np.abs(corr_inter).mean())
    print(mode + " inter max: ", np.abs(corr_inter).max())


import torch

def phi(x):
    return x ** -2.5


def phi_inv(y):
    return y ** (-2 / 5)


def phi_prime(x):
    return -2.5 * x ** -3.5


def phi_double_prime(x):
    return 8.75 * x ** -4.5


def phi_inv_prime(x):
    return -2 / 5 * x ** (-7 / 5)

def phi_inv_double_prime(x):
    return 14 / 25 * x ** (-12 / 5)

    
def H2(a, gu, gl, p, eps, g_ln):
    # 计算 phi 的反函数及其导数
    phi_inv_g_ln = phi_inv(g_ln)
    phi_prime_g_ln = phi_prime(g_ln)
    phi_double_prime_g_ln = phi_double_prime(g_ln)

    # 计算第一项
    term1 = (-2 * p * gu / (gu ** 2 - gl ** 2) * phi_prime_g_ln) * (1 - 2 * (a - g_ln) / eps) * (
            (a - g_ln + eps) / eps) ** 2

    # 计算第二项
    term2 = (-2 * p / (gu ** 2 - gl ** 2) * (phi_prime_g_ln ** 2 + phi_double_prime_g_ln * gu)) * (a - g_ln) * (
            (a - g_ln + eps) / eps) ** 2

    # 合并两项得到最终结果
    H2_gamma_n = term1 + term2

    return H2_gamma_n


def H2_prime(a, gu, gl, p, eps, g_ln):
    part1 = 12 * p * phi_inv_prime(gl) * 200 / eps**3 / (200*2 - 20*2) * (a - gl) * (a - gl -eps)
    part2 = - 12 * p *(phi_inv_prime(gl) ** 2 + phi_inv_double_prime(gl)) / eps**2 / (200*2 - 20*2) * (a - gl + eps) * (3 * a - 3 * gl + eps)

    return  part1 + part2

    
def calculate_expression(p, gu, gl, gamma_n):
    denominator = gu ** 2 - gl ** 2
    phi_inv_gamma_n = phi_inv(gamma_n)
    phi_inv_prime_gamma_n = phi_inv_prime(gamma_n)
    return -2 * p / denominator * phi_inv_gamma_n * phi_inv_prime_gamma_n
def p_upsilon(a, eps, gl, gu, p):
    # 创建布尔掩码
    mask1 = (a > 0) & (a < eps)
    mask3 = (a >= (gl - eps)) & (a < gl)
    mask4 = (a >= gl) & (a <= gu)
    
    # 初始化结果张量
    result = torch.zeros_like(a)
    
    # 处理各个区域
    # 区域1: 0 < a < eps
    term1 = (1 - p) * (1 + a/eps) * (a/eps - 1)**2
    result = torch.where(mask1, term1, result)
    
    # 区域3: gl-eps <= a < gl
    h2_vals = H2(a, gu, gl, p, eps, a)
    result = torch.where(mask3, h2_vals, result)
    
    # 区域4: gl <= a <= gu
    calc_vals = calculate_expression(p, gu, gl, a)
    result = torch.where(mask4, calc_vals, result)
    
    return result

def p_upsilon_prime(a, eps, gl, gu, p):
    # 创建布尔掩码
    mask1 = (a > 0) & (a < eps)
    mask3 = (a >= (gl - eps)) & (a < gl)
    mask4 = (a >= gl) & (a <= gu)
    
    result = torch.zeros_like(a)
    
    # 区域1: 0 < a < eps
    term1 = 6 * (1 - p) / eps**3 * a * (a - eps)
    result = torch.where(mask1, term1, result)
    
    # 区域3: gl-eps <= a < gl
    h2_prime_vals = H2_prime(a, gu, gl, p, eps, a)
    result = torch.where(mask3, h2_prime_vals, result)
    
    # 区域4: gl <= a <= gu
    denominator = gu**2 - gl**2
    calc_prime_vals = -2 * p / denominator * (phi_inv_prime(a)**2 + phi_inv_double_prime(a))
    result = torch.where(mask4, calc_prime_vals, result)
    
    return result

def p_upsilon_ratio(a):
    # 参数设置（转换为与a同设备和类型）
    eps_val = 0.0000001
    gl_val = 200**(-2.5)
    gu_val = 20**(-2.5)
    p_val = 0.05
    
    device = a.device
    dtype = a.dtype
    eps = torch.tensor(eps_val, dtype=dtype, device=device)
    gl = torch.tensor(gl_val, dtype=dtype, device=device)
    gu = torch.tensor(gu_val, dtype=dtype, device=device)
    p = torch.tensor(p_val, dtype=dtype, device=device)
    
    # 创建排除区间掩码 [eps, gl-eps)
    exclude_mask = (a >= eps) & (a < (gl - eps))
    
    # 计算分子和分母
    numerator = p_upsilon(a, eps, gl, gu, p)
    denominator = p_upsilon_prime(a, eps, gl, gu, p)
    
    # 安全除法（避免除以零）
    safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    ratio = numerator / safe_denominator
    ratio = torch.where(denominator == 0, torch.zeros_like(ratio), ratio)
    
    # 应用排除区间掩码
    final_result = torch.where(exclude_mask, torch.zeros_like(ratio), ratio)
    
    return final_result