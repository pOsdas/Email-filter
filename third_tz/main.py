import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import random
import math

random.seed(123)


def exp_sample(n, lam):
    """n независимых экспоненциальных случайных величин с параметром лямбда.
       X = -ln(U)/lam, U~Uniform(0,1)"""
    return [-math.log(1.0 - random.random()) / lam for _ in range(n)]


def sample_mean(xs):
    """Среднее арифметическое"""
    return sum(xs) / len(xs) if xs else float('nan')


def replicate_means(N, n, lam):
    """
    Для каждого сгенерировать выборку размера n, 
    и возвращаем список выборочных средних
    """
    means = []
    for _ in range(N):
        s = exp_sample(n, lam)
        means.append(sample_mean(s))
    return means


def normal_pdf(x, mu, sigma):
    """Плотность нормального распределения"""
    if sigma <= 0:
        return 0.0
    coef = 1.0 / (math.sqrt(2*math.pi) * sigma)
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coef * math.exp(exponent)


def linspace(a, b, m):
    if m <= 1:
        return [a]
    step = (b - a) / (m - 1)
    return [a + i * step for i in range(m)]


def mean_and_var(arr):
    """Среднее и несмещенная/смещенная дисперсия"""
    n = len(arr)
    if n == 0:
        return (float('nan'), float('nan'))
    mu = sum(arr) / n
    var = sum((x - mu)**2 for x in arr) / n
    return mu, var


# Параметры
lam = 2.0  
n = 30 
N = 10000

# Генерация выборочных средних
means = replicate_means(N, n, lam)

# Теоретические параметры распределения выборочного среднего
mu_theor = 1.0 / lam
sigma_theor = math.sqrt(1.0 / (n * lam * lam))

# Вычислим стандартизованные значения: lambda * корень n * (X с чертой - 1/лямбда)
z_list = [lam * math.sqrt(n) * (m - mu_theor) for m in means]

# Статистики для печати
mu_emp, var_emp = mean_and_var(means)
z_mu, z_var = mean_and_var(z_list)

print(f"Параметры: lambda={lam}, n={n}, N={N}")
print(f"Теоретическое среднее выборочного среднего = {mu_theor:.6f}, теоретическая дисперсия = {sigma_theor**2:.8f}")
print(f"Эмпирическое: mean(Xbar) = {mu_emp:.6f}, var(Xbar) = {var_emp:.8f}")
print(f"Для стандартизованных z: mean(z) = {z_mu:.6f}, var(z) = {z_var:.6f} (теоретически 0 и 1)")

# Гистограмма выборочных средних + теоретическая нормальная плотность N(1/lambda, 1/(n*lambda^2)) ---
fig1 = plt.figure(figsize=(8,5))
counts, bins, patches = plt.hist(means, bins=50, density=True, alpha=0.6)
# теоретическая кривая
xs = linspace(min(bins), max(bins), 300)
ys = [normal_pdf(x, mu_theor, sigma_theor) for x in xs]
plt.plot(xs, ys)
plt.title("Гистограмма выборочных средних и теоретическая N(1/лямбда, 1/(n*лямбда^2))")
plt.xlabel("Выборочное среднее X с чертой")
plt.ylabel("Плотность")
plt.grid(True)

# Гистограмма стандартизованных значений z + стандартная нормальная плотность N(0,1) ---
fig2 = plt.figure(figsize=(8,5))
plt.hist(z_list, bins=50, density=True, alpha=0.6)
xs2 = linspace(min(z_list), max(z_list), 300)
ys2 = [normal_pdf(x, 0.0, 1.0) for x in xs2]
plt.plot(xs2, ys2)
plt.title("Стандартизованные значения z = лямбда * корень n (X с чертой - 1/лямбда) и N(0,1)")
plt.xlabel("z")
plt.ylabel("Плотность")
plt.grid(True)


plt.show()

print("\nПервые 10 выборочных средних:", [round(x,6) for x in means[:10]])
print("Первые 10 z:", [round(x,6) for x in z_list[:10]])
