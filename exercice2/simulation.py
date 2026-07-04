"""Compare le predicteur de Bayes pour la perte quadratique (moyenne) et pour la perte absolue (mediane)."""

import numpy as np

RNG = np.random.default_rng(0)
LN2 = np.log(2.0)


def sample(n):
    """Tire n couples (x, y) avec Y = 2x + bruit exponentiel."""
    x = RNG.uniform(0.0, 1.0, size=n)
    z = RNG.exponential(scale=1.0, size=n)
    y = 2.0 * x + z
    return x, y


def f_quad(x):
    """Predicteur de Bayes pour la perte quadratique (moyenne conditionnelle)."""
    return 2.0 * x + 1.0


def f_abs(x):
    """Predicteur de Bayes pour la perte absolue (mediane conditionnelle)."""
    return 2.0 * x + LN2


def main():
    """Calcule et affiche les risques absolu et quadratique des deux predicteurs."""
    x, y = sample(5_000_000)

    R_abs_of_median = np.mean(np.abs(y - f_abs(x)))
    R_abs_of_mean = np.mean(np.abs(y - f_quad(x)))
    R_sq_of_mean = np.mean((y - f_quad(x)) ** 2)
    R_sq_of_median = np.mean((y - f_abs(x)) ** 2)

    theo_R_abs_median = LN2
    theo_R_abs_mean = 2.0 / np.e

    print(f"R_abs(f*_abs) simu = {R_abs_of_median:.4f}, theo = {theo_R_abs_median:.4f}")
    print(f"R_abs(f*_sq)  simu = {R_abs_of_mean:.4f}, theo = {theo_R_abs_mean:.4f}")
    print(f"R_sq(f*_abs)  simu = {R_sq_of_median:.4f}")
    print(f"R_sq(f*_sq)   simu = {R_sq_of_mean:.4f}")
    print(f"R_abs(f*_abs) < R_abs(f*_sq) ? {R_abs_of_median < R_abs_of_mean}")
    print(f"R_sq(f*_sq) < R_sq(f*_abs) ?   {R_sq_of_mean < R_sq_of_median}")


if __name__ == "__main__":
    main()
