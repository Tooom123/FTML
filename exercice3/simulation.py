"""Estime sigma^2 par OLS dans un modele lineaire a design fixe, et verifie que l'estimateur est sans biais."""

import numpy as np

RNG = np.random.default_rng(0)


def ols(X, y):
    """Resout les equations normales pour obtenir theta_hat."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def estimate_sigma2(n, d, sigma_true, n_repeat=2000):
    """Repete n_repeat fois le tirage du bruit et calcule sigma_hat^2 a chaque fois."""
    X = RNG.normal(size=(n, d))
    theta_star = RNG.normal(size=d)

    estimates = np.empty(n_repeat)
    for k in range(n_repeat):
        eps = RNG.normal(0.0, sigma_true, size=n)
        y = X @ theta_star + eps
        theta_hat = ols(X, y)
        residual = y - X @ theta_hat
        estimates[k] = residual @ residual / (n - d)
    return estimates


def main():
    """Compare la moyenne des estimations de sigma^2 a sa valeur theorique."""
    n, d = 200, 50
    sigma_true = 2.0
    sigma2_true = sigma_true ** 2

    est = estimate_sigma2(n, d, sigma_true)

    print(f"n = {n}, d = {d}, sigma^2 theorique = {sigma2_true:.4f}")
    print(f"Moyenne des estimations sigma_chapeau^2 = {est.mean():.4f}")
    print(f"Ecart-type des estimations              = {est.std():.4f}")
    print(f"Biais relatif                           = "
          f"{(est.mean() - sigma2_true) / sigma2_true:.2%}")
    biased = est * (n - d) / n
    print(f"\nEstimateur biaise ||residu||^2/n (moyenne) = {biased.mean():.4f}")
    print(f"Valeur attendue (n-d)/n * sigma^2          = "
          f"{(n - d) / n * sigma2_true:.4f}")


if __name__ == "__main__":
    main()
