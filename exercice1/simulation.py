"""Compare le predicteur de Bayes f* et un predicteur constant, pour la perte quadratique."""

import numpy as np

A = 0.3
B = 0.5
SIGMA = 0.5
X_LOW, X_HIGH = 15.0, 35.0
RNG = np.random.default_rng(0)


def sample(n):
    """Tire n couples (x, y) selon Y = A*x + B + bruit gaussien."""
    x = RNG.uniform(X_LOW, X_HIGH, size=n)
    eps = RNG.normal(0.0, SIGMA, size=n)
    y = A * x + B + eps
    return x, y


def f_star(x):
    """Predicteur de Bayes pour la perte quadratique."""
    return A * x + B


def main():
    """Calcule et affiche les risques theoriques et empiriques de f* et du predicteur constant."""
    n_test = 2_000_000

    var_x = (X_HIGH - X_LOW) ** 2 / 12.0
    bayes_risk = SIGMA ** 2
    mean_y = A * (X_LOW + X_HIGH) / 2.0 + B
    risk_constant = A ** 2 * var_x + SIGMA ** 2

    x_tr, y_tr = sample(100_000)
    c = y_tr.mean()

    def f_tilde(x):
        return np.full_like(x, c)

    x_te, y_te = sample(n_test)
    emp_risk_star = np.mean((y_te - f_star(x_te)) ** 2)
    emp_risk_tilde = np.mean((y_te - f_tilde(x_te)) ** 2)

    print(f"Risque de Bayes theorique R*            = {bayes_risk:.4f} kWh^2")
    print(f"Risque empirique test de f*             = {emp_risk_star:.4f} kWh^2")
    print(f"Risque theorique du predicteur constant = {risk_constant:.4f} kWh^2")
    print(f"Risque empirique test de f_tilde        = {emp_risk_tilde:.4f} kWh^2")
    print(f"R(f*) < R(f_tilde) ?                     {emp_risk_star < emp_risk_tilde}")
    print(f"|R_emp(f*) - R*|                         = {abs(emp_risk_star - bayes_risk):.4f}")


if __name__ == "__main__":
    main()
