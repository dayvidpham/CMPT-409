import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings('ignore')


def make_soudry_dataset(n=200, d=5000, margin=0.1, sigma=3.0):
    v = np.ones(d) / np.sqrt(d)
    n2 = n // 2

    noise_pos = sigma * np.random.randn(n2, d)
    noise_neg = sigma * np.random.randn(n2, d)

    X_pos = margin * v + noise_pos
    X_neg = -margin * v + noise_neg

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n2), -np.ones(n2)])

    perm = np.random.permutation(n)
    return X[perm], y[perm], v


def get_empirical_max_margin(X, y):
    print("Computing empirical max-margin solution (SVM)...")
    clf = LinearSVC(C=1e6, fit_intercept=False, dual="auto", max_iter=20000, tol=1e-6)
    clf.fit(X, y)
    w_svm = clf.coef_.flatten()

    preds = clf.predict(X)
    acc = np.mean(preds == y)
    print(f"SVM Separation Accuracy: {acc * 100:.2f}%")
    if acc < 1.0:
        print("WARNING: Data is not linearly separable by SVM (Try increasing d or decreasing sigma)")

    return w_svm / np.linalg.norm(w_svm)


def get_angle(u, v):
    dot = np.dot(u, v)
    denom = np.linalg.norm(u) * np.linalg.norm(v) + 1e-12
    val = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(val)


def get_direction_distance(w, w_star):
    w_norm = w / (np.linalg.norm(w) + 1e-12)
    w_star_norm = w_star / (np.linalg.norm(w_star) + 1e-12)
    return np.linalg.norm(w_norm - w_star_norm)


def exponential_loss(w, X, y):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -100, 100)
    return np.mean(np.exp(-safe_margins))


def get_error_rate(w, X, y):
    preds = np.sign(X @ w)
    return np.mean(preds != y)


def step_gd(w, X, y, lr):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)

    grad = - (X.T @ (y * coeffs)) / len(y)
    return w - lr * grad


def step_ngd_stable(w, X, y, lr):
    margins = y * (X @ w)
    neg_margins = -margins
    shift = np.max(neg_margins)
    exps = np.exp(neg_margins - shift)
    softmax_weights = exps / np.sum(exps)

    direction = - (X.T @ (y * softmax_weights))
    return w - lr * direction


def step_sam_stable(w, X, y, lr, rho=0.05):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)
    grad = - (X.T @ (y * coeffs)) / len(y)
    gnorm = np.linalg.norm(grad) + 1e-12

    eps = rho * grad / gnorm
    w_adv = w + eps

    margins_adv = y * (X @ w_adv)
    safe_margins_adv = np.clip(margins_adv, -50, None)
    coeffs_adv = np.exp(-safe_margins_adv)
    grad_adv = - (X.T @ (y * coeffs_adv)) / len(y)

    return w - lr * grad_adv


def step_sam_ngd_stable(w, X, y, lr, rho=0.05):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)
    grad = - (X.T @ (y * coeffs)) / len(y)
    gnorm = np.linalg.norm(grad) + 1e-12

    eps = rho * grad / gnorm
    w_adv = w + eps

    return step_ngd_stable(w_adv, X, y, lr)


def main():
    N, D = 200, 5000
    X, y, v_pop = make_soudry_dataset(n=N, d=D, margin=0.1, sigma=3.0)

    w_star = get_empirical_max_margin(X, y)

    print(f"Angle(v, w*): {get_angle(v_pop, w_star):.4f} rad")

    TOTAL_ITERS = 100_000
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    optimizers = {
        "GD": step_gd,
        "SAM": step_sam_stable,
        "NGD": step_ngd_stable,
        "SAM_NGD": step_sam_ngd_stable
    }

    results = {}

    print("Starting Training...")

    for lr in learning_rates:
        results[lr] = {}
        print(f"\n--- LR = {lr} ---")

        for name, step_fn in optimizers.items():
            w = np.random.randn(D) * 1e-6

            history = {"steps": [], "dist": [], "angle": [], "loss": [], "err": []}
            record_steps = np.unique(np.logspace(0, np.log10(TOTAL_ITERS), 200).astype(int))
            rec_idx = 0

            for t in tqdm(range(1, TOTAL_ITERS + 1), leave=False, desc=name):
                try:
                    w = step_fn(w, X, y, lr)
                except:
                    break

                if rec_idx < len(record_steps) and t == record_steps[rec_idx]:
                    dist = get_direction_distance(w, w_star)
                    ang = get_angle(w, w_star)
                    ls = exponential_loss(w, X, y)
                    err = get_error_rate(w, X, y)

                    history["steps"].append(t)
                    history["dist"].append(dist)
                    history["angle"].append(ang)
                    history["loss"].append(ls)
                    history["err"].append(err)
                    rec_idx += 1

            results[lr][name] = history

    print("\nPlotting All Metrics...")

    metrics_to_plot = [
        ("dist", "Direction Distance", "_distance_convergence.png"),
        ("angle", "Angle with w* (radians)", "_angle_convergence.png"),
        ("loss", "Exponential Loss", "_loss_convergence.png"),
        ("err", "Classification Error", "_error_convergence.png")
    ]

    for key, title, filename in metrics_to_plot:
        fig, axes = plt.subplots(1, len(learning_rates), figsize=(18, 5))
        if len(learning_rates) == 1: axes = [axes]

        fig.suptitle(f"{title} vs Iterations (Log-Log)", fontsize=16)

        for i, lr in enumerate(learning_rates):
            ax = axes[i]
            for name in optimizers:
                hist = results[lr][name]
                if len(hist["steps"]) > 0:
                    ax.plot(hist["steps"], hist[key], label=name, linewidth=2)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"LR = {lr}")
            ax.set_xlabel("Iterations")
            ax.grid(True, which="both", alpha=0.3)
            if i == 0: ax.legend()

        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()