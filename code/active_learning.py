"""
active_learning.py: the main loop driving ORCA (the oracle) by pool-based Active Learning.

Flow (repeated every round):
  1) represent the pool structures with cheap features (featurize)
  2) train the GPR surrogate on the already computed labeled set
       input=features, output=diagnostic band positions (P=O/P-F/C=N/...)  [config.target_mode]
  3) acquisition: from the GPR predictive variance (uncertainty) plus batch diversity,
     select batch_size structures to be computed next with ORCA
  4) oracle: run ORCA opt+freq+Raman on the selected structures → parse the spectra → add to labeled
  5) record the learning curve (cumulative number of DFT calls vs surrogate RMSE) → save al_state

Key measurement: "how many DFT calls are needed to reach the target accuracy, relative to exhaustive computation" (quantification of the novelty).

oracle modes:
  --oracle orca : run the real ORCA (local to Ku)
  --oracle mock : for sandbox logic checks (synthetic function, no ORCA required)
"""
from __future__ import annotations
import argparse, os, json, csv, sys
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import featurize as F
import spectrum as S


# ------------------------------ load data ------------------------------
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def read_pool(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def read_labeled(path: str) -> dict:
    """labeled.csv: name,smiles,<target cols...>  → {name: {targets}}"""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["name"]] = r
    return out


# ---------------------------- surrogate ----------------------------
def build_gpr():
    kernel = ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=1.0)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=2, alpha=1e-6)


class Surrogate:
    """Per-target GPR (provides uncertainty). Features reduced by PCA."""
    def __init__(self, targets, n_pca=25, seed=42):
        self.targets = targets
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=seed)
        self.models = {t: build_gpr() for t in targets}
        self._fit = False

    def fit(self, X, Y):
        Xr = self.pca.fit_transform(self.scaler.fit_transform(X))
        for j, t in enumerate(self.targets):
            y = Y[:, j]
            m = ~np.isnan(y)
            if m.sum() >= 3:
                self.models[t].fit(Xr[m], y[m])
        self._fit = True

    def predict(self, X):
        Xr = self.pca.transform(self.scaler.transform(X))
        mean = np.zeros((X.shape[0], len(self.targets)))
        std = np.zeros_like(mean)
        for j, t in enumerate(self.targets):
            try:
                mu, sd = self.models[t].predict(Xr, return_std=True)
                mean[:, j], std[:, j] = mu, sd
            except Exception:
                std[:, j] = 1e3   # target never trained on -> high uncertainty
        return mean, std


# ---------------------------- acquisition ----------------------------
def select_batch(std, Xr, k, kappa, div_lambda, mode):
    """Greedy selection by uncertainty (mean std) + diversity (similarity penalty within the selected batch)."""
    n = std.shape[0]
    unc = std.mean(axis=1)                      # mean uncertainty per candidate
    if mode == "random":
        rng = np.random.default_rng(0)
        return list(rng.choice(n, size=min(k, n), replace=False))
    if mode == "uncertainty":
        return list(np.argsort(-unc)[:k])

    # ucb_diversity / diversity: greedy
    Xn = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-9)
    chosen, avail = [], list(range(n))
    score = kappa * unc if mode != "diversity" else np.zeros(n)
    cur = score.copy()
    while len(chosen) < min(k, n):
        i = avail[int(np.argmax(cur[avail]))]
        chosen.append(i)
        avail.remove(i)
        if not avail:
            break
        sim = Xn[avail] @ Xn[i]                  # cosine similarity
        cur[avail] = cur[avail] - div_lambda * np.maximum(sim, 0)
    return chosen


# ---------------------------- oracle ----------------------------
def oracle_orca(rows, cfg):
    """Compute the selected structures with ORCA → {name: peaks}. Failures are excluded."""
    import structures as ST
    import orca_io as IO
    import run_orca as R
    res = {}
    for r in rows:
        xyz = ST.gen_xyz(r["smiles"], seed=cfg["active_learning"]["seed"])
        if xyz is None:
            print(f"  [skip] geometry failed: {r['name']}")
            continue
        inp = IO.save_input(r["name"], xyz, cfg, cfg["paths"]["orca_inputs"])
        out = os.path.join(cfg["paths"]["orca_outputs"], f"{r['name']}.out")
        rr = R.run_one(inp, out, cfg["orca"]["exe"])
        if rr["ok"] and IO.parse_converged(out):
            res[r["name"]] = IO.parse_raman(out)
        else:
            print(f"  [fail] {r['name']}: {rr.get('reason')}")
    return res


def oracle_mock(rows, cfg):
    """Synthetic oracle for sandbox validation: band positions generated deterministically from SMILES features."""
    res = {}
    for r in rows:
        v = F.featurize_smiles(r["smiles"])
        if v is None:
            continue
        h = float(np.tanh(v[:9].sum() % 7))     # structure-dependent pseudo-random value
        peaks = [
            {"freq": 1245 + 15 * h, "activity": 30 + 10 * abs(h), "depol": 0.3},   # P=O
            {"freq": 725 + 20 * h,  "activity": 12, "depol": 0.4},                  # P-F
            {"freq": 1630 + 45 * np.sin(h), "activity": 20, "depol": 0.2},          # C=N
            {"freq": 1060 + 8 * h,  "activity": 8,  "depol": 0.5},                  # P-O-C_asym
        ]
        res[r["name"]] = peaks
    return res


# -------------------- extract and store targets --------------------
def peaks_to_targets(peaks, cfg):
    d = S.diagnostic_bands(peaks, cfg["active_learning"]["targets"], cfg)
    return d


def append_labeled(path, name, smiles, targets):
    exists = os.path.exists(path)
    cols = ["name", "smiles"] + list(targets.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        row = {"name": name, "smiles": smiles}
        row.update({k: targets[k] for k in targets})
        w.writerow(row)


# ------------------------------ main loop ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--oracle", choices=["orca", "mock"], default="orca")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    al = cfg["active_learning"]
    targets = al["targets"]
    rng = np.random.default_rng(al["seed"])
    n_rounds = args.rounds if args.rounds is not None else al["n_rounds"]

    pool = read_pool(cfg["paths"]["pool_csv"])
    pool = [r for r in pool if r.get("set", "pool") == "pool"]
    names = [r["name"] for r in pool]
    smiles = [r["smiles"] for r in pool]
    keep, X = F.featurize_frame(names, smiles)
    idx = {nm: i for i, nm in enumerate(keep)}
    pool = [r for r in pool if r["name"] in idx]
    Xr_full = X  # raw features (scaling/PCA happen inside the surrogate)
    print(f"[init] pool={len(pool)} featurized, targets={targets}, oracle={args.oracle}")

    labeled = read_labeled(cfg["paths"]["labeled_csv"])
    oracle = oracle_orca if args.oracle == "orca" else oracle_mock

    # initial seed: if nothing is labeled, select by diversity
    if len(labeled) < 3:
        seed_k = al["init_labeled"]
        # random + distance-maximising approximation instead of k-means (greedy farthest-point)
        sel = [int(rng.integers(len(pool)))]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        while len(sel) < min(seed_k, len(pool)):
            d = 1 - (Xn @ Xn[sel].T).max(axis=1)
            d[sel] = -1
            sel.append(int(np.argmax(d)))
        seed_rows = [pool[i] for i in sel]
        print(f"[seed] initial {len(seed_rows)} computations (diversity)")
        got = oracle(seed_rows, cfg)
        for nm, pk in got.items():
            r = next(p for p in pool if p["name"] == nm)
            append_labeled(cfg["paths"]["labeled_csv"], nm, r["smiles"], peaks_to_targets(pk, cfg))
        labeled = read_labeled(cfg["paths"]["labeled_csv"])

    history = []
    for rnd in range(1, n_rounds + 1):
        # assemble the training data
        lab_names = [nm for nm in labeled if nm in idx]
        if len(lab_names) < 3:
            print("[stop] labeled<3, halting"); break
        Xtr = X[[idx[nm] for nm in lab_names]]
        Ytr = np.array([[float(labeled[nm].get(t, "nan")) for t in targets] for nm in lab_names])

        sur = Surrogate(targets, n_pca=min(25, Xtr.shape[0] - 1, Xtr.shape[1]))
        sur.fit(Xtr, Ytr)

        # candidates = the pool entries not yet labeled
        cand = [p for p in pool if p["name"] not in labeled]
        if not cand:
            print("[done] pool exhausted"); break
        Xc = X[[idx[p["name"]] for p in cand]]
        _, std = sur.predict(Xc)

        # for the learning curve: LOO approximation of the current surrogate on the labeled set (train RMSE)
        mu_tr, _ = sur.predict(Xtr)
        rmse = float(np.sqrt(np.nanmean((mu_tr - Ytr) ** 2)))

        Xc_r = sur.pca.transform(sur.scaler.transform(Xc))
        pick = select_batch(std, Xc_r, al["batch_size"], al["ucb_kappa"],
                            al["diversity_lambda"], al["acquisition"])
        batch = [cand[i] for i in pick]
        print(f"[round {rnd}] labeled={len(lab_names)} | train_RMSE={rmse:.1f}cm-1 | "
              f"mean_unc={std.mean():.1f} | selected {len(batch)}: {[b['name'] for b in batch]}")

        got = oracle(batch, cfg)
        for nm, pk in got.items():
            r = next(p for p in pool if p["name"] == nm)
            append_labeled(cfg["paths"]["labeled_csv"], nm, r["smiles"], peaks_to_targets(pk, cfg))
        labeled = read_labeled(cfg["paths"]["labeled_csv"])

        history.append({"round": rnd, "n_labeled": len(lab_names),
                        "train_rmse": rmse, "mean_uncertainty": float(std.mean()),
                        "dft_calls_total": len(labeled)})
        _save_state(cfg, history)

    print(f"[end] labeled total {len(labeled)} | history → {cfg['paths']['al_state']}/history.json")


def _save_state(cfg, history):
    os.makedirs(cfg["paths"]["al_state"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["al_state"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
