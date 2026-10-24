"""
ramanforge Step 4-b
Pool-based Active Learning *replay* engine.

Premise: the whole pool (44 species) is computed once with DFT and those labels are
      cached, so several acquisition strategies can be compared in 'replay'.
      -> computational cost = pool size x 1 (independent of the number of strategies).

Baselines:
  ucb_diversity : AL (uncertainty + diversity)   <- the proposed method
  ucb_only      : uncertainty alone
  diversity_only: diversity alone
  random        : random selection
  full          : training on everything (the ceiling, the top of the learning curve)

Metrics (per round):
  - held-out spectrum prediction RMSE (regression: structural descriptors -> full spectrum)
  - identification accuracy (nearest-neighbor: predicted spectrum -> nearest label in the pool)
  vs the cumulative number of DFT calls (= the number of labels).

surrogate:
  multi-output GPR (RBF). uncertainty = mean predictive variance, diversity = minimum
  distance in feature space (to the existing labels). Regression target = full spectrum
  vector (801-dim, PCA reduced).
"""

from __future__ import annotations
import warnings
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

# suppress the harmless convergence warning raised when fitting a GPR to little data (console readability)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import spectra as S
from . import descriptors as D  # structure -> feature vector (with the fallback below)


# ------------------------------------------------------------------ config
def load_config(root: Path) -> dict:
    cfg_path = root / "scripts" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------- load data
@dataclass
class Pool:
    names: list[str]
    X_feat: np.ndarray      # (N, d)  structural features (input to the AL loop)
    Y_spec: np.ndarray      # (N, P)  the full spectrum (the label)
    is_anchor: np.ndarray   # (N,) bool
    grid: np.ndarray


def load_pool(root: Path, cfg: dict) -> Pool:
    """
    Build the pool from pool_bench.csv plus the completed .out files.
    Molecules whose calculation has not finished are excluded automatically (incremental runs are possible).
    """
    bench = pd.read_csv(root / "data" / "pool_bench.csv")
    outdir = root / "orca" / "outputs"

    names, feats, specs, anchors = [], [], [], []
    for _, r in bench.iterrows():
        out = outdir / str(r["out_file"])
        if not out.exists():
            continue
        try:
            spec = S.spectrum_vector(out, fwhm=cfg["spectrum"]["broadening_fwhm"])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {r['name']}: {e}")
            continue
        feat = D.smiles_to_features(r["smiles"])
        names.append(r["name"]); feats.append(feat); specs.append(spec)
        anchors.append(bool(r["is_anchor"]))

    grid = S._grid()
    return Pool(
        names=names,
        X_feat=np.vstack(feats),
        Y_spec=np.vstack(specs),
        is_anchor=np.array(anchors, dtype=bool),
        grid=grid,
    )


# ------------------------------------------------------------------ surrogate
def _make_gpr() -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-6, n_restarts_optimizer=1
    )


@dataclass
class Surrogate:
    """Structural features -> (PCA-reduced) spectrum. Independent GPR for each PC."""
    n_pc: int = 8
    scaler: StandardScaler = field(default_factory=StandardScaler)
    pca: PCA = None
    gprs: list = field(default_factory=list)

    def fit(self, Xf: np.ndarray, Ys: np.ndarray):
        Xs = self.scaler.fit_transform(Xf)
        npc = min(self.n_pc, Xs.shape[0], Ys.shape[1])
        self.pca = PCA(n_components=npc).fit(Ys)
        Z = self.pca.transform(Ys)
        self.gprs = []
        for k in range(npc):
            g = _make_gpr().fit(Xs, Z[:, k])
            self.gprs.append(g)
        return self

    def predict(self, Xf: np.ndarray):
        Xs = self.scaler.transform(Xf)
        means, stds = [], []
        for g in self.gprs:
            m, s = g.predict(Xs, return_std=True)
            means.append(m); stds.append(s)
        Zmean = np.vstack(means).T          # (n, npc)
        Zstd = np.vstack(stds).T
        Yhat = self.pca.inverse_transform(Zmean)
        # uncertainty: mean of the per-PC standard deviations (one scalar per sample)
        unc = Zstd.mean(axis=1)
        return Yhat, unc


# ------------------------------------------------------------------ acquisition
def acquire(strategy: str, cand_idx, surr, Xf, labeled_idx, cfg, rng):
    """Select the next batch indices."""
    bs = cfg["active_learning"]["batch_size"]
    kappa = cfg["active_learning"]["ucb_kappa"]
    lam = cfg["active_learning"]["diversity_lambda"]

    if strategy == "random":
        return list(rng.choice(cand_idx, size=min(bs, len(cand_idx)), replace=False))

    _, unc = surr.predict(Xf[cand_idx])
    # diversity: minimum feature-space distance to the labeled set
    Xl = Xf[labeled_idx]
    d = np.array([np.min(np.linalg.norm(Xl - Xf[i], axis=1)) for i in cand_idx])

    u = (unc - unc.min()) / (unc.max() - unc.min() + 1e-12)
    dv = (d - d.min()) / (d.max() - d.min() + 1e-12)

    if strategy == "ucb_only":
        score = u
    elif strategy == "diversity_only":
        score = dv
    elif strategy == "ucb_diversity":
        score = kappa * u + lam * dv
    else:
        raise ValueError(strategy)

    order = np.argsort(score)[::-1][:bs]
    return [cand_idx[i] for i in order]


# ------------------------------------------------------------------- evaluation
def spectral_rmse(surr, pool, test_idx):
    """How well the surrogate predicts the spectra of test molecules (structure->spectrum)."""
    Yhat, _ = surr.predict(pool.X_feat[test_idx])
    return float(np.sqrt(np.mean((Yhat - pool.Y_spec[test_idx]) ** 2)))


def add_noise(spec, snr_db, rng):
    """Measurement emulation: Gaussian noise at SNR (dB) relative to the spectrum signal + slight baseline drift."""
    sig_p = np.mean(spec ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    noisy = spec + rng.normal(0, np.sqrt(noise_p), size=spec.shape)
    # gentle baseline drift (a low-order polynomial)
    x = np.linspace(0, 1, len(spec))
    noisy = noisy + rng.normal(0, 0.02) * x + rng.normal(0, 0.01)
    return np.clip(noisy, 0, None)


def identify_accuracy(surr, pool, labeled_idx, query_idx, seed,
                      snr_db=20.0, reps=5):
    """
    Operational identification scenario:
      DB = actual DFT spectra for the labeled molecules, surrogate predictions for the uncomputed ones (hybrid).
      query = actual spectrum of the query molecule with noise injected (simulating an experiment).
      identification = nearest cosine neighbor in the DB; a correct call returns the molecule itself.
    AL sharpens the DB by choosing 'which molecules to actually compute', which raises the identification rate.
    """
    rng = np.random.default_rng(10_000 + seed)
    N = len(pool.names)
    labeled_set = set(int(i) for i in labeled_idx)
    DB = pool.Y_spec.copy()
    unl = [i for i in range(N) if i not in labeled_set]
    if unl:
        Yhat, _ = surr.predict(pool.X_feat[unl])
        for k, i in enumerate(unl):
            DB[i] = Yhat[k]
    DBn = np.linalg.norm(DB, axis=1) + 1e-12

    correct = total = 0
    for i in query_idx:
        for _ in range(reps):
            q = add_noise(pool.Y_spec[i], snr_db, rng)
            sims = DB @ q / (DBn * (np.linalg.norm(q) + 1e-12))
            if int(np.argmax(sims)) == int(i):
                correct += 1
            total += 1
    return correct / max(total, 1)


def evaluate(surr, pool, labeled_idx, test_idx, seed):
    """Return (spectrum RMSE, identification accuracy) together."""
    rmse = spectral_rmse(surr, pool, test_idx)
    acc = identify_accuracy(surr, pool, labeled_idx, test_idx, seed)
    return rmse, acc


# -------------------------------------------------------------- a single replay
def pick_test_set(pool: Pool, cfg: dict, rng) -> np.ndarray:
    """
    Draw a fixed held-out test set from the non-anchor molecules.
    Capped so as not to exceed half of the available non-anchors (to preserve the training set), minimum 2.
    """
    non_anchor = np.where(~pool.is_anchor)[0]
    target = cfg["active_learning"].get("n_test", 8)
    n_test = int(min(target, max(2, len(non_anchor) // 2)))
    n_test = min(n_test, len(non_anchor))
    return np.sort(rng.choice(non_anchor, n_test, replace=False))


def run_full_ceiling(pool: Pool, cfg: dict, test_idx: np.ndarray, seed: int = 0):
    """
    Ceiling: train on everything except the test set → evaluate on the same fixed test set.
    The performance attainable when all labels are used; the horizontal ceiling of the learning curve.
    """
    N = len(pool.names)
    test_set = set(test_idx.tolist())
    train = np.array([i for i in range(N) if i not in test_set])
    surr = Surrogate(n_pc=cfg.get("surrogate_npc", 8)).fit(
        pool.X_feat[train], pool.Y_spec[train])
    rmse, acc = evaluate(surr, pool, train, test_idx, seed)
    return pd.DataFrame([dict(round=0, n_labeled=len(train), rmse=rmse, acc=acc)])


def run_replay(pool: Pool, strategy: str, cfg: dict, seed: int,
               test_idx: np.ndarray):
    """
    Fixed test set protocol: test_idx is never labeled or acquired, and every round
    is evaluated on the 'same' test_idx → a fair comparison across strategies.
    """
    if strategy == "full":
        return run_full_ceiling(pool, cfg, test_idx, seed)

    rng = np.random.default_rng(seed + 1000)  # the acquisition random stream is kept separate from the test split
    N = len(pool.names)
    all_idx = np.arange(N)
    test_set = set(test_idx.tolist())

    # initial labels = the anchors (excluding the test set); with no anchors, init_labeled are drawn at random from the non-test entries
    init = [i for i in np.where(pool.is_anchor)[0] if i not in test_set]
    if not init:
        pool_ids = [i for i in all_idx if i not in test_set]
        k = min(cfg["active_learning"]["init_labeled"], len(pool_ids))
        init = list(rng.choice(pool_ids, k, replace=False))
    labeled = list(init)
    candidates = [i for i in all_idx if i not in test_set and i not in labeled]

    history = []
    rnd = 0
    while True:
        surr = Surrogate(n_pc=cfg.get("surrogate_npc", 8)).fit(
            pool.X_feat[labeled], pool.Y_spec[labeled])
        rmse, acc = evaluate(surr, pool, labeled, test_idx, seed)
        history.append(dict(round=rnd, n_labeled=len(labeled), rmse=rmse, acc=acc))
        if not candidates:
            break
        pick = acquire(strategy, list(candidates), surr, pool.X_feat, labeled, cfg, rng)
        for p in pick:
            candidates.remove(p)
        labeled += pick
        rnd += 1

    return pd.DataFrame(history)


# ------------------------------------------------------------ the full benchmark
def run_benchmark(root: Path):
    cfg = load_config(root)
    pool = load_pool(root, cfg)
    print(f"[pool] completed molecules {len(pool.names)}, anchors {pool.is_anchor.sum()}, "
          f"features {pool.X_feat.shape[1]}d, spectra {pool.Y_spec.shape[1]}d")
    if len(pool.names) < 10:
        print("  ! completed molecules <10 : learning-curve preliminary only (rerun as more computations accumulate).")

    strategies = cfg["active_learning"]["baselines"]
    n_seeds = cfg["active_learning"]["n_seeds"]
    rows = []
    for sd in range(n_seeds):
        # one fixed test set is drawn per seed and shared by every strategy, so the comparison is fair
        test_idx = pick_test_set(pool, cfg, np.random.default_rng(sd))
        for strat in strategies:
            h = run_replay(pool, strat, cfg, seed=sd, test_idx=test_idx)
            h["strategy"] = strat; h["seed"] = sd; h["n_test"] = len(test_idx)
            rows.append(h)
    out = pd.concat(rows, ignore_index=True)
    outpath = root / "data" / "al_benchmark.csv"
    out.to_csv(outpath, index=False)
    print(f"[saved] {outpath}  ({len(out)} rows)")
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    run_benchmark(root)
