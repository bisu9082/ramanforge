"""
ramanforge Step 4-b
Pool-based Active Learning *replay* 엔진.

전제: 풀 전체(44종)를 DFT로 1회 계산해두고, 그 라벨을 캐시로 삼아
      여러 acquisition 전략을 '소급(replay)' 비교한다.
      -> 계산비용 = 풀크기 x 1 (전략 수와 무관).

비교군(baseline):
  ucb_diversity : AL (uncertainty + diversity)   <- 제안 방법
  ucb_only      : uncertainty 단독
  diversity_only: diversity 단독
  random        : 무작위 선택
  full          : 전수 학습 (상한선, 학습곡선의 천장)

지표(라운드별):
  - held-out 스펙트럼 예측 RMSE (회귀: 구조기술자 -> full 스펙트럼)
  - 식별 정확도 (nearest-neighbor: 예측 스펙트럼 -> 풀 내 최근접 라벨)
  vs 누적 DFT 호출수(=라벨 개수).

surrogate:
  다중출력 GPR (RBF). uncertainty = 예측 분산 평균, diversity = 특징공간
  최소거리(기존 라벨과의). 회귀 타깃 = full 스펙트럼 벡터(801-dim, PCA 축소).
"""

from __future__ import annotations
import warnings
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

# 소량 데이터 GPR 적합 시 뜨는 무해한 수렴 경고 억제(콘솔 가독성)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import spectra as S
from . import descriptors as D  # 구조 -> 특징벡터 (아래 fallback 포함)


# ------------------------------------------------------------------ config
def load_config(root: Path) -> dict:
    cfg_path = root / "scripts" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ 데이터 로드
@dataclass
class Pool:
    names: list[str]
    X_feat: np.ndarray      # (N, d)  구조 특징 (AL 입력)
    Y_spec: np.ndarray      # (N, P)  full 스펙트럼 (라벨)
    is_anchor: np.ndarray   # (N,) bool
    grid: np.ndarray


def load_pool(root: Path, cfg: dict) -> Pool:
    """
    pool_bench.csv + 완료된 .out 로 풀 구성.
    아직 계산 안 끝난 분자는 자동 제외(증분 실행 가능).
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
    """구조특징 -> (PCA 축소) 스펙트럼. 각 PC마다 독립 GPR."""
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
        # 불확실성: PC별 표준편차 평균 (스칼라 per sample)
        unc = Zstd.mean(axis=1)
        return Yhat, unc


# ------------------------------------------------------------------ acquisition
def acquire(strategy: str, cand_idx, surr, Xf, labeled_idx, cfg, rng):
    """다음 배치 인덱스 선택."""
    bs = cfg["active_learning"]["batch_size"]
    kappa = cfg["active_learning"]["ucb_kappa"]
    lam = cfg["active_learning"]["diversity_lambda"]

    if strategy == "random":
        return list(rng.choice(cand_idx, size=min(bs, len(cand_idx)), replace=False))

    _, unc = surr.predict(Xf[cand_idx])
    # diversity: 라벨셋과의 특징공간 최소거리
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


# ------------------------------------------------------------------ 평가
def spectral_rmse(surr, pool, test_idx):
    """surrogate가 test 분자의 스펙트럼을 얼마나 잘 예측하나 (구조->스펙트럼)."""
    Yhat, _ = surr.predict(pool.X_feat[test_idx])
    return float(np.sqrt(np.mean((Yhat - pool.Y_spec[test_idx]) ** 2)))


def add_noise(spec, snr_db, rng):
    """실측 모사: 스펙트럼 신호 대비 SNR(dB) 가우시안 노이즈 + 소량 baseline 드리프트."""
    sig_p = np.mean(spec ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    noisy = spec + rng.normal(0, np.sqrt(noise_p), size=spec.shape)
    # 완만한 baseline (저차 다항) 드리프트
    x = np.linspace(0, 1, len(spec))
    noisy = noisy + rng.normal(0, 0.02) * x + rng.normal(0, 0.01)
    return np.clip(noisy, 0, None)


def identify_accuracy(surr, pool, labeled_idx, query_idx, seed,
                      snr_db=20.0, reps=5):
    """
    실사용 식별 시나리오:
      DB = 라벨분자는 실측 DFT 스펙트럼, 미계산분은 surrogate 예측(하이브리드).
      쿼리 = query 분자의 실측 스펙트럼에 노이즈 주입(실험 모사).
      식별 = DB에서 코사인 최근접. 맞으면 자기 자신.
    AL은 '어느 분자를 실제 계산할지' 골라 DB를 예리하게 만들어 식별률을 높인다.
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
    """(스펙트럼 RMSE, 식별 정확도) 동시 반환."""
    rmse = spectral_rmse(surr, pool, test_idx)
    acc = identify_accuracy(surr, pool, labeled_idx, test_idx, seed)
    return rmse, acc


# ------------------------------------------------------------------ 한 번의 replay
def pick_test_set(pool: Pool, cfg: dict, rng) -> np.ndarray:
    """
    고정 held-out test set을 비앵커 분자에서 추출.
    가용 비앵커의 절반을 넘지 않도록 캡(학습셋 확보), 최소 2.
    """
    non_anchor = np.where(~pool.is_anchor)[0]
    target = cfg["active_learning"].get("n_test", 8)
    n_test = int(min(target, max(2, len(non_anchor) // 2)))
    n_test = min(n_test, len(non_anchor))
    return np.sort(rng.choice(non_anchor, n_test, replace=False))


def run_full_ceiling(pool: Pool, cfg: dict, test_idx: np.ndarray, seed: int = 0):
    """
    상한선(ceiling): test 제외 전량 학습 → 동일 고정 test set 평가.
    모든 라벨을 다 썼을 때 달성 가능한 성능. 학습곡선의 수평 천장.
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
    고정 test set 프로토콜: test_idx는 절대 라벨/획득하지 않고,
    매 라운드 '동일한' test_idx로만 평가 → 전략 간 공정 비교.
    """
    if strategy == "full":
        return run_full_ceiling(pool, cfg, test_idx, seed)

    rng = np.random.default_rng(seed + 1000)  # 획득 난수는 test 분할과 분리
    N = len(pool.names)
    all_idx = np.arange(N)
    test_set = set(test_idx.tolist())

    # 초기 라벨 = 앵커(테스트 제외). 앵커 없으면 비테스트서 init_labeled 무작위.
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


# ------------------------------------------------------------------ 전체 벤치마크
def run_benchmark(root: Path):
    cfg = load_config(root)
    pool = load_pool(root, cfg)
    print(f"[pool] 완료분자 {len(pool.names)}종, 앵커 {pool.is_anchor.sum()}종, "
          f"특징 {pool.X_feat.shape[1]}d, 스펙트럼 {pool.Y_spec.shape[1]}d")
    if len(pool.names) < 10:
        print("  ! 완료분자 <10 : 학습곡선 예비용(계산 더 쌓이면 재실행).")

    strategies = cfg["active_learning"]["baselines"]
    n_seeds = cfg["active_learning"]["n_seeds"]
    rows = []
    for sd in range(n_seeds):
        # seed마다 고정 test set을 한 번 뽑아 모든 전략이 공유(공정 비교)
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
