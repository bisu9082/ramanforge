"""
active_learning.py — pool-based Active Learning으로 ORCA(oracle)를 구동하는 메인 루프.

흐름 (라운드 반복):
  1) pool 구조를 값싼 피처로 표현 (featurize)
  2) 이미 계산된 labeled 집합으로 GPR surrogate 학습
       입력=피처, 출력=진단 밴드 위치(P=O/P-F/C=N/...)  [config.target_mode]
  3) acquisition: GPR 예측분산(uncertainty) + 배치 다양성(diversity)로
     다음에 ORCA로 계산할 batch_size개 구조 선택
  4) oracle: 선택 구조를 ORCA opt+freq+Raman 계산 → 스펙트럼 파싱 → labeled에 추가
  5) 학습곡선(누적 DFT 호출수 vs surrogate RMSE) 기록 → al_state 저장

핵심 측정: "전수계산 대비 몇 번의 DFT 호출로 목표 정확도 도달하는가" (노블티 정량화).

oracle 모드:
  --oracle orca : 실제 ORCA 실행 (Ku 로컬)
  --oracle mock : 샌드박스 로직 검증용 (합성 함수, ORCA 불필요)
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


# ---------------------------- 데이터 로드 ----------------------------
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
    """타깃별 GPR (uncertainty 제공). 피처는 PCA로 축소."""
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
                std[:, j] = 1e3   # 미학습 타깃 → 높은 불확실성
        return mean, std


# ---------------------------- acquisition ----------------------------
def select_batch(std, Xr, k, kappa, div_lambda, mode):
    """불확실성(평균 std) + 다양성(선택배치 내 유사도 페널티) 그리디 선택."""
    n = std.shape[0]
    unc = std.mean(axis=1)                      # 후보별 평균 불확실성
    if mode == "random":
        rng = np.random.default_rng(0)
        return list(rng.choice(n, size=min(k, n), replace=False))
    if mode == "uncertainty":
        return list(np.argsort(-unc)[:k])

    # ucb_diversity / diversity: 그리디
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
        sim = Xn[avail] @ Xn[i]                  # cosine 유사도
        cur[avail] = cur[avail] - div_lambda * np.maximum(sim, 0)
    return chosen


# ---------------------------- oracle ----------------------------
def oracle_orca(rows, cfg):
    """선택 구조를 ORCA로 계산 → {name: peaks}. 실패는 제외."""
    import structures as ST
    import orca_io as IO
    import run_orca as R
    res = {}
    for r in rows:
        xyz = ST.gen_xyz(r["smiles"], seed=cfg["active_learning"]["seed"])
        if xyz is None:
            print(f"  [skip] 지오메트리 실패: {r['name']}")
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
    """샌드박스 검증용 합성 oracle: SMILES 피처로 밴드 위치를 결정적으로 생성."""
    res = {}
    for r in rows:
        v = F.featurize_smiles(r["smiles"])
        if v is None:
            continue
        h = float(np.tanh(v[:9].sum() % 7))     # 구조 의존 유사난수
        peaks = [
            {"freq": 1245 + 15 * h, "activity": 30 + 10 * abs(h), "depol": 0.3},   # P=O
            {"freq": 725 + 20 * h,  "activity": 12, "depol": 0.4},                  # P-F
            {"freq": 1630 + 45 * np.sin(h), "activity": 20, "depol": 0.2},          # C=N
            {"freq": 1060 + 8 * h,  "activity": 8,  "depol": 0.5},                  # P-O-C_asym
        ]
        res[r["name"]] = peaks
    return res


# ---------------------------- 타깃 추출/저장 ----------------------------
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


# ---------------------------- 메인 루프 ----------------------------
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
    Xr_full = X  # 원피처 (surrogate 내부에서 스케일/PCA)
    print(f"[init] pool={len(pool)} featurized, targets={targets}, oracle={args.oracle}")

    labeled = read_labeled(cfg["paths"]["labeled_csv"])
    oracle = oracle_orca if args.oracle == "orca" else oracle_mock

    # 초기 seed: labeled 없으면 다양성으로 선택
    if len(labeled) < 3:
        seed_k = al["init_labeled"]
        # k-means 대신 무작위+거리 최대화 근사 (그리디 farthest-point)
        sel = [int(rng.integers(len(pool)))]
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        while len(sel) < min(seed_k, len(pool)):
            d = 1 - (Xn @ Xn[sel].T).max(axis=1)
            d[sel] = -1
            sel.append(int(np.argmax(d)))
        seed_rows = [pool[i] for i in sel]
        print(f"[seed] 초기 {len(seed_rows)}개 계산 (diversity)")
        got = oracle(seed_rows, cfg)
        for nm, pk in got.items():
            r = next(p for p in pool if p["name"] == nm)
            append_labeled(cfg["paths"]["labeled_csv"], nm, r["smiles"], peaks_to_targets(pk, cfg))
        labeled = read_labeled(cfg["paths"]["labeled_csv"])

    history = []
    for rnd in range(1, n_rounds + 1):
        # 학습 데이터 구성
        lab_names = [nm for nm in labeled if nm in idx]
        if len(lab_names) < 3:
            print("[stop] labeled<3, 중단"); break
        Xtr = X[[idx[nm] for nm in lab_names]]
        Ytr = np.array([[float(labeled[nm].get(t, "nan")) for t in targets] for nm in lab_names])

        sur = Surrogate(targets, n_pca=min(25, Xtr.shape[0] - 1, Xtr.shape[1]))
        sur.fit(Xtr, Ytr)

        # 후보 = 아직 labeled 아닌 pool
        cand = [p for p in pool if p["name"] not in labeled]
        if not cand:
            print("[done] 풀 소진"); break
        Xc = X[[idx[p["name"]] for p in cand]]
        _, std = sur.predict(Xc)

        # 학습곡선용: 현재 surrogate를 labeled에 대해 LOO 근사(train RMSE)
        mu_tr, _ = sur.predict(Xtr)
        rmse = float(np.sqrt(np.nanmean((mu_tr - Ytr) ** 2)))

        Xc_r = sur.pca.transform(sur.scaler.transform(Xc))
        pick = select_batch(std, Xc_r, al["batch_size"], al["ucb_kappa"],
                            al["diversity_lambda"], al["acquisition"])
        batch = [cand[i] for i in pick]
        print(f"[round {rnd}] labeled={len(lab_names)} | train_RMSE={rmse:.1f}cm-1 | "
              f"mean_unc={std.mean():.1f} | 선택 {len(batch)}: {[b['name'] for b in batch]}")

        got = oracle(batch, cfg)
        for nm, pk in got.items():
            r = next(p for p in pool if p["name"] == nm)
            append_labeled(cfg["paths"]["labeled_csv"], nm, r["smiles"], peaks_to_targets(pk, cfg))
        labeled = read_labeled(cfg["paths"]["labeled_csv"])

        history.append({"round": rnd, "n_labeled": len(lab_names),
                        "train_rmse": rmse, "mean_uncertainty": float(std.mean()),
                        "dft_calls_total": len(labeled)})
        _save_state(cfg, history)

    print(f"[end] labeled 총 {len(labeled)} | history → {cfg['paths']['al_state']}/history.json")


def _save_state(cfg, history):
    os.makedirs(cfg["paths"]["al_state"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["al_state"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
