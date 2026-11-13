"""Table S-errinject from the outputs of v3_error_injection.py."""
import os, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = pd.read_csv(os.path.join(ROOT, "data", "v3_error_injection.csv"))
M = json.load(open(os.path.join(ROOT, "data", "v3_error_injection_meta.json")))
E, OP, OPP = M["dft"], M["operating_point"], M["operating_point_position_only"]
cols = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
real = D[D.sigma == 0].set_index("m").m_realised
rows = []
for s in np.round(np.arange(0, 1.01, 0.1), 2):
    cells = []
    for m in cols:
        v = D[np.isclose(D.sigma, s) & np.isclose(D.m, m)].mol_mean.iloc[0]
        cells.append(f"\\textbf{{{v:.1f}}}" if v >= 27 else f"{v:.1f}")
    rows.append(f"{s:.1f} & " + " & ".join(cells) + " \\\\")
hdr = " & ".join(f"{real[m]*100:.0f}\\,\\%" for m in cols)
tex = r"""\begin{table}[htbp]
\centering
\caption{Molecular top-1 of 30 for a reference library with known, graded error, built from
the measured line lists and rendered as computed entries are (Lorentzian, $\Gamma$ =
22\,\wn{}, native grid). Rows: band-position error, RMS in units of $\Gamma$. Columns:
band-set mismatch, the realised share of band intensity removed (spurious bands carrying
0.59 of that share are added). Intensities exact. Mean over %d random draws per cell; bold,
at least 27 of 30. Produced by \texttt{code/v3\_error\_injection.py}.}
\label{tab:si-errinject}
\small
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{6}{c}{band-set mismatch (intensity share removed)} \\
\cmidrule(lr){2-7}
$\sigma/\Gamma$ & %s \\
\midrule
%s
\bottomrule
\end{tabular}

\begin{flushleft}\footnotesize The computed library's own error, measured with the same
one-to-one pairing within 35\,\wn{} over the ten agents other than the vial labeled HN2, is
$\sigma$ = %.3f$\,\Gamma$ (%.1f\,\wn{} over %d pairs), a measured-only intensity share of
%.3f, a computed-only share of %.3f and an intensity error of $\tau$ = %.2f (s.d.\ of the
log height ratio of paired bands). At that operating point the model returns %.1f of 30
(5--95\,\%% of draws %.0f--%.0f; %.1f of 27 without the vial labeled HN2) against the 13 of
30 observed; with the positional error alone it returns %.1f (%.0f--%.0f).\end{flushleft}
\end{table}
""" % (M["R"], hdr, "\n".join(rows), E["sigma"], E["rmse"], E["n_pairs"], E["m_miss"],
       E["m_extra"], E["tau"], OP["mol_mean"], OP["mol_p05"], OP["mol_p95"], OP["mol27_mean"],
       OPP["mol_mean"], OPP["mol_p05"], OPP["mol_p95"])
open(os.path.join(ROOT, "latex", "sitables", "errinject.tex"), "w").write(tex)
print(tex)

# Table S-errinject-tau: intensity error added (from data/v3_error_injection_tau.csv)
T = pd.read_csv(os.path.join(ROOT, "data", "v3_error_injection_tau.csv"))
m_dft = sorted(T.m.unique())[-1]
def cell(s, t, m):
    v = T[np.isclose(T.sigma, s) & np.isclose(T.tau, t) & np.isclose(T.m, m)].mol_mean.iloc[0]
    return f"\\textbf{{{v:.1f}}}" if v >= 27 else f"{v:.1f}"
taus = sorted(T.tau.unique())
rows = [f"{s:.1f} & " + " & ".join(cell(s, t, m) for t in taus for m in (0.0, m_dft)) + " \\\\"
        for s in np.round(np.arange(0, 1.01, 0.2), 2)]
tex2 = r"""\begin{table}[htbp]
\centering
\caption{The same test with intensity error added: multiplicative log-normal noise of s.d.\
$\tau$ on every band height, with and without the band-set mismatch of the computed library
(28\,\%% of band intensity removed). Mean molecular top-1 of 30 over %d draws; bold, at least
27 of 30.}
\label{tab:si-errinject-tau}
\small
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{$\tau$ = %.2f} & \multicolumn{2}{c}{$\tau$ = %.2f (computed library)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
$\sigma/\Gamma$ & no mismatch & 28\,\%% mismatch & no mismatch & 28\,\%% mismatch \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (int(T.draws.iloc[0]), taus[0], taus[1], "\n".join(rows))
open(os.path.join(ROOT, "latex", "sitables", "errinject_tau.tex"), "w").write(tex2)
print(tex2)

# Table S-errinject-ext: systematic positional error and intensity error alone
# (from data/v3_error_injection_ext.csv, written by code/v3_error_injection_ext.py)
P = os.path.join(ROOT, "data", "v3_error_injection_ext.csv")
if os.path.exists(P):
    X = pd.read_csv(P)
    iid = D[np.isclose(D.m, 0)].set_index("sigma").mol_mean
    def v(var, s):
        y = X[(X.variant == var) & np.isclose(X.sigma, s)].mol_mean.iloc[0]
        return f"\\textbf{{{y:.1f}}}" if y >= 27 else f"{y:.1f}"
    def vi(s):
        y = iid[np.isclose(iid.index, s)].iloc[0]
        return f"\\textbf{{{y:.1f}}}" if y >= 27 else f"{y:.1f}"
    def cross(var):
        if var == "iid":
            s, y = list(iid.index), list(iid.values)
        else:
            e = X[X.variant == var]; s, y = [0.0] + list(e.sigma), [30.0] + list(e.mol_mean)
        for i in range(1, len(s)):
            if y[i - 1] >= 27 > y[i]:
                return s[i - 1] + (y[i - 1] - 27) / (y[i - 1] - y[i]) * (s[i] - s[i - 1])
    V = [("iid", "independent per band"), ("shift_rho0.5", "half shared shift"),
         ("shift_rho1.0", "shared shift"), ("scale_rho1.0", "scale error")]
    rows = []
    for s in np.round(np.arange(0.1, 1.001, 0.1), 2):
        rows.append(f"{s:.1f} & " + " & ".join(vi(s) if k == "iid" else v(k, s) for k, _ in V) + " \\\\")
    rows.append("\\midrule\n27 of 30 at $\\sigma/\\Gamma$ & " + " & ".join(f"{cross(k):.2f}" for k, _ in V) + " \\\\")
    T = X[X.variant == "tau_alone"]
    tau_txt = "; ".join(f"$\\tau$ = {t:.2f}: {y:.1f}" for t, y in zip(T.tau, T.mol_mean))
    tex3 = r"""\begin{table}[htbp]
\centering
\caption{Systematic positional error. The positional error of RMS $\sigma$ is injected as
in Table~\ref{tab:si-errinject} (independent per band), as a shift shared by every band of
an entry carrying half or all of the variance, or as a per-entry scale error (shift
proportional to wavenumber) with the same RMS over the entry's bands. Band set and
intensities exact; mean molecular top-1 of 30 over %d draws; bold, at least 27 of 30; last
row, the interpolated $\sigma/\Gamma$ at which 27 of 30 is crossed. Produced by
\texttt{code/v3\_error\_injection\_ext.py}.}
\label{tab:si-errinject-ext}
\small
\begin{tabular}{lcccc}
\toprule
$\sigma/\Gamma$ & %s \\
\midrule
%s
\bottomrule
\end{tabular}

\begin{flushleft}\footnotesize Intensity error alone (exact positions and band set): %s;
27 of 30 is crossed near $\tau$ = 0.44, between the slices of
Table~\ref{tab:si-errinject-tau}.\end{flushleft}
\end{table}
""" % (int(X.draws.iloc[0]), " & ".join(n for _, n in V), "\n".join(rows), tau_txt)
    open(os.path.join(ROOT, "latex", "sitables", "errinject_ext.tex"), "w").write(tex3)
    print(tex3)

# Table S-errinject-boot: agent-level bootstrap intervals on the 27-of-30 crossings
# (from data/v3_error_injection_boot_meta.json, written by code/v3_error_injection_boot.py)
P = os.path.join(ROOT, "data", "v3_error_injection_boot_meta.json")
if os.path.exists(P):
    BM = json.load(open(P)); G = M["gamma"]
    def row(key, label, unit, dft):
        c, (lo, hi) = BM[key]["crossing"], BM[key]["ci95"]
        if unit == "G":
            return (f"{label} & {c:.2f}$\\,\\Gamma$ ({c*G:.1f}\\,\\wn{{}}) & "
                    f"{lo:.2f}--{hi:.2f}$\\,\\Gamma$ ({lo*G:.1f}--{hi*G:.1f}\\,\\wn{{}}) & {dft} \\\\")
        if unit == "%":
            return f"{label} & {c*100:.0f}\\,\\% & {lo*100:.0f}--{hi*100:.0f}\\,\\% & {dft} \\\\"
        return f"{label} & {c:.2f} & {lo:.2f}--{hi:.2f} & {dft} \\\\"
    rows = [row("sigma", "band position, other errors absent", "G", f"{E['sigma']:.2f}$\\,\\Gamma$ (outside)"),
            row("m", "band-set mismatch, other errors absent", "%", f"{E['m_miss']*100:.0f}\\,\\% (inside)"),
            row("tau", "intensity error $\\tau$, other errors absent", "", f"{E['tau']:.2f} (inside)"),
            row("sigma_tau0.3", "band position at $\\tau$ = 0.30", "G", "---")]
    tex4 = r"""\begin{table}[htbp]
\centering
\caption{Uncertainty of the error-injection thresholds. For each error the point at which
the expected molecular top-1 falls through 90\,\%% of the queries (27 of 30) is located by
linear interpolation along a one-dimensional path, with the other errors absent (last row:
intensity error fixed at $\tau$ = 0.30). The interval resamples the %d queried vials (the vial labeled HN2 included) with
replacement (%d replicates), the vial being the independent unit; %d draws per condition.
Last column: the computed library's measured error and whether it lies inside the interval.
Produced by \texttt{code/v3\_error\_injection\_boot.py}.}
\label{tab:si-errinject-boot}
\small
\begin{tabular}{lccc}
\toprule
Error & Crossing & 95\%% interval over vials & Computed library \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (len(BM["agents"]), BM["B"], BM["R"], "\n".join(rows))
    open(os.path.join(ROOT, "latex", "sitables", "errinject_boot.tex"), "w").write(tex4)
    print(tex4)
