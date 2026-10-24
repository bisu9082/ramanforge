"""
Sensitivity of the headline result to the three acquisitions of the vial labeled HN2.

Section S11 establishes that this vial does not contain HN2: none of the four bands Kondo
et al. report for HN2 at 1064 nm is present in it, and it carries a 994/1585 cm-1 pair that
no nitrogen mustard has in either the free base or the hydrochloride. Its contents are not
identified here. The three acquisitions of it therefore cannot be counted as identifications
of an authentic agent in either arm, and the headline result is quoted on the remaining
twenty-seven queries.

This script re-runs the two analyses that the headline rests on with those three queries
removed and prints the full and reduced numbers side by side. The libraries are untouched,
so the reduced run differs from the full one only in the query set. The abstention analysis
of Section 3.6 already treats the HN2 acquisitions as a separate condition and never counts
them among its 27 in-library cases, so it is unaffected and is not re-run here. Note that removing them
*raises* the computed library's molecular accuracy, from 13/30 to 13/27, because all three
were computed-arm failures: the exclusion is unfavorable to the claim of this paper.

Every write is suppressed, so this cannot overwrite a deposited result. Run it after the
main pipeline; it reads nothing the main pipeline does not already write.
"""
import io, os, sys, json, contextlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))

EXCLUDE = "HN2"


def _noop(*a, **k):
    return None


@contextlib.contextmanager
def no_writes(mods):
    import pandas as pd
    import matplotlib.figure as mfig
    devnull = lambda *a, **k: open(os.devnull, "w")
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.object(pd.DataFrame, "to_csv", _noop))
        st.enter_context(mock.patch.object(mfig.Figure, "savefig", _noop))
        st.enter_context(mock.patch.object(json, "dump", _noop))
        for m in mods:
            st.enter_context(mock.patch.object(m, "open", devnull, create=True))
        yield


def run(excl):
    os.environ["RF_EXCL_QUERY"] = excl
    for m in [k for k in list(sys.modules) if k.startswith("v3_")]:
        del sys.modules[m]
    import v3_step3_headtohead as H
    import v3_t118_metrics as T
    buf = io.StringIO()
    with no_writes([H, T]), contextlib.redirect_stdout(buf):
        H.main(); T.main()
    return buf.getvalue()


def grab(txt, keys):
    out = {}
    for line in txt.splitlines():
        for k, tag in keys.items():
            if tag in line:
                out.setdefault(k, line.strip())
    return out


KEYS = {
    "queries":  "common queries",
    "meas":     "measured library (11 agents)",
    "closed":   "computed library (11, closed)",
    "open":     "computed library (54, open)",
    "metric_b": "best     (",
    "metric_0": "baseline (cosine",
}

if __name__ == "__main__":
    full = run("")
    red = run(EXCLUDE)
    print(f"{'='*78}\nHN2 label sensitivity: full query set vs the same analyses without the\n"
          f"three acquisitions of the vial labeled HN2\n{'='*78}")
    for tag, txt in (("FULL (30 queries)", full), (f"REDUCED (without {EXCLUDE})", red)):
        print(f"\n---- {tag}")
        g = grab(txt, KEYS)
        for k in KEYS:
            if k in g:
                print("   " + g[k])
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    log = os.path.join(ROOT, "data", "v3_hn2_sensitivity.log")
    with open(log, "w") as f:
        f.write("=== FULL QUERY SET ===\n" + full +
                f"\n\n=== WITHOUT {EXCLUDE} ===\n" + red)
    print(f"\nfull stdout of both runs: data/v3_hn2_sensitivity.log")
