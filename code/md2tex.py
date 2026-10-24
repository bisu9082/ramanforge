"""markdown draft -> achemso LaTeX section conversion.

- [@a; @b]         -> \cite{a,b}
- **x** / *x*      -> \textbf / \textit
- ## / ###         -> \subsection / \subsubsection
- "Figure 4c"      -> Figure~\ref{fig:identification}c
- Unicode (cm-1, Greek letters, dashes and so on) -> LaTeX
- Korean-language notes, `---` separators and the top-level title are removed
"""
from __future__ import annotations
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGREF = {"1": "overview", "2": "calibration", "3": "spectra",
          "4": "identification", "5": "failure", "6": "validation"}

UNI = [
    ("\u2014", "---"), ("\u2013", "--"), ("\u2018", "`"), ("\u2019", "'"),
    ("\u201c", "``"), ("\u201d", "''"), ("\u00d7", r"$\times$"),
    ("\u2248", r"$\approx$"), ("\u2265", r"$\geq$"), ("\u2264", r"$\leq$"),
    ("\u00b1", r"$\pm$"), ("\u2192", r"$\rightarrow$"), ("\u0393", r"$\Gamma$"),
    ("\u03bd", r"$\nu$"), ("\u03b1", r"$\alpha$"), ("\u03bb", r"$\lambda$"),
    ("\u00b9", r"$^{1}$"), ("\u00b2", r"$^{2}$"), ("\u00b3", r"$^{3}$"), ("\u00b0", r"$^{\circ}$"),
    ("\u00c5", r"\AA{}"), ("\u2026", r"\ldots"), ("\u00a0", " "),
]


def convert(md: str) -> str:
    lines = md.split("\n")
    out, in_math = [], False
    for ln in lines:
        s = ln.rstrip()
        if s.startswith("# "):                       # drop the top-level heading
            continue
        if re.match(r"^\*\(.*\)\*$", s.strip()):     # drafting notes
            continue
        if s.strip() == "---":
            continue
        if s.strip().startswith(">"):                # block quotes (drafting notes)
            continue
        if re.search(r"[\uac00-\ud7a3]", s):         # drop the whole line if it contains Korean text
            continue
        if s.startswith("### "):
            out.append(r"\subsubsection{%s}" % s[4:].strip()); continue
        if s.startswith("## "):
            out.append(r"\subsection{%s}" % s[3:].strip()); continue
        out.append(s)
    t = "\n".join(out)

    # protect math blocks
    blocks = []
    def stash(m):
        blocks.append(m.group(0)); return f"@@MATH{len(blocks)-1}@@"
    t = re.sub(r"\$\$.*?\$\$", stash, t, flags=re.S)

    # quotation
    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        return r"\cite{%s}" % ",".join(keys)
    t = re.sub(r"\[((?:@[A-Za-z0-9_\-\.]+\s*;?\s*)+)\]", cite, t)

    # subscript notation
    t = re.sub(r"\*([A-Za-z])\*<sub>([A-Za-z]+)</sub>", r"$\1_{\\mathrm{\2}}$", t)
    t = re.sub(r"<sub>([^<]+)</sub>", r"$_{\1}$", t)
    t = re.sub(r"<sup>([^<]+)</sup>", r"$^{\1}$", t)

    # inline code
    t = re.sub(r"`([^`\n]+)`", r"\\texttt{\1}", t)

    # emphasis
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t, flags=re.S)
    t = re.sub(r"(?<![\*\w])\*([^\*\n]+?)\*(?!\*)", r"\\textit{\1}", t)

    # figure reference
    def figref(m):
        n, suf = m.group(1), m.group(2) or ""
        return "Figure~\\ref{fig:%s}%s" % (FIGREF.get(n, "overview"), suf)
    t = re.sub(r"Figure\s+(\d)([a-z](?:\u2013[a-z])?(?:,[a-z])?)?", figref, t)

    # units / unicode
    t = t.replace("cm\u207b\u00b9", r"\wn{}").replace("cm^-1", r"\wn{}")
    for a, b in UNI:
        t = t.replace(a, b)
    t = re.sub(r"(\d)\s*%", r"\1\\%", t)
    t = re.sub(r"(?<![\\$])%(?![\d\}])", r"\\%", t)

    for i, b in enumerate(blocks):
        body = b.strip("$").strip()
        t = t.replace(f"@@MATH{i}@@", "\\[\n%s\n\\]" % body)

    t = re.sub(r"\n{3,}", "\n\n", t).strip() + "\n"
    return t


def main():
    jobs = [("01_introduction.md", "introduction.tex"),
            ("02_experimental.md", "experimental.tex"),
            ("03_results.md", "results.tex"),
            ("04_conclusions.md", "conclusions.tex")]
    for src, dst in jobs:
        md = open(os.path.join(ROOT, "latex", src), encoding="utf-8").read()
        tex = convert(md)
        outp = os.path.join(ROOT, "latex", "build", "sections", "_new_" + dst)
        open(outp, "w", encoding="utf-8").write(tex)
        print(f"{src:24} -> {outp}  ({len(tex)} chars)")


if __name__ == "__main__":
    main()
