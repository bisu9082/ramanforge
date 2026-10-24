"""markdown 초고 -> achemso LaTeX 섹션 변환.

- [@a; @b]         -> \cite{a,b}
- **x** / *x*      -> \textbf / \textit
- ## / ###         -> \subsection / \subsubsection
- "Figure 4c"      -> Figure~\ref{fig:identification}c
- 유니코드(cm-1, 그리스문자, 대시 등) -> LaTeX
- 한글 메모, `---` 구분선, 최상위 제목은 제거
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
        if s.startswith("# "):                       # 최상위 제목 제거
            continue
        if re.match(r"^\*\(.*\)\*$", s.strip()):     # 드래프트 메모
            continue
        if s.strip() == "---":
            continue
        if s.strip().startswith(">"):                # 인용 블록(한글 메모)
            continue
        if re.search(r"[\uac00-\ud7a3]", s):         # 한글이 있으면 통째로 제거
            continue
        if s.startswith("### "):
            out.append(r"\subsubsection{%s}" % s[4:].strip()); continue
        if s.startswith("## "):
            out.append(r"\subsection{%s}" % s[3:].strip()); continue
        out.append(s)
    t = "\n".join(out)

    # 수식 블록 보호
    blocks = []
    def stash(m):
        blocks.append(m.group(0)); return f"@@MATH{len(blocks)-1}@@"
    t = re.sub(r"\$\$.*?\$\$", stash, t, flags=re.S)

    # 인용
    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        return r"\cite{%s}" % ",".join(keys)
    t = re.sub(r"\[((?:@[A-Za-z0-9_\-\.]+\s*;?\s*)+)\]", cite, t)

    # 아래첨자 표기
    t = re.sub(r"\*([A-Za-z])\*<sub>([A-Za-z]+)</sub>", r"$\1_{\\mathrm{\2}}$", t)
    t = re.sub(r"<sub>([^<]+)</sub>", r"$_{\1}$", t)
    t = re.sub(r"<sup>([^<]+)</sup>", r"$^{\1}$", t)

    # 인라인 코드
    t = re.sub(r"`([^`\n]+)`", r"\\texttt{\1}", t)

    # 강조
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t, flags=re.S)
    t = re.sub(r"(?<![\*\w])\*([^\*\n]+?)\*(?!\*)", r"\\textit{\1}", t)

    # 그림 참조
    def figref(m):
        n, suf = m.group(1), m.group(2) or ""
        return "Figure~\\ref{fig:%s}%s" % (FIGREF.get(n, "overview"), suf)
    t = re.sub(r"Figure\s+(\d)([a-z](?:\u2013[a-z])?(?:,[a-z])?)?", figref, t)

    # 단위 / 유니코드
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
