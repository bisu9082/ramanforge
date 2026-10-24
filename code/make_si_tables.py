import pickle, numpy as np, pandas as pd, glob, os, re, json
from scipy.signal import find_peaks
from scipy.stats import pearsonr

OUT = open('si_tables.tex','w')
W = OUT.write
def tab(label, caption, header, rows, colspec, note=None, small=True):
    W("\\begin{table}[htbp]\n\\centering\n")
    W("\\caption{%s}\n\\label{%s}\n" % (caption, label))
    if small: W("\\small\n")
    W("\\begin{tabular}{%s}\n\\toprule\n" % colspec)
    W(" & ".join(header) + " \\\\\n\\midrule\n")
    for r in rows:
        W(" & ".join(str(x) for x in r) + " \\\\\n")
    W("\\bottomrule\n\\end{tabular}\n")
    if note: W("\n\\begin{flushleft}\\footnotesize %s\\end{flushleft}\n" % note)
    W("\\end{table}\n\n")

d2 = pickle.load(open('data/_fig2_data.pkl','rb'))
d3 = pickle.load(open('data/_fig3_data.pkl','rb'))
d4 = pickle.load(open('data/_fig4_data.pkl','rb'))
d5 = pickle.load(open('data/_fig5_data.pkl','rb'))
d6 = pickle.load(open('data/_fig6_data.pkl','rb'))
GRID = d3['grid']

# ---------- S1 library composition ----------
lib = sorted(os.path.basename(f)[:-4] for f in glob.glob('orca/outputs/*.out')
             if os.path.basename(f)[:-4] != 'GB_sarin_fast')
PRETTY = {'EXP_A230':'A-230','EXP_A232':'A-232','EXP_A234':'A-234','EXP_A242':'A-242',
          'EXP_GD_soman':'GD (soman)','EXP_HD':'HD (sulfur mustard)','EXP_HN1':'HN1','EXP_HN2':'HN2',
          'EXP_HN3':'HN3','GF_cyclosarin':'GF (cyclosarin)','GB_sarin':'GB (sarin)',
          'EtGB':'ethyl analogue of GB','nPrGB':'$n$-propyl analogue of GB','iBuGB':'$i$-butyl analogue of GB'}
pg = pd.read_csv('data/pool_grid.csv'); pg['kept'] = pg.natoms <= 28
NAMEMAP = {'GB_sarin':('iPr','Me'),'GF_cyclosarin':('cHx','Me'),'EtGB':('Et','Me'),
           'nPrGB':('nPr','Me'),'iBuGB':('iBu','Me')}
AGSMI={'EXP_GD_soman':'CC(C)(C)C(C)OP(C)(=O)F','GF_cyclosarin':'CP(=O)(F)OC1CCCCC1',
       'EXP_HD':'ClCCSCCCl','EXP_HN1':'CCN(CCCl)CCCl','EXP_HN2':'CN(CCCl)CCCl',
       'EXP_HN3':'ClCCN(CCCl)CCCl','EXP_A230':'CP(=O)(F)N=C(C)N(CC)CC',
       'EXP_A232':'COP(=O)(F)N=C(C)N(CC)CC','EXP_A234':'CCOP(=O)(F)N=C(C)N(CC)CC',
       'EXP_A242':'CP(=O)(F)N=C(N(CC)CC)N(CC)CC'}
rows=[]
for n in lib:
    if n.startswith('EXP_') or n=='GF_cyclosarin':
        rows.append((n.replace('_','\\_'), PRETTY.get(n,n), 'measured agent',
                     '\\texttt{%s}' % AGSMI[n]))
for n in lib:
    if n.startswith('EXP_') or n=='GF_cyclosarin': continue
    o,p = NAMEMAP.get(n, tuple(n.split('_')[1:3]))
    key = f'OP_{o}_{p}'
    sm = pg.loc[pg.name==key,'smiles']
    rows.append((n.replace('_','\\_'), f'{o}O--P({p})(=O)F', 'OP--F analogue', '\\texttt{%s}' % (sm.iloc[0] if len(sm) else '')))
tab('tab:si-library',
    'Composition of the 53-member computed spectral library. GF occupies the cyclohexyl/methyl cell of the analogue grid and is therefore both a measured agent and a grid member; it is counted once. HN2 was measured but failed the data-quality criterion (Table~\\ref{tab:si-quality}) and was retained in the library without being queried.',
    ['ORCA job','Molecule','Role','SMILES'], rows, 'lllp{4.2cm}',
    note='All spectra computed at B3LYP/6-311++G(d,p) with ORCA 6.1.1.')

# ---------- S2 grid ----------
order=['Me','Et','nPr','iPr','iBu','sBu','cPn','cHx']
piv = pg.pivot(index='o',columns='p',values='natoms').loc[order,order]
rows=[]
for o in order:
    r=[o]
    for p in order:
        v=int(piv.loc[o,p]); r.append(f'\\textbf{{{v}}}' if v>28 else str(v))
    rows.append(r)
tab('tab:si-grid',
    'Heavy-atom count of every cell of the $8\\times8$ R$^1$O--P(R$^2$)(=O)F substituent grid. Rows are the O-alkyl substituent R$^1$, columns the P-alkyl substituent R$^2$. Cells in bold exceed the 28 heavy-atom ceiling and were not computed; the remaining 44 cells constitute the analogue set.',
    ['R$^1$O $\\backslash$ R$^2$']+order, rows, 'l'+'r'*8,
    note='Substituents: Me = methyl, Et = ethyl, nPr = $n$-propyl, iPr = $iso$-propyl, iBu = $iso$-butyl, sBu = $sec$-butyl, cPn = cyclopentyl, cHx = cyclohexyl. The ceiling was imposed to keep the numerical-frequency cost of each job tractable.')

# ---------- S3 sessions ----------
inv={}
for f in glob.glob('data/exp_raman/augmented_csv/*.csv'):
    b=os.path.basename(f)
    m=re.match(r'([a-z0-9]+)_[A-Z]+_(\d+)_([0-9\-]+)_\d+\.csv$', b)
    if m: ag,s,ts = m.group(1), m.group(2), m.group(3)
    else: ag,s,ts = b.split('_')[0], '--', '--'
    inv.setdefault((ag,s,ts),0); inv[(ag,s,ts)]+=1
LAB={'gd':'GD','gf':'GF','hd':'HD','hn1':'HN1','hn2':'HN2','hn3':'HN3',
     'a230':'A-230','a232':'A-232','a234':'A-234','a242':'A-242'}
rows=[]
for (ag,s,ts),n in sorted(inv.items(), key=lambda x:(LAB[x[0][0]], x[0][1])):
    t = ts.replace('2026-07-20-','') if ts!='--' else 'not recorded'
    t = re.sub(r'^(\d\d)-(\d\d)-(\d\d).*$', r'\1:\2:\3', t)
    rows.append((LAB[ag], s, t, n))
tab('tab:si-sessions',
    'Inventory of measured Raman acquisitions. Each session is one independent acquisition; the replicate count is the number of spectra written by the instrument software for that session. All acquisitions belong to a single campaign on 20 July 2026.',
    ['Agent','Session ID','Start time (UTC+9)','Replicates'], rows, 'llrr',
    note='HD spectra were exported without session identifiers or timestamps, so HD contributes to the identification analysis but not to the session-level analyses of Figure~6; this is stated in the main text. HN2 sessions are listed for completeness but were excluded before identification.')

# ---------- S4 quality ----------
q=d6['qual']
rows=[(LAB[k], v[0], f'{v[1]:.3f}', 'retained' if v[0]>=5 else 'excluded') for k,v in
      sorted(q.items(), key=lambda x:-x[1][0])]
tab('tab:si-quality',
    'Data-quality criterion applied before any identification was attempted. A spectrum was retained if it showed at least five resolved bands above a normalized intensity of 0.15.',
    ['Agent','Resolved bands','Mean normalized intensity','Decision'], rows, 'lrrl',
    note='The threshold was fixed before the analysis. HN2 is the only agent that fails it, and fails by a wide margin.')

# ---------- S5/S6 band matching ----------
rows=[]; allp=[]
for ag,v in d3['data'].items():
    e=v['exp']/v['exp'].max(); c=v['dft']/v['dft'].max()
    pe,_=find_peaks(e,height=0.12,distance=6); pc,_=find_peaks(c,height=0.12,distance=6)
    we,wc=GRID[pe],GRID[pc]; pr=[]
    for w in we:
        j=int(np.argmin(abs(wc-w)))
        if abs(wc[j]-w)<=35: pr.append((ag,w,wc[j]))
    d=np.array([p[2]-p[1] for p in pr])
    rows.append((ag,len(we),len(wc),len(pr),f'{np.sqrt((d**2).mean()):.1f}',
                 f'{abs(d).mean():.1f}',f'{d.mean():+.1f}',f"{v['cos']:.3f}"))
    allp+=pr
A=np.array([[p[1],p[2]] for p in allp]); dd=A[:,1]-A[:,0]
rows.append(('\\textbf{All}', '', '', f'\\textbf{{{len(A)}}}', f'\\textbf{{{np.sqrt((dd**2).mean()):.1f}}}',
             f'\\textbf{{{abs(dd).mean():.1f}}}', f'\\textbf{{{dd.mean():+.1f}}}',
             f"\\textbf{{{np.mean([v['cos'] for v in d3['data'].values()]):.3f}}}"))
tab('tab:si-bands',
    'Agreement between calibrated computed and measured band positions, agent by agent. Bands were located with a local-maximum criterion (normalized intensity $>$ 0.12, minimum separation 6 grid points) and paired when computed and measured maxima fell within 35\\,cm$^{-1}$.',
    ['Agent','Measured bands','Computed bands','Matched','RMSE (\\wn)','MAE (\\wn)','Bias (\\wn)','Cosine'],
    rows, 'lrrrrrrr',
    note='Bias is the mean signed difference (computed $-$ measured); its near-zero value across the set indicates that the single frequency-scaling factor introduces no systematic offset. Not every measured band finds a computed partner: weak measured features and modes that the single-conformer calculation does not reproduce remain unmatched, and these are counted in the ``Measured bands`` column but not in ``Matched``.')

# full band list
per={}
for ag,we,wc in allp: per.setdefault(ag,[]).append((we,wc))
rows=[]
for ag in d3['data']:
    for we,wc in per.get(ag,[]):
        rows.append((ag, f'{we:.0f}', f'{wc:.0f}', f'{wc-we:+.0f}'))
half=(len(rows)+1)//2
two=[rows[i]+(rows[i+half] if i+half<len(rows) else ('','','',''))
     for i in range(half)]
tab('tab:si-bandlist',
    'All 137 matched band pairs used for the parity analysis of Figure~3j. Positions in cm$^{-1}$.',
    ['Agent','Meas.','Comp.','$\\Delta$','Agent','Meas.','Comp.','$\\Delta$'],
    [list(t) for t in two], 'lrrr|lrrr')

# ---------- S7 calibration grid ----------
sc=np.array(d2['map']['scales']); fw=np.array(d2['map']['fwhms']); T=np.array(d2['map']['top1'])
rows=[[int(fw[i])]+[f'{T[i,j]*9:.0f}' for j in range(len(sc))] for i in range(len(fw))]
tab('tab:si-calibgrid',
    'Full calibration grid search. Each entry is the number of the nine agents whose top-ranked library match is the correct molecule, for a given frequency scaling factor $s$ (columns) and Lorentzian FWHM $\\Gamma$ (rows, cm$^{-1}$).',
    ['$\\Gamma$']+[f'{v:.3f}' for v in sc], rows, 'r'+'r'*len(sc),
    note='The maximum of 6/9 is reached over a contiguous region spanning $s = 0.996$--$1.004$ and $\\Gamma = 25$--$40$\\,cm$^{-1}$; the adopted values $s = 1.000$, $\\Gamma = 30$\\,\\wn{} lie in its interior. Identification is therefore not sensitive to the precise calibration values.')

# ---------- S8 similarity vs accuracy ----------
rows=[(int(w), f'{ms:.3f}', f'{a1*9:.0f} ({a1*100:.1f}\\%)', f'{ac*9:.0f} ({ac*100:.1f}\\%)', f'{mg:+.4f}')
      for w,ms,a1,ac,mg in d2['pitfall']]
tab('tab:si-pitfall',
    'Mean spectral similarity and identification accuracy as a function of Lorentzian FWHM, at fixed $s = 1.000$. Mean similarity rises monotonically with broadening while top-1 accuracy peaks and then falls.',
    ['$\\Gamma$ (\\wn)','Mean cosine','Molecular top-1','Class top-1','Mean margin'],
    rows, 'rrrrr',
    note='``Mean cosine`` is averaged over all measured--computed pairs, which is the quantity a similarity-based fit would maximize. ``Mean margin`` is the mean of (similarity to the correct library member) $-$ (similarity to the best incorrect member); it becomes less negative with broadening even as top-1 accuracy declines, showing that the two criteria are not merely noisy versions of one another.')

# ---------- S9 session similarity ----------
wa,bc,ba = np.array(d6['wa']), np.array(d6['bc']), np.array(d6['ba'])
rows=[('Same agent, different session', len(wa), f'{wa.mean():.3f}', f'{wa.std(ddof=1):.3f}', f'{wa.min():.3f}', f'{wa.max():.3f}'),
      ('Different agent, same class',    len(bc), f'{bc.mean():.3f}', f'{bc.std(ddof=1):.3f}', f'{bc.min():.3f}', f'{bc.max():.3f}'),
      ('Different class',                len(ba), f'{ba.mean():.3f}', f'{ba.std(ddof=1):.3f}', f'{ba.min():.3f}', f'{ba.max():.3f}')]
tab('tab:si-sessionsim',
    'Pairwise cosine similarities between session-mean measured spectra, partitioned by relationship. 27 sessions across eight agents give 351 pairs.',
    ['Group','$n$ pairs','Mean','SD','Min','Max'], rows, 'lrrrrr',
    note='One-sided Mann--Whitney $U$ test, same-agent versus all other pairs: $p = %.1f\\times10^{-21}$. HD and HN2 are absent from this analysis (Table~\\ref{tab:si-sessions}).' % (d6['pv']*1e21))

# PCA variance
ev=np.array(d6['ev'])
rows=[(i+1, f'{v*100:.1f}', f'{ev[:i+1].sum()*100:.1f}') for i,v in enumerate(ev[:6])]
tab('tab:si-pca','Explained variance of the principal components of the 27 session-mean spectra (Figure~6b).',
    ['Component','Variance explained (\\%)','Cumulative (\\%)'], rows, 'rrr')

# ---------- S10 AL ----------
al=pd.read_csv('data/al_benchmark.csv')
piv=al.pivot_table(index='n_labeled',columns='strategy',values='rmse',aggfunc='mean')
sd =al.pivot_table(index='n_labeled',columns='strategy',values='rmse',aggfunc='std')
cols=['random','ucb_only','diversity_only','ucb_diversity']
rows=[]
for n in piv.index:
    r=[int(n)]
    for c in cols:
        v=piv.loc[n,c]
        r.append('--' if pd.isna(v) else f'{v:.4f} $\\pm$ {sd.loc[n,c]:.4f}')
    rows.append(r)
full=al[al.strategy=='full'].rmse
tab('tab:si-al',
    'Active-learning replay benchmark on the organophosphorus analogue pool. Entries are mean $\\pm$ SD of held-out RMSE over five seeds, with a test set fixed before the first acquisition round.',
    ['Labelled','Random','UCB','Diversity','UCB $+$ diversity'], rows, 'rllll',
    note='The full-pool ceiling (all 36 candidates labelled) is RMSE $= %.4f \\pm %.4f$. No acquisition strategy separates from random selection at any budget: the largest gap at any round is smaller than the seed-to-seed standard deviation. The pool is a single homologous series in which every candidate carries nearly the same information, which is the regime in which active learning has least to offer. Earlier versions of this benchmark that allowed the test set to shrink as candidates were acquired produced an apparent advantage for UCB; that advantage is an artifact of the changing test set and disappears under the fixed-test protocol reported here.' % (full.mean(), full.std(ddof=1)))

# ---------- S11 descriptors ----------
p=pd.read_csv('data/agent_properties.csv')
rows=[(r.agent, r.cls, f'{r.MW:.1f}', int(r.heavy), int(r.rotB), f'{r.fCSP3:.3f}', f'{r.TPSA:.1f}',
       int(r.nN), int(r.nO), int(r.nHal), int(r.vib), f'{r.self_cos:.3f}', f'{r.margin:+.3f}', f'{r.tani_bw:.3f}')
      for r in p.itertuples()]
tab('tab:si-desc',
    'Molecular descriptors of the nine agents, computed with RDKit, together with the spectral quantities used in the failure analysis.',
    ['Agent','Class','MW','Heavy','RotB','Fsp$^3$','TPSA','N','O','Hal','Modes','$s_\\mathrm{self}$','Margin','$T_\\mathrm{bw}$'],
    rows, 'llrrrrrrrrrrrr',
    note='RotB = rotatable bonds; TPSA = topological polar surface area (\\AA$^2$); Modes = number of real harmonic vibrational modes; $s_\\mathrm{self}$ = cosine similarity between the measured spectrum and its own computed spectrum; Margin = $s_\\mathrm{self}$ minus the similarity to the best incorrect library member (negative for the three misassigned agents); $T_\\mathrm{bw}$ = Morgan-fingerprint Tanimoto similarity to that best incorrect member.')

rows=[]
for c,lab in [('MW','Molecular weight'),('heavy','Heavy-atom count'),('rotB','Rotatable bonds'),
              ('fCSP3','Fraction sp$^3$ carbon'),('TPSA','TPSA'),('nN','Nitrogen count'),
              ('nO','Oxygen count'),('nHal','Halogen count'),('vib','Vibrational modes'),
              ('tani_bw','Tanimoto to best wrong match'),('self_cos','Measured--computed self-similarity')]:
    r,pv=pearsonr(p[c],p['margin'])
    rows.append((lab, f'{r:+.3f}', f'{pv:.3f}', 'yes' if pv<0.05 else 'no'))
tab('tab:si-corr',
    'Pearson correlation of each descriptor with the identification margin across the nine agents.',
    ['Quantity','$r$','$p$','$p<0.05$'], rows, 'lrrl',
    note='With $n = 9$ no structural descriptor reaches significance, and none survives correction for the eleven tests performed ($\\alpha = 0.0045$). The only quantity that correlates at the nominal level is the measured--computed self-similarity, which is not a molecular descriptor but a direct measure of how well the calculation reproduces that particular measurement; it therefore diagnoses a failure after the fact rather than predicting one for an agent that has not been measured. We report the table to document that the search was made and was negative, not to propose a predictor.')

# ---------- S12 Novichok cross-similarity ----------
DD=np.array(d5['DD']); NOV=d5['NOV']
rows=[[NOV[i]]+[('---' if i==j else f'{DD[i,j]:.3f}') for j in range(4)] for i in range(4)]
tab('tab:si-nov',
    'Cosine similarities among the four \\emph{computed} Novichok spectra. These contain no measurement noise; their mutual similarity sets a floor on how well the four can be told apart by library matching.',
    ['']+list(NOV), rows, 'l'+'r'*4,
    note='Values range from 0.795 to 0.906. The four agents share the P(=O)(F)--N=C amidine core and differ only in the phosphorus substituent and, for A-242, in guanidine substitution.')

nl=d5['nlow']
rows=[(k, v if not hasattr(v,'__len__') else v) for k,v in nl.items()]
tab('tab:si-lowfreq',
    'Number of resolved bands below 700\\,cm$^{-1}$ in the measured spectrum of each Novichok agent.',
    ['Agent','Bands below 700 \\wn'], rows, 'lr',
    note='A-232 is the only member of the series with no resolved low-frequency structure. The region carries the skeletal deformations that most strongly distinguish these congeners, and its absence in one member of an otherwise consistent series points to the measurement or to the condition of the material rather than to the chemistry.')

OUT.close()
print(open('si_tables.tex').read()[:1200])
print('...\nlines:', sum(1 for _ in open('si_tables.tex')))

import pandas as pd
D=pd.read_csv('si_freqcheck.csv')
OUT=open('si_tables.tex','a'); W=OUT.write
def tab2(label,caption,header,rows,colspec,note=None):
    W("\\begin{table}[htbp]\n\\centering\n")
    W("\\caption{%s}\n\\label{%s}\n" % (caption,label)); W("\\small\n")
    W("\\begin{tabular}{%s}\n\\toprule\n" % colspec)
    W(" & ".join(header)+" \\\\\n\\midrule\n")
    for r in rows: W(" & ".join(str(x) for x in r)+" \\\\\n")
    W("\\bottomrule\n\\end{tabular}\n")
    if note: W("\n\\begin{flushleft}\\footnotesize %s\\end{flushleft}\n" % note)
    W("\\end{table}\n\n")
D=D.sort_values('job')
half=(len(D)+1)//2
rows=[]
for i in range(half):
    a=D.iloc[i]; r=[a.job.replace('_','\\_'), int(a.n_real), int(a.n_imag), f'{a.lowest_real:.1f}']
    if i+half < len(D):
        b=D.iloc[i+half]; r+=[b.job.replace('_','\\_'), int(b.n_real), int(b.n_imag), f'{b.lowest_real:.1f}']
    else: r+=['','','','']
    rows.append(r)
tab2('tab:si-freq',
 'Vibrational analysis of every library member. $N_\\mathrm{real}$ is the number of real harmonic modes, $N_\\mathrm{imag}$ the number of imaginary modes, and $\\nu_\\mathrm{min}$ the lowest real wavenumber.',
 ['Job','$N_\\mathrm{real}$','$N_\\mathrm{imag}$','$\\nu_\\mathrm{min}$ (\\wn)']*2,
 rows, 'lrrr|lrrr',
 note='No structure in the library has an imaginary frequency, so all 53 geometries are true minima on the B3LYP/6-311++G(d,p) surface. The lowest real wavenumbers (5--73\\,\\wn) are soft torsions that fall below the 250\\,\\wn{} lower bound of the analysis window and therefore do not enter the spectral comparison.')
OUT.close()
