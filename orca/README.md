# ORCA inputs and outputs

`outputs/` holds the 57 production calculations that make up the computed library, plus
the batch driver logs. `outputs_b2/` holds the ten agent–correction pairs used for the
level-of-theory escalation reported in Section S8.

`inputs/` and `inputs_b2/` are reconstructed **verbatim** from the `INPUT FILE` echo that
ORCA writes at the top of each output, so every input here is byte-for-byte what was run.
That is also why some of them, and the batch logs, carry Korean comments: they are the
working record as it was produced, and translating them would make the deposited input
differ from the one the output was generated with. The recurring comments are:

| Korean | English |
|---|---|
| `# 출발 기하: X.out 수렴 구조` | Starting geometry: converged structure from `X.out` |
| `# P·F·S·Cl만 def2-TZVPP로 승격. 인 진동수의 고전적 병목(tight-d 부족) 해소.` | def2-TZVPP on P, F, S and Cl only: relieves the classic bottleneck for phosphorus frequencies (insufficient tight d functions) |
| `# 응축상 유전 효과. 순수액상 작용제의 유전율(~20)에 가까운 아세톤으로 근사.` | Condensed-phase dielectric effect, approximated with acetone, whose permittivity (~20) is close to that of the neat liquid agents |
| `# Kim 2026이 future work로 예고한 분산·장거리 보정 범함수.` | Dispersion-corrected, range-separated functional, the escalation flagged as future work by Kim 2026 |

## Production level of theory

```
! B3LYP 6-311++G(d,p) TightOpt NumFreq
%pal nprocs 8 end
%maxcore 3000
%elprop Polar 1 end
```

Every production structure converged with zero imaginary frequencies. ORCA 6.1.1.
