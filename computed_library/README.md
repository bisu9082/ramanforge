# computed_library

Calibrated line lists for the 54 members of the computed library, one CSV per ORCA job,
with `MANIFEST.csv` giving the agent, the role, the isomeric SMILES derived from the
optimised geometry and the InChIKey.

## Do not use this as a search library

This is the object the paper evaluates, and the paper's result is that it does not work
for identification. Against the authentic agents it was calibrated on, the entries
reproduce the correct molecule at a cosine of 0.65 to 0.80 and assign 13 of 30 genuine
acquisitions correctly. No operating threshold separates a query whose answer is in the
library from one whose answer is not. It is deposited so that the result can be checked and
so that a better calibration can be tried against the same line lists, not so that it can
be searched.

Some members of the analogue grid are Schedule 1 chemicals. They are present as
computational distractors; the structures are in the open literature and, as above, their
predicted spectra do not identify them.
