# SAMMY tr147: Al-27 and O-16 in R-Matrix Limited form

From SAMMY's test suite, `samtry/tr147/answers` (code-int.ornl.gov `rnsd/sammy`,
commit `b1205aac`; the `.lpt` files are identical at code.ornl.gov `RNSD/SAMMY`
commit `339d9f83`).

- `ra1.ndf`, `rb1.ndf`: LRF=7 (KRM=3) File 2 that SAMMY wrote for Al-27 over
  10–840 keV. Case a has one neutron channel per spin group; case b gives every
  allowed channel, with zero widths where the evaluation has none.
- `ra2.lpt`, `rb2.lpt`: SAMRML elastic and nonelastic cross sections (barn) of
  those files at 13 energies, unbroadened.
- `rf1.ndf`: LRF=7 (KRM=3) File 2 that SAMMY wrote for O-16 over 1e-5 eV to
  6.3 MeV, with an (n,α) pair (Q = −2.2156 MeV) that opens inside the range.
- `rf2.lpt`: SAMRML elastic, nonelastic and (n,α) cross sections (barn) of
  that file at 8 energies, unbroadened.
