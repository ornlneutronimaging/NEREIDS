# SAMMY tr147: Al-27 in R-Matrix Limited form

From SAMMY's test suite, `samtry/tr147/answers` (code-int.ornl.gov `rnsd/sammy`,
commit `b1205aac`; the `.lpt` files are identical at code.ornl.gov `RNSD/SAMMY`
commit `339d9f83`).

- `ra1.ndf`, `rb1.ndf`: LRF=7 (KRM=3) File 2 that SAMMY wrote for Al-27 over
  10–840 keV. Case a has one neutron channel per spin group; case b gives every
  allowed channel, with zero widths where the evaluation has none.
- `ra2.lpt`, `rb2.lpt`: SAMRML elastic and nonelastic cross sections (barn) of
  those files at 13 energies, unbroadened.
