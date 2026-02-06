//! Beer-Lambert transmission: `T(E) = exp(-Σ_i n_i * d_i * σ_i(E))`
//!
//! Also computes the Jacobian `dT/dn_i` for gradient-based fitting.
