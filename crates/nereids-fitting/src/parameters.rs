//! Fit parameter types, bounds, and constraints.
//!
//! Parameters for the forward model that can be fitted or held fixed.
//! Supports non-negativity constraints and sum-to-one constraints for
//! isotopes of the same element.

/// A single fit parameter with value, bounds, and fixed/free flag.
#[derive(Debug, Clone)]
pub struct FitParameter {
    /// Parameter name (for reporting).
    pub name: String,
    /// Current value.
    pub value: f64,
    /// Lower bound (f64::NEG_INFINITY if unbounded).
    pub lower: f64,
    /// Upper bound (f64::INFINITY if unbounded).
    pub upper: f64,
    /// If true, parameter is held fixed during fitting.
    pub fixed: bool,
}

impl FitParameter {
    /// Create a new free parameter with non-negativity constraint.
    pub fn non_negative(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            lower: 0.0,
            upper: f64::INFINITY,
            fixed: false,
        }
    }

    /// Create a new free parameter with no bounds.
    pub fn unbounded(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            fixed: false,
        }
    }

    /// Create a fixed parameter.
    pub fn fixed(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            fixed: true,
        }
    }

    /// Clamp value to bounds.
    pub fn clamp(&mut self) {
        self.value = self.value.clamp(self.lower, self.upper);
    }
}

/// Collection of fit parameters for a forward model fit.
#[derive(Debug, Clone)]
pub struct ParameterSet {
    /// All parameters (fixed + free).
    pub params: Vec<FitParameter>,
}

impl ParameterSet {
    pub fn new(params: Vec<FitParameter>) -> Self {
        Self { params }
    }

    /// Number of free (non-fixed) parameters.
    pub fn n_free(&self) -> usize {
        self.params.iter().filter(|p| !p.fixed).count()
    }

    /// Get the values of all free parameters as a vector.
    pub fn free_values(&self) -> Vec<f64> {
        self.params
            .iter()
            .filter(|p| !p.fixed)
            .map(|p| p.value)
            .collect()
    }

    /// Set the values of free parameters from a vector.
    pub fn set_free_values(&mut self, values: &[f64]) {
        let mut j = 0;
        for p in &mut self.params {
            if !p.fixed {
                p.value = values[j];
                p.clamp();
                j += 1;
            }
        }
    }

    /// Get the value of all parameters (fixed + free) as a vector.
    pub fn all_values(&self) -> Vec<f64> {
        self.params.iter().map(|p| p.value).collect()
    }

    /// Indices (into `params`) of free parameters.
    pub fn free_indices(&self) -> Vec<usize> {
        self.params
            .iter()
            .enumerate()
            .filter(|(_, p)| !p.fixed)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_set_free_values() {
        let params = ParameterSet::new(vec![
            FitParameter::non_negative("a", 1.0),
            FitParameter::fixed("b", 2.0),
            FitParameter::non_negative("c", 3.0),
        ]);

        assert_eq!(params.n_free(), 2);
        assert_eq!(params.free_values(), vec![1.0, 3.0]);
        assert_eq!(params.all_values(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_set_free_values_with_clamping() {
        let mut params = ParameterSet::new(vec![
            FitParameter::non_negative("a", 1.0),
            FitParameter::fixed("b", 2.0),
            FitParameter::non_negative("c", 3.0),
        ]);

        // Set a to -0.5 (should clamp to 0.0) and c to 5.0
        params.set_free_values(&[-0.5, 5.0]);
        assert_eq!(params.params[0].value, 0.0); // clamped
        assert_eq!(params.params[1].value, 2.0); // fixed, unchanged
        assert_eq!(params.params[2].value, 5.0);
    }
}
