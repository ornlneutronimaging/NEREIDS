//! Shared fitting types: `FlatMatrix` and the `FitModel` trait.

use crate::error::FittingError;

/// Row-major flat matrix.
#[derive(Debug, Clone)]
pub struct FlatMatrix {
    /// Row-major storage: `data[i * ncols + j]`.
    pub data: Vec<f64>,
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
}

impl FlatMatrix {
    /// A zero-filled matrix.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let len = nrows
            .checked_mul(ncols)
            .expect("FlatMatrix dimensions overflow usize");
        Self {
            data: vec![0.0; len],
            nrows,
            ncols,
        }
    }

    /// Element at `(row, col)`.
    #[inline(always)]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.nrows && col < self.ncols);
        self.data[row * self.ncols + col]
    }

    /// Mutable element at `(row, col)`.
    #[inline(always)]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        debug_assert!(row < self.nrows && col < self.ncols);
        &mut self.data[row * self.ncols + col]
    }
}

/// A model that can be fitted to data.
pub trait FitModel {
    /// The model's predictions at `params`, one per data point.
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError>;

    /// The Jacobian at `params` if available analytically, else `None`.
    fn analytical_jacobian(
        &self,
        _params: &[f64],
        _free_param_indices: &[usize],
        _y_current: &[f64],
    ) -> Option<FlatMatrix> {
        None
    }
}

impl<M: FitModel + ?Sized> FitModel for &M {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        (**self).evaluate(params)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        (**self).analytical_jacobian(params, free_param_indices, y_current)
    }
}

impl<M: FitModel + ?Sized> FitModel for Box<M> {
    fn evaluate(&self, params: &[f64]) -> Result<Vec<f64>, FittingError> {
        (**self).evaluate(params)
    }

    fn analytical_jacobian(
        &self,
        params: &[f64],
        free_param_indices: &[usize],
        y_current: &[f64],
    ) -> Option<FlatMatrix> {
        (**self).analytical_jacobian(params, free_param_indices, y_current)
    }
}
