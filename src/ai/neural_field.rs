use crate::physics::continuous::ContinuousField;
use ndarray::ArrayD;
use num_traits::Zero;
use ort::{session::Session, value::Value};
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Mul};
use std::path::Path;

/// A continuous scalar field defined by a set of weights for a Fourier basis.
#[derive(Debug, Clone)]
pub struct NeuralScalarField<T> {
    weights: Vec<T>,
    order: usize,
}

impl<T> NeuralScalarField<T> {
    pub fn new(weights: Vec<T>) -> Result<Self, String> {
        let len = weights.len();
        let side = (len as f64).sqrt() as usize;
        if side * side != len {
            return Err(format!("Weights length {} must be a perfect square", len));
        }
        if side == 0 {
            return Err("Weights cannot be empty".to_string());
        }
        if !(side - 1).is_multiple_of(2) {
            return Err(format!(
                "Invalid number of terms per dimension: {}. Must be odd (1 + 2*order)",
                side
            ));
        }
        let order = (side - 1) / 2;
        Ok(Self { weights, order })
    }

    fn evaluate_basis_1d(x: f64, order: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(1 + 2 * order);
        values.push(1.0);
        for k in 1..=order {
            let k_pi_x = (k as f64) * PI * x;
            values.push(k_pi_x.cos());
            values.push(k_pi_x.sin());
        }
        values
    }

    fn integrate_basis_1d(a: f64, b: f64, order: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(1 + 2 * order);
        values.push(b - a);

        for k in 1..=order {
            let k_pi = (k as f64) * PI;
            let div = 1.0 / k_pi;
            let int_cos = div * ((k_pi * b).sin() - (k_pi * a).sin());
            values.push(int_cos);
            let int_sin = -div * ((k_pi * b).cos() - (k_pi * a).cos());
            values.push(int_sin);
        }
        values
    }
}

impl NeuralScalarField<f64> {
    pub fn from_onnx<P: AsRef<Path>>(
        model_path: P,
        input: ArrayD<f32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut session = Session::builder()?.commit_from_file(model_path)?;

        let shape: Vec<usize> = input.shape().to_vec();
        let (data, _offset) = input.into_raw_vec_and_offset();

        let input_tensor = Value::from_array((shape, data))?;
        let outputs = session.run(ort::inputs![input_tensor])?;

        let (_, output_tensor) = outputs.iter().next().ok_or("No output from model")?;

        let weights: Vec<f64> = if let Ok((_, data)) = output_tensor.try_extract_tensor::<f32>() {
            data.iter().map(|&x| x as f64).collect()
        } else if let Ok((_, data)) = output_tensor.try_extract_tensor::<f64>() {
            data.to_vec()
        } else {
            return Err("Output tensor data type must be f32 or f64".into());
        };

        Ok(Self::new(weights)?)
    }
}

impl<T> ContinuousField<T> for NeuralScalarField<T>
where
    T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
{
    fn at(&self, u: f64, v: f64) -> T {
        let u_basis = Self::evaluate_basis_1d(u, self.order);
        let v_basis = Self::evaluate_basis_1d(v, self.order);

        let mut sum = T::zero();
        let mut idx = 0;

        for u_val in &u_basis {
            for v_val in &v_basis {
                let term = self.weights[idx].clone() * (*u_val * *v_val);
                sum += term;
                idx += 1;
            }
        }
        sum
    }

    fn integrate(&self, min_u: f64, max_u: f64, min_v: f64, max_v: f64) -> T {
        let u_int = Self::integrate_basis_1d(min_u, max_u, self.order);
        let v_int = Self::integrate_basis_1d(min_v, max_v, self.order);

        let mut sum = T::zero();
        let mut idx = 0;

        for u_val in &u_int {
            for v_val in &v_int {
                let term = self.weights[idx].clone() * (*u_val * *v_val);
                sum += term;
                idx += 1;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_field() {
        let weights = vec![5.0];
        let field = NeuralScalarField::new(weights).unwrap();
        assert_eq!(field.at(0.0, 0.0), 5.0);
        assert_eq!(field.at(0.5, 0.5), 5.0);
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!((integral - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simple_sine_field() {
        let mut weights = vec![0.0; 9];
        weights[6] = 1.0;
        let field = NeuralScalarField::new(weights).unwrap();
        assert!((field.at(0.5, 0.0) - 1.0).abs() < 1e-6);
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        let expected = 2.0 / PI;
        assert!((integral - expected).abs() < 1e-6);
    }

    #[test]
    fn test_neural_field_not_square() {
        let weights = vec![1.0, 2.0, 3.0];
        let result = NeuralScalarField::new(weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("perfect square"));
    }

    #[test]
    fn test_neural_field_empty() {
        let weights: Vec<f64> = vec![];
        let result = NeuralScalarField::new(weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn test_neural_field_invalid_order() {
        let weights = vec![1.0; 4];
        let result = NeuralScalarField::new(weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Must be odd"));
    }

    #[test]
    fn test_neural_field_order_2() {
        let weights = vec![1.0; 25];
        let field = NeuralScalarField::new(weights).unwrap();
        let val = field.at(0.0, 0.0);
        assert!(val.is_finite());
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!(integral.is_finite());
    }

    #[test]
    fn test_neural_field_evaluate_basis() {
        let basis = NeuralScalarField::<f64>::evaluate_basis_1d(0.0, 1);
        assert_eq!(basis.len(), 3);
        assert!((basis[0] - 1.0).abs() < 1e-10);
        assert!((basis[1] - 1.0).abs() < 1e-10);
        assert!((basis[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_integrate_basis() {
        let int = NeuralScalarField::<f64>::integrate_basis_1d(0.0, 1.0, 1);
        assert_eq!(int.len(), 3);
        assert!((int[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_evaluate_basis_order_0() {
        let basis = NeuralScalarField::<f64>::evaluate_basis_1d(0.5, 0);
        assert_eq!(basis.len(), 1);
        assert!((basis[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_integrate_basis_order_0() {
        let int = NeuralScalarField::<f64>::integrate_basis_1d(0.0, 2.0, 0);
        assert_eq!(int.len(), 1);
        assert!((int[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_integrate_subrange() {
        let weights = vec![5.0];
        let field = NeuralScalarField::new(weights).unwrap();
        let integral = field.integrate(0.25, 0.75, 0.25, 0.75);
        assert!((integral - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_neural_field_at_various_points() {
        let weights = vec![3.0];
        let field = NeuralScalarField::new(weights).unwrap();
        assert_eq!(field.at(0.0, 0.0), 3.0);
        assert_eq!(field.at(1.0, 1.0), 3.0);
        assert_eq!(field.at(0.33, 0.67), 3.0);
    }

    #[test]
    fn test_neural_field_order_1_at_origin() {
        let weights = vec![1.0; 9];
        let field = NeuralScalarField::new(weights).unwrap();
        let val = field.at(0.0, 0.0);
        assert!(val.is_finite());
    }

    #[test]
    fn test_neural_field_clone() {
        let weights = vec![5.0];
        let field = NeuralScalarField::new(weights).unwrap();
        let cloned = field.clone();
        assert_eq!(field.at(0.0, 0.0), cloned.at(0.0, 0.0));
        assert_eq!(field.order, cloned.order);
    }

    #[test]
    fn test_neural_field_debug_format() {
        let weights = vec![5.0];
        let field = NeuralScalarField::new(weights).unwrap();
        let debug_str = format!("{:?}", field);
        assert!(debug_str.contains("NeuralScalarField"));
    }

    #[test]
    fn test_neural_field_evaluate_basis_at_half() {
        let basis = NeuralScalarField::<f64>::evaluate_basis_1d(0.5, 2);
        assert_eq!(basis.len(), 5);
        // cos(pi * 0.5) = 0, sin(pi * 0.5) = 1
        assert!((basis[1] - 0.0).abs() < 1e-10);
        assert!((basis[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_integrate_basis_symmetric() {
        let int = NeuralScalarField::<f64>::integrate_basis_1d(-1.0, 1.0, 1);
        assert_eq!(int.len(), 3);
        // Integral of cos(k*pi*x) from -1 to 1 for k=1 should be 0
        assert!((int[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_order_2_at_multiple_points() {
        let weights = vec![2.0; 25];
        let field = NeuralScalarField::new(weights).unwrap();

        for u in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for v in [0.0, 0.5, 1.0] {
                let val = field.at(u, v);
                assert!(val.is_finite(), "Value at ({}, {}) should be finite", u, v);
            }
        }
    }

    #[test]
    fn test_neural_field_integrate_full_domain() {
        let weights = vec![1.0; 9];
        let field = NeuralScalarField::new(weights).unwrap();
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!(integral.is_finite());
        // For order=1 with all weights=1, the integral should be positive
        assert!(integral > 0.0);
    }

    #[test]
    fn test_neural_field_order_2_integrate_non_unit_domain() {
        let weights = vec![1.0; 25];
        let field = NeuralScalarField::new(weights).unwrap();
        let integral = field.integrate(-1.0, 1.0, -1.0, 1.0);
        assert!(integral.is_finite());
    }

    #[test]
    fn test_neural_field_order_1_integrate_subrange() {
        let weights = vec![2.0; 9];
        let field = NeuralScalarField::new(weights).unwrap();
        let integral = field.integrate(0.0, 0.5, 0.0, 0.5);
        assert!(integral.is_finite());
        assert!(integral > 0.0);
    }

    #[test]
    fn test_neural_field_evaluate_basis_at_quarter() {
        let basis = NeuralScalarField::<f64>::evaluate_basis_1d(0.25, 2);
        assert_eq!(basis.len(), 5);
        assert!((basis[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_integrate_basis_at_half_domain() {
        let int = NeuralScalarField::<f64>::integrate_basis_1d(0.0, 0.5, 1);
        assert_eq!(int.len(), 3);
        assert!((int[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_order_2_at_corner_points() {
        let weights = vec![1.0; 25];
        let field = NeuralScalarField::new(weights).unwrap();
        let val_00 = field.at(0.0, 0.0);
        let val_11 = field.at(1.0, 1.0);
        assert!(val_00.is_finite());
        assert!(val_11.is_finite());
    }

    #[test]
    fn test_neural_field_new_error_not_square() {
        let weights = vec![1.0; 5];
        let result = NeuralScalarField::new(weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_neural_field_new_error_even_side() {
        let weights = vec![1.0; 16];
        let result = NeuralScalarField::new(weights);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Must be odd"));
    }

    #[test]
    fn test_neural_field_evaluate_basis_1d_order_3() {
        let basis = NeuralScalarField::<f64>::evaluate_basis_1d(0.0, 3);
        assert_eq!(basis.len(), 7);
        assert!((basis[0] - 1.0).abs() < 1e-10);
        assert!((basis[1] - 1.0).abs() < 1e-10);
        assert!((basis[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_integrate_basis_1d_order_2() {
        let int = NeuralScalarField::<f64>::integrate_basis_1d(0.0, 1.0, 2);
        assert_eq!(int.len(), 5);
        assert!((int[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_neural_field_order_3_evaluation() {
        let weights = vec![1.0; 49];
        let field = NeuralScalarField::new(weights).unwrap();
        let val = field.at(0.5, 0.5);
        assert!(val.is_finite());
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!(integral.is_finite());
    }
}
