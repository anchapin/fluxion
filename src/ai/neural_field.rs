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
}
