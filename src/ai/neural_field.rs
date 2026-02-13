use crate::physics::continuous::ContinuousField;
use ndarray::ArrayD;
use num_traits::Zero;
use ort::{session::Session, value::Value};
use std::f64::consts::PI;
use std::ops::{Add, AddAssign, Mul};
use std::path::Path;

/// A continuous scalar field defined by a set of weights for a Fourier basis.
///
/// The field is defined on the domain [0, 1] x [0, 1].
/// The basis functions are a tensor product of Fourier series terms.
///
/// Basis terms are ordered as follows:
/// Let K be the truncation order (number of harmonics).
/// The 1D basis functions are: 1, cos(pi*x), sin(pi*x), cos(2*pi*x), sin(2*pi*x), ...
///
/// The 2D basis is the product of these 1D bases.
/// The weights vector corresponds to the flattened tensor product.
#[derive(Debug, Clone)]
pub struct NeuralScalarField<T> {
    weights: Vec<T>,
    order: usize,
}

impl<T> NeuralScalarField<T> {
    /// Creates a new NeuralScalarField from a vector of weights.
    /// The length of weights must be perfect square, and (sqrt(len) - 1) % 2 == 0.
    /// Number of 1D terms = 1 + 2 * order.
    /// Total weights = (1 + 2 * order)^2.
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

    /// Helper to evaluate 1D Fourier basis at a point x
    /// Returns a vector of values [1, cos(pi*x), sin(pi*x), ...]
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

    /// Helper to integrate 1D Fourier basis from a to b
    /// Returns a vector of values [ (b-a), (sin(k*pi*b) - sin(k*pi*a))/(k*pi), ... ]
    fn integrate_basis_1d(a: f64, b: f64, order: usize) -> Vec<f64> {
        let mut values = Vec::with_capacity(1 + 2 * order);
        values.push(b - a); // Integral of 1 is x -> b - a

        for k in 1..=order {
            let k_pi = (k as f64) * PI;
            let div = 1.0 / k_pi;

            // Integral of cos(k*pi*x) is (1/k*pi) * sin(k*pi*x)
            let int_cos = div * ((k_pi * b).sin() - (k_pi * a).sin());
            values.push(int_cos);

            // Integral of sin(k*pi*x) is -(1/k*pi) * cos(k*pi*x)
            let int_sin = -div * ((k_pi * b).cos() - (k_pi * a).cos());
            values.push(int_sin);
        }
        values
    }
}

impl NeuralScalarField<f64> {
    /// Loads an ONNX model, runs it with the provided input, and creates a NeuralScalarField
    /// from the output weights.
    ///
    /// The model must have a single output tensor that can be flattened to a vector of f64 (or f32).
    pub fn from_onnx<P: AsRef<Path>>(
        model_path: P,
        input: ArrayD<f32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut session = Session::builder()?.commit_from_file(model_path)?;

        // Convert ndarray to shape+vec to bypass version mismatch
        // OwnedTensorArrayData requires Vec<T>, not &[T]
        let shape: Vec<i64> = input.shape().iter().map(|&x| x as i64).collect();
        let input_vec = input
            .as_slice()
            .ok_or("Input array is not contiguous")?
            .to_vec();

        let input_tensor = Value::from_array((shape, input_vec))?;
        let outputs = session.run(ort::inputs![input_tensor])?;

        // Assuming the first output is the weights
        let (_, output_tensor) = outputs.iter().next().ok_or("No output from model")?;

        // Try to extract as f32 (common for ONNX) or f64
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
        // same as v_basis.len()

        let mut sum = T::zero();
        let mut idx = 0;

        // The weights are flattened in row-major order
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
        // Order 0: Only 1 term (constant). 1x1 = 1 weight.
        let weights = vec![5.0];
        let field = NeuralScalarField::new(weights).unwrap();

        // at(u,v) should always be 5.0 * 1 * 1 = 5.0
        assert_eq!(field.at(0.0, 0.0), 5.0);
        assert_eq!(field.at(0.5, 0.5), 5.0);

        // Integrate over [0,1]x[0,1] -> 5.0 * 1 * 1 = 5.0
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        assert!((integral - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_simple_sine_field() {
        // Order 1. Terms: 1, cos(pi*x), sin(pi*x). (3 terms).
        // Total weights: 3x3 = 9.
        // We want f(u,v) = sin(pi*u).
        // This corresponds to u_term = sin(pi*u) (index 2), v_term = 1 (index 0).
        // Weight index: i=2, j=0 => index = 2 * 3 + 0 = 6.
        let mut weights = vec![0.0; 9];
        weights[6] = 1.0;

        let field = NeuralScalarField::new(weights).unwrap();

        // at(0.5, 0.0) -> sin(pi*0.5) = 1.0
        assert!((field.at(0.5, 0.0) - 1.0).abs() < 1e-6);

        // Integrate sin(pi*u) on [0,1].
        // Integral sin(pi*u) du = [-1/pi cos(pi*u)]_0^1 = -1/pi (-1 - 1) = 2/pi.
        // v integral of 1 on [0,1] is 1.
        // Total = 2/pi approx 0.6366
        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        let expected = 2.0 / PI;
        assert!((integral - expected).abs() < 1e-6);
    }
}
