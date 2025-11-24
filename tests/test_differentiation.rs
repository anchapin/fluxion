use fluxion::ai::neural_field::NeuralScalarField;
use fluxion::physics::continuous::ContinuousField;
use fluxion::sim::components::WallSurface;
use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

/// Minimal Auto-Differentiation Scalar (Dual Number)
/// Represents a value f(x) and its derivative f'(x).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dual {
    val: f64,
    grad: f64,
}

impl Dual {
    fn new(val: f64, grad: f64) -> Self {
        Dual { val, grad }
    }
}

// Implement Add: (u, u') + (v, v') = (u+v, u'+v')
impl Add for Dual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Dual {
            val: self.val + rhs.val,
            grad: self.grad + rhs.grad,
        }
    }
}

// Implement AddAssign
impl AddAssign for Dual {
    fn add_assign(&mut self, rhs: Self) {
        self.val += rhs.val;
        self.grad += rhs.grad;
    }
}

// Implement Mul<f64>: (u, u') * c = (u*c, u'*c)
impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Dual {
            val: self.val * rhs,
            grad: self.grad * rhs,
        }
    }
}

// Implement Zero
impl Zero for Dual {
    fn zero() -> Self {
        Dual { val: 0.0, grad: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.val == 0.0 && self.grad == 0.0
    }
}

#[test]
fn test_backpropagation_through_integration() {
    // 1. Define Weight with Gradient = 1.0 (seed for backprop check)
    // We are computing d(HeatGain)/d(Weight).
    // In forward-mode AD (Dual numbers), we set input grad = 1.0.
    // The output grad will be d(Output)/d(Input).
    let weight_val = 2.0;
    let weight = Dual::new(weight_val, 1.0); // grad=1 means we are differentiating wrt this weight

    // 2. Create NeuralScalarField
    let weights = vec![weight];
    let field = NeuralScalarField::new(weights).unwrap();

    // 3. Define Surface
    let surface = WallSurface::new(2.0, 0.5); // Area = 2.0

    // 4. Calculate Heat Gain
    // Heat Gain = Area * Integral(Field)
    // Field(u,v) = Weight * 1.0 (Order 0 basis)
    // Integral = Weight * 1.0 * (1.0 - 0.0)^2 = Weight
    // Heat Gain = 2.0 * Weight

    // d(HeatGain)/d(Weight) should be 2.0.
    let heat_gain = surface.calculate_heat_gain(&field);

    println!("Heat Gain Value: {}", heat_gain.val);
    println!("Heat Gain Gradient (dGain/dWeight): {}", heat_gain.grad);

    // Verify Value: 2.0 * 2.0 = 4.0
    assert!((heat_gain.val - 4.0).abs() < 1e-6);

    // Verify Gradient: 2.0
    assert!((heat_gain.grad - 2.0).abs() < 1e-6);
}

#[test]
fn test_complex_field_gradient() {
    // Test with Order 1 (sine wave)
    // Field(u,v) = w0 + w1*cos(pi*u) + w2*sin(pi*u) ...
    // Let's set w2 = 1.0 (the sin(pi*u) term) and differentiate wrt w2.
    // Index for sin(pi*u) in u-basis is 2. Index for 1 in v-basis is 0.
    // Flattened index = 2 * (2*order+1) + 0?
    // Order 1 -> 3 terms: 1, cos, sin.
    // Flattened size = 9.
    // u_term index 2 (sin), v_term index 0 (1).
    // Index in weights = 2 * 3 + 0 = 6.

    let mut weights = vec![Dual::new(0.0, 0.0); 9];
    // Set w[6] to 1.0 and we want d/dw[6]
    weights[6] = Dual::new(1.0, 1.0);

    let field = NeuralScalarField::new(weights).unwrap();

    // Integral of sin(pi*u)*1 over [0,1]x[0,1] is 2/pi.
    // Area = 1.0
    let surface = WallSurface::new(1.0, 0.5);

    let heat_gain = surface.calculate_heat_gain(&field);

    let expected_val = 2.0 / std::f64::consts::PI;

    println!("Complex Heat Gain Value: {}", heat_gain.val);
    println!("Complex Heat Gain Gradient: {}", heat_gain.grad);

    assert!((heat_gain.val - expected_val).abs() < 1e-6);
    // Since HeatGain is linear w.r.t weights: HeatGain = C * w6 + ...
    // d(HeatGain)/dw6 = C = expected_val.
    assert!((heat_gain.grad - expected_val).abs() < 1e-6);
}
