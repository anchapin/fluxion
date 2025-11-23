pub trait ContinuousField {
    fn at(&self, u: f64, v: f64) -> f64;

    fn integrate(&self, min_u: f64, max_u: f64, min_v: f64, max_v: f64) -> f64 {
        let steps = 100;
        let du = (max_u - min_u) / steps as f64;
        let dv = (max_v - min_v) / steps as f64;
        let mut sum = 0.0;

        for i in 0..steps {
            for j in 0..steps {
                let u = min_u + (i as f64 + 0.5) * du;
                let v = min_v + (j as f64 + 0.5) * dv;
                sum += self.at(u, v) * du * dv;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstantField {
        value: f64,
    }

    impl ContinuousField for ConstantField {
        fn at(&self, _u: f64, _v: f64) -> f64 {
            self.value
        }
    }

    #[test]
    fn test_constant_field_integration() {
        let field = ConstantField { value: 5.0 };
        let result = field.integrate(0.0, 1.0, 0.0, 1.0);
        // Area is 1.0 * 1.0 = 1.0. Value is 5.0. Integral should be 5.0.
        // Using midpoint rule, it should be exact for constant functions.
        assert!((result - 5.0).abs() < 1e-6);
    }
}
