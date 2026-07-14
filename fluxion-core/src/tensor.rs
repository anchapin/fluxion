use num_traits::Zero;
use std::ops::{Add, AddAssign, Mul};

pub trait ContinuousField<T>
where
    T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
{
    fn at(&self, u: f64, v: f64) -> T;

    fn integrate(&self, min_u: f64, max_u: f64, min_v: f64, max_v: f64) -> T {
        let steps = 100;
        let du = (max_u - min_u) / steps as f64;
        let dv = (max_v - min_v) / steps as f64;
        let mut sum = T::zero();

        for i in 0..steps {
            for j in 0..steps {
                let u = min_u + (i as f64 + 0.5) * du;
                let v = min_v + (j as f64 + 0.5) * dv;
                sum += self.at(u, v) * (du * dv);
            }
        }
        sum
    }
}

pub struct ConstantField<T> {
    pub value: T,
}

impl<T> ContinuousField<T> for ConstantField<T>
where
    T: Add<Output = T> + AddAssign + Mul<f64, Output = T> + Zero + Clone,
{
    fn at(&self, _u: f64, _v: f64) -> T {
        self.value.clone()
    }
}
