use crate::physics::continuous::ContinuousField;

/// Represents a wall surface in a thermal zone.
#[derive(Clone, Debug)]
pub struct WallSurface {
    /// Area of the surface in square meters (m²).
    pub area: f64,
    /// Thermal transmittance of the surface (W/m²K).
    pub u_value: f64,
}

impl WallSurface {
    /// Create a new WallSurface.
    pub fn new(area: f64, u_value: f64) -> Self {
        WallSurface { area, u_value }
    }

    /// Calculate heat gain for this surface given a continuous field representing
    /// the heat flux (W/m²) or similar potential over the surface.
    ///
    /// The field is assumed to be defined over the normalized domain [0, 1] x [0, 1].
    /// The integration result (total value over normalized domain) is scaled by the area.
    pub fn calculate_heat_gain(&self, field: &impl ContinuousField) -> f64 {
        // Integrate the field over the normalized domain [0, 1] x [0, 1]
        // If the field represents W/m², the integral over normalized domain represents the average flux * 1.0?
        // Wait, if the field is defined on [0,1]x[0,1], we treat it as a mapping from the physical surface.
        // Usually, Integral(Flux dA) = Total Heat.
        // If the field returns Flux (W/m²) at normalized coordinates (u,v):
        // Integral(Flux(u,v) * dA) = Integral(Flux(u,v) * |J| du dv).
        // For a rectangular surface of area A mapped to [0,1]x[0,1], |J| = A.
        // So Total Heat = A * Integral_{0}^{1} Integral_{0}^{1} Flux(u,v) du dv.

        let integral = field.integrate(0.0, 1.0, 0.0, 1.0);
        self.area * integral
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::continuous::ConstantField;

    #[test]
    fn test_heat_gain_constant() {
        let surface = WallSurface::new(10.0, 0.5);
        let field = ConstantField { value: 2.0 }; // 2.0 W/m²

        // Total heat = 10.0 m² * 2.0 W/m² * 1.0 (integral over unit square) = 20.0 W
        let heat_gain = surface.calculate_heat_gain(&field);
        assert!((heat_gain - 20.0).abs() < 1e-6);
    }
}
