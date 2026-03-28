#!/usr/bin/env rust-script
//! Test script to check CTF coefficient values

use fluxion::physics::ctf_coefficients::{CTFCalculator, CTFMaterial};

fn main() {
    let layers = vec![
        CTFMaterial::new("Gypsum", 0.013, 0.16, 800.0, 1090.0),
        CTFMaterial::new("Concrete", 0.100, 0.51, 2240.0, 920.0),
        CTFMaterial::new("Foam", 0.0615, 0.04, 30.0, 840.0),
        CTFMaterial::new("Wood", 0.009, 0.16, 500.0, 2090.0),
    ];

    let coeffs = CTFCalculator::with_defaults(&layers, 3600.0).compute_coefficients();

    println!("First 10 coefficients:");
    println!("  X[0..10] = {:?}", &coeffs.x[0..10]);
    println!("  Y[0..10] = {:?}", &coeffs.y[0..10]);
    println!("  Φ[0..10] = {:?}", &coeffs.phi[0..10]);
    println!("\nSum X = {:.6}", coeffs.x.iter().sum::<f64>());
    println!("Sum Y = {:.6}", coeffs.y.iter().sum::<f64>());
    println!("Sum Φ = {:.6}", coeffs.phi.iter().sum::<f64>());
}
