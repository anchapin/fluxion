use fluxion::physics::cta::VectorField;
use fluxion::sim::engine::ThermalModel;
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

#[test]
fn verify_hext() {
    let spec = ASHRAE140Case::Case195.spec();
    let model = ThermalModel::<VectorField>::from_spec(&spec);

    println!("=== Current h values ===");
    println!("h_tr_w: {:.2}", model.h_tr_w[0]); // Window conductance
    println!("h_ve: {:.2}", model.h_ve[0]); // Ventilation
    println!("h_tr_em: {:.2}", model.h_tr_em[0]); // Envelope-to-mass
    println!("h_tr_is: {:.2}", model.h_tr_is[0]); // Interior-to-surface
    println!("h_tr_ms: {:.2}", model.h_tr_ms[0]); // Mass-to-surface

    // For Case 195 with NO windows and NO infiltration:
    // h_ext should be h_tr_em only (no windows, no ventilation)
    // Current: h_tr_w + h_ve + h_tr_em = 57.71
    // But with zero windows: h_tr_w = 0, h_ve = 0

    let h_ext_no_ve = model.h_tr_em[0]; // Only envelope conductance
    println!("\nh_ext (h_tr_em only): {:.2}", h_ext_no_ve);

    // Now compute sensitivity with h_tr_ms = 0
    let h_tr_is = model.h_tr_is[0];
    let sens_zero_mass = h_tr_is / (h_tr_is * h_ext_no_ve);
    println!("\nWith h_tr_ms=0 and h_ext=h_tr_em:");
    println!("sensitivity = h_tr_is / (h_tr_is * h_ext)");
    println!("           = {} / ({} * {})", h_tr_is, h_tr_is, h_ext_no_ve);
    println!("           = {:.4}", 1.0 / h_ext_no_ve);

    // That's wrong too! The formula must be different for zero-mass
    // Let me reconsider...

    // For zero mass case, the network is simpler:
    // T_i = (h_tr_is * T_s + h_ext * T_ext) / (h_tr_is + h_ext)
    // sensitivity = h_tr_is / (h_tr_is + h_ext)

    let h_ext_corrected = model.h_tr_em[0]; // envelope only
    let sens_simple = h_tr_is / (h_tr_is + h_ext_corrected);
    println!("\nSimple formula (h_tr_is / (h_tr_is + h_ext)):");
    println!(
        "sensitivity = {} / ({} + {})",
        h_tr_is, h_tr_is, h_ext_corrected
    );
    println!("           = {:.4}", sens_simple);

    // Expected ~0.91 means h_ext should be ~65
    // Let's see what h_ext would give 0.91:
    // 0.91 = h_tr_is / (h_tr_is + h_ext)
    // 0.91 * (h_tr_is + h_ext) = h_tr_is
    // 0.91 * h_tr_is + 0.91 * h_ext = h_tr_is
    // 0.91 * h_ext = h_tr_is - 0.91 * h_tr_is = 0.09 * h_tr_is
    // h_ext = 0.09 * h_tr_is / 0.91 = 0.099 * h_tr_is = 0.099 * 592 = 58.6

    let target_sens = 0.91;
    let required_hext = h_tr_is * (1.0 - target_sens) / target_sens;
    println!(
        "\nFor sensitivity = 0.91, h_ext would need to be: {:.2}",
        required_hext
    );
}
