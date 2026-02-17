// Quick debug test to see actual values
use fluxion::validation::ashrae_140_cases::ASHRAE140Case;
use fluxion::validation::ASHRAE140Validator;

fn main() {
    let mut validator = ASHRAE140Validator::new();
    let report = validator.validate_analytical_engine();
    
    println!("Case | Metric | Fluxion | Ref Min | Ref Max | Status");
    println!("-----|--------|---------|---------|---------|--------");
    
    for result in &report.results {
        println!("{} | {:?} | {:.2} | {:.2} | {:.2} | {:?}", 
            result.case_id, 
            result.metric, 
            result.fluxion_value, 
            result.ref_min, 
            result.ref_max,
            result.status
        );
    }
    
    println!("\n=== Summary ===");
    println!("Pass Rate: {:.1}%", report.pass_rate());
    println!("MAE: {:.2}%", report.mae());
}
