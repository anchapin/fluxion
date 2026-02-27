use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    // Full suite of ASHRAE 140 test cases
    // Issue #323: Implement ASHRAE 140 CI Pipeline
    let cases = vec![
        // Low mass cases (600 series)
        ("Case 600", ASHRAE140Case::Case600),
        ("Case 610", ASHRAE140Case::Case610),
        ("Case 620", ASHRAE140Case::Case620),
        ("Case 630", ASHRAE140Case::Case630),
        ("Case 640", ASHRAE140Case::Case640),
        ("Case 650", ASHRAE140Case::Case650),
        ("Case 600FF", ASHRAE140Case::Case600FF),
        ("Case 650FF", ASHRAE140Case::Case650FF),
        
        // High mass cases (900 series)
        ("Case 900", ASHRAE140Case::Case900),
        ("Case 910", ASHRAE140Case::Case910),
        ("Case 920", ASHRAE140Case::Case920),
        ("Case 930", ASHRAE140Case::Case930),
        ("Case 940", ASHRAE140Case::Case940),
        ("Case 950", ASHRAE140Case::Case950),
        ("Case 900FF", ASHRAE140Case::Case900FF),
        ("Case 950FF", ASHRAE140Case::Case950FF),
        
        // Special cases
        ("Case 960", ASHRAE140Case::Case960),
        ("Case 195", ASHRAE140Case::Case195),
    ];

    println!("ASHRAE 140 Test Cases (Full Suite)\n");
    println!("========================\n");
    
    for (name, case) in cases {
        let spec = case.spec();
        let case_type = if spec.num_zones > 1 {
            "Multi-zone"
        } else if spec.is_free_floating() {
            "Free-floating"
        } else {
            "Controlled"
        };
        
        println!("{}: {} [{}]", name, spec.case_id, case_type);
    }
    
    println!("\nTotal: {} test cases", cases.len());
}
