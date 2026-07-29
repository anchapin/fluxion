use fluxion::validation::ashrae_140_cases::ASHRAE140Case;

fn main() {
    let cases = vec![
        ("Case 640", ASHRAE140Case::Case640),
        ("Case 650", ASHRAE140Case::Case650),
        ("Case 940", ASHRAE140Case::Case940),
        ("Case 950", ASHRAE140Case::Case950),
    ];

    for (name, case) in cases {
        let spec = case.spec();
        println!("{}: {} - free_floating={}", name, spec.case_id, spec.is_free_floating());
    }
}
