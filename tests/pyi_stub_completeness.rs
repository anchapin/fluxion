//! fluxion.pyi stub-completeness drift gate — Issue #2509.
//!
//! Asserts that every `#[pyclass]` registered in the primary Python module
//! (`#[pymodule] fn fluxion` in `src/lib.rs`) has a matching `class X` entry in
//! `fluxion.pyi`. Also checks the four exception types added via `m.add(...,
//! ...)` and the top-level free functions wired through `add_function` /
//! `wrap_pyfunction!` (including those added by the `multi_zone` submodule).
//!
//! This is a **self-contained text-based** check: it parses the raw Rust
//! source (independent of the `python-bindings` feature) and the `.pyi` text,
//! so it runs in default `cargo test` with no Python toolchain required. The
//! companion script `scripts/check_pyi_drift.py` implements the same logic for
//! CI/IDE consumption.
//!
//! If this test fails, either add the missing declaration to `fluxion.pyi` or
//! update the registration in `src/lib.rs` / `src/python/bindings.rs`.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Locate the body (text between the outer `{` and its matching `}`) of the
/// `#[pymodule] fn <name>` item in `src`, starting the search at `from`.
fn pymodule_body<'a>(src: &'a str, fn_name: &str) -> Option<&'a str> {
    let needle = "#[pymodule]".to_string();
    let mut search_from = 0usize;
    while let Some(rel) = src[search_from..].find(&needle) {
        let abs = search_from + rel;
        // The `fn <name>` must follow the attribute on the next non-blank token.
        let tail = &src[abs..];
        if let Some(fn_rel) = tail.find(&format!("fn {}", fn_name)) {
            let after = fn_rel + format!("fn {}", fn_name).len();
            let brace_rel = tail[after..].find('{')?;
            let brace_abs = abs + after + brace_rel;
            return balanced_block(src, brace_abs);
        }
        search_from = abs + needle.len();
    }
    None
}

/// Return the text inside the `{...}` block whose opening brace is at `brace_idx`.
fn balanced_block(src: &str, brace_idx: usize) -> Option<&str> {
    let bytes = src.as_bytes();
    if brace_idx >= bytes.len() || bytes[brace_idx] != b'{' {
        return None;
    }
    let mut depth = 0i32;
    for (i, &b) in bytes[brace_idx..].iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&src[brace_idx + 1..brace_idx + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Build a map of `rust struct/enum name -> python class name` by scanning the
/// binding sources for `#[pyclass(name = "...")]`. Falls back to stripping a
/// leading `Py` prefix when no explicit `name` is present.
fn pyclass_name_map(root: &Path) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    let sources = [
        "src/lib.rs",
        "src/api/parameters.rs",
        "src/python/bindings.rs",
        "src/python/hvac_bindings.rs",
        "src/python/model_bindings.rs",
        "src/python/multi_node_bindings.rs",
        "src/python/osm_bindings.rs",
    ];
    for rel in sources {
        let path = root.join(rel);
        let Ok(src) = fs::read_to_string(&path) else {
            continue;
        };
        for (idx, _) in src.match_indices("#[pyclass") {
            let attr_end = idx + "#[pyclass".len();
            // Attribute options run until the matching ']'.
            let opts_end = match src[attr_end..].find(']') {
                Some(e) => attr_end + e,
                None => continue,
            };
            let opts = &src[attr_end..opts_end];
            let explicit = extract_name_attr(opts);
            // Find the next `struct`/`enum` within the following 600 bytes.
            let tail = &src[opts_end..(opts_end + 600).min(src.len())];
            if let Some(rel) = tail.find("struct").or_else(|| tail.find("enum")) {
                let after_kw = &tail[rel..];
                if let Some(name) = next_ident(after_kw) {
                    let py = explicit.unwrap_or_else(|| {
                        name.strip_prefix("Py")
                            .map(|stripped| stripped.to_owned())
                            .unwrap_or_else(|| name.clone())
                    });
                    map.entry(name).or_insert(py);
                }
            }
        }
    }
    map
}

/// Extract the `"..."` value following `name` in a `#[pyclass(...)]` options
/// string, e.g. `name = "VectorField"` → `VectorField`. Returns `None` if no
/// such attribute is present.
fn extract_name_attr(opts: &str) -> Option<String> {
    let name_idx = opts.find("name")?;
    let after_name = &opts[name_idx + 4..];
    let open = after_name.find('"')?;
    let rest = &after_name[open + 1..];
    let close = rest.find('"')?;
    Some(rest[..close].to_owned())
}

/// Return the first Rust identifier in `s` (after any leading whitespace).
fn next_ident(s: &str) -> Option<String> {
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
    let mut id = String::new();
    for c in chars {
        if c.is_alphanumeric() || c == '_' {
            id.push(c);
        } else {
            break;
        }
    }
    if id.is_empty() {
        None
    } else {
        Some(id)
    }
}

/// Extract every `add_class::<RustName>` target from the primary fluxion body
/// AND the `multi_zone` submodule body (which is called from the fluxion body
/// via `python::multi_zone(_py, m)` and registers into the same module), each
/// resolved to its Python class name.
fn registered_classes(root: &Path) -> Vec<String> {
    let lib = fs::read_to_string(root.join("src/lib.rs")).expect("src/lib.rs readable");
    let name_map = pyclass_name_map(root);
    let mut out = Vec::new();
    let bodies: Vec<&str> = match pymodule_body(&lib, "fluxion") {
        Some(b) => vec![b],
        None => panic!("#[pymodule] fn fluxion not found in src/lib.rs"),
    };
    // The multi_zone submodule registers classes into the same top-level module.
    let mut mz_classes: Vec<String> = Vec::new();
    if let Ok(bindings) = fs::read_to_string(root.join("src/python/bindings.rs")) {
        if let Some(mz_body) = pymodule_body(&bindings, "multi_zone") {
            collect_add_classes(mz_body, &name_map, &mut mz_classes);
        }
    }
    for body in bodies {
        collect_add_classes(body, &name_map, &mut out);
    }
    out.extend(mz_classes);
    out
}

fn collect_add_classes(
    body: &str,
    name_map: &std::collections::HashMap<String, String>,
    out: &mut Vec<String>,
) {
    for m in body.match_indices("add_class::<") {
        let after = &body[m.0 + "add_class::<".len()..];
        let end = after.find('>').expect("add_class closes>");
        let rust_full = after[..end].trim();
        let rust = rust_full.rsplit("::").next().unwrap_or(rust_full);
        let py = name_map.get(rust).cloned().unwrap_or_else(|| {
            rust.strip_prefix("Py")
                .map(|stripped| stripped.to_owned())
                .unwrap_or_else(|| rust.to_owned())
        });
        out.push(py);
    }
}

/// Extract the four exception names added via `m.add("Name", ...)`.
fn registered_exceptions(root: &Path) -> Vec<String> {
    let lib = fs::read_to_string(root.join("src/lib.rs")).expect("src/lib.rs readable");
    let body = pymodule_body(&lib, "fluxion").expect("#[pymodule] fn fluxion found");
    let mut out = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("m.add(\"") {
            if let Some(end) = rest.find('"') {
                out.push(rest[..end].to_owned());
            }
        }
    }
    out
}

/// Extract registered free-function names: `wrap_pyfunction!(name, ...)` in the
/// primary fluxion body plus the `multi_zone` submodule body.
fn registered_functions(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let lib = fs::read_to_string(root.join("src/lib.rs")).expect("src/lib.rs readable");
    if let Some(body) = pymodule_body(&lib, "fluxion") {
        collect_wrap_pyfunctions(body, &mut out);
    }
    let bindings =
        fs::read_to_string(root.join("src/python/bindings.rs")).expect("bindings.rs readable");
    if let Some(mz_body) = pymodule_body(&bindings, "multi_zone") {
        collect_wrap_pyfunctions(mz_body, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

fn collect_wrap_pyfunctions(body: &str, out: &mut Vec<String>) {
    for m in body.match_indices("wrap_pyfunction!(") {
        let after = &body[m.0 + "wrap_pyfunction!(".len()..];
        // skip whitespace, then read an ident path; take the last segment.
        let id = match next_ident_path(after) {
            Some(p) => p,
            None => continue,
        };
        out.push(id.rsplit("::").next().unwrap_or(&id).to_owned());
    }
}

/// Read a `path::to::name` (ident segments joined by `::`), tolerating leading whitespace.
fn next_ident_path(s: &str) -> Option<String> {
    let trimmed = s.trim_start();
    let mut out = String::new();
    let mut chars = trimmed.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_alphanumeric() || c == '_' {
            out.push(c);
            chars.next();
        } else if c == ':' && out.ends_with(|c: char| c.is_alphanumeric() || c == '_') {
            // lookahead for '::'
            let mut clone = chars.clone();
            clone.next();
            if clone.peek() == Some(&':') {
                out.push_str("::");
                chars.next();
                chars.next();
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Parse `class Name` declarations at column 0 from the `.pyi`.
fn pyi_classes(pyi: &str) -> HashSet<String> {
    pyi.lines()
        .filter_map(|l| {
            let t = l.trim_start();
            // Only top-level (column-0) class declarations count.
            if !l.starts_with("class ") {
                return None;
            }
            t.strip_prefix("class ").and_then(next_ident)
        })
        .collect()
}

/// Parse top-level `def name` declarations from the `.pyi`.
fn pyi_functions(pyi: &str) -> HashSet<String> {
    pyi.lines()
        .filter_map(|l| {
            if !l.starts_with("def ") {
                return None;
            }
            l.strip_prefix("def ").and_then(next_ident)
        })
        .collect()
}

#[test]
fn fluxion_pyi_contains_every_registered_pyclass() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pyi = fs::read_to_string(root.join("fluxion.pyi")).expect("fluxion.pyi readable");

    let classes = registered_classes(&root);
    let pyi_cls = pyi_classes(&pyi);

    let missing: Vec<String> = classes
        .iter()
        .filter(|c| !pyi_cls.contains(*c))
        .cloned()
        .collect();

    assert!(
        missing.is_empty(),
        "fluxion.pyi drift detected (Issue #2509). The following classes are registered in \
         `#[pymodule] fn fluxion` (src/lib.rs) but have no `class X` entry in fluxion.pyi:\n  {}\n\
         Add the missing class stubs to fluxion.pyi, or update the registrations in src/lib.rs. \
         Run `python3 scripts/check_pyi_drift.py` for the same check outside cargo.",
        missing.join("\n  ")
    );
    println!(
        "pyi drift gate: all {} registered classes present in fluxion.pyi",
        classes.len()
    );
}

#[test]
fn fluxion_pyi_contains_every_registered_exception() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pyi = fs::read_to_string(root.join("fluxion.pyi")).expect("fluxion.pyi readable");
    let excs = registered_exceptions(&root);
    let pyi_cls = pyi_classes(&pyi);
    let missing: Vec<String> = excs
        .iter()
        .filter(|e| !pyi_cls.contains(*e))
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "fluxion.pyi is missing exception stubs: {}\nAdd them as `class X(Exception): ...` entries.",
        missing.join(", ")
    );
}

#[test]
fn fluxion_pyi_contains_every_registered_function() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pyi = fs::read_to_string(root.join("fluxion.pyi")).expect("fluxion.pyi readable");
    let funcs = registered_functions(&root);
    let pyi_fns = pyi_functions(&pyi);
    let missing: Vec<String> = funcs
        .iter()
        .filter(|f| !pyi_fns.contains(*f))
        .cloned()
        .collect();
    assert!(
        missing.is_empty(),
        "fluxion.pyi is missing top-level function stubs: {}\nAdd them as `def f(...): ...` entries.",
        missing.join(", ")
    );
}

#[test]
fn fluxion_pyi_parses_as_valid_python() {
    // Lightweight sanity gate: ensure the stub is syntactically parseable as
    // Python. We shell out to `python3` if available; if it is absent we
    // perform a brace/paren/quote balance check instead so the test never
    // spuriously fails on a Python-less CI runner.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pyi = fs::read_to_string(root.join("fluxion.pyi")).expect("fluxion.pyi readable");

    if let Ok(out) = std::process::Command::new("python3")
        .args([
            "-c",
            "import ast, sys; ast.parse(open(sys.argv[1]).read())",
            "fluxion.pyi",
        ])
        .current_dir(&root)
        .output()
    {
        assert!(
            out.status.success(),
            "fluxion.pyi is not valid Python:\n--- stderr ---\n{}\n---------------",
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }

    // Fallback: balance check on (), [], {}, and triple-quoted strings.
    balance_check(&pyi);
    // Ensure no tab/space-mixed indentation is obviously broken and that every
    // `class`/`def` line ends with `...` or a `->` annotation + colon pattern.
    for line in pyi.lines() {
        if (line.starts_with("def ") || line.starts_with("    def ")) && !line.contains("...") {
            assert!(
                line.ends_with(':'),
                "stub method line missing `...`/colon: {line}"
            );
        }
    }
}

fn balance_check(src: &str) {
    let mut paren = 0i32;
    let mut brack = 0i32;
    let mut brace = 0i32;
    let mut in_triple = false;
    let triple = "\"\"\"";
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !in_triple {
            if src[i..].starts_with(triple) {
                in_triple = true;
                i += 3;
                continue;
            }
            match bytes[i] {
                b'(' => paren += 1,
                b')' => paren -= 1,
                b'[' => brack += 1,
                b']' => brack -= 1,
                b'{' => brace += 1,
                b'}' => brace -= 1,
                _ => {}
            }
        } else if src[i..].starts_with(triple) {
            in_triple = false;
            i += 3;
            continue;
        }
        i += 1;
    }
    assert_eq!(paren, 0, "unbalanced () in fluxion.pyi");
    assert_eq!(brack, 0, "unbalanced [] in fluxion.pyi");
    assert_eq!(brace, 0, "unbalanced {{}} in fluxion.pyi");
    assert!(!in_triple, "unclosed triple-quoted string in fluxion.pyi");
}
