use std::fs;
use std::path::Path;

fn count_lines<P: AsRef<Path>>(p: P) -> usize {
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(p) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                total += count_lines(path);
            } else if let Some(ext) = path.extension() {
                if ext == "rs" {
                    if let Ok(s) = fs::read_to_string(&path) {
                        total += s.lines().count();
                    }
                }
            }
        }
    }
    total
}

fn main() {
    let sim = count_lines("src/sim");
    let physics = count_lines("src/physics");
    println!("Lines in src/sim: {}", sim);
    println!("Lines in src/physics: {}", physics);
}
