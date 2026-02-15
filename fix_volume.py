with open('src/sim/engine.rs', 'r') as f:
    content = f.read()

# Add total_volume accumulator after line "let mut load_data = vec![0.0; num_zones];"
old = "let mut load_data = vec![0.0; num_zones];"
new = """let mut load_data = vec![0.0; num_zones];
        let mut total_volume = 0.0;"""

content = content.replace(old, new)

# Add total_volume += volume inside the loop, after "let volume = geo.volume();"
old2 = "let volume = geo.volume();\n            let wall_area"
new2 = "let volume = geo.volume();\n            total_volume += volume;\n            let wall_area"
content = content.replace(old2, new2)

# Change "vent.fan_capacity / volume" to "vent.fan_capacity / total_volume"
old3 = "let fan_ach = vent.fan_capacity / volume;"
new3 = "let fan_ach = vent.fan_capacity / total_volume;"
content = content.replace(old3, new3)

with open('src/sim/engine.rs', 'w') as f:
    f.write(content)

print("Done!")
