use anyhow::{Result, anyhow};
use serde::Serialize;
use serde_json;
use std::path::Path;

/// Time series dataset for plotting.
#[derive(Debug, Clone, Serialize)]
pub struct TimeSeriesData {
    pub timestamps: Vec<usize>,
    pub datasets: Vec<Dataset>,
}

/// Panel assignment for animation subplots.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum PlotPanel {
    Temperature,
    HVAC,
    Solar,
}

/// A single dataset (line or area) in the time series.
#[derive(Debug, Clone, Serialize)]
pub struct Dataset {
    pub label: String,
    pub values: Vec<f64>,
    pub color: Option<String>,
    #[serde(skip)]
    pub panel: Option<PlotPanel>,
}

fn dataset_to_trace(dataset: &Dataset, timestamps: &[usize]) -> serde_json::Value {
    serde_json::json!({
        "x": timestamps,
        "y": dataset.values,
        "type": "scatter",
        "mode": "lines",
        "name": dataset.label,
        "line": if let Some(color) = &dataset.color {
            serde_json::json!({"color": color})
        } else {
            serde_json::json!(null)
        },
    })
}

/// Generate a standalone HTML file with Plotly.js chart.
///
/// The HTML includes:
/// - Plotly.js from CDN
/// - Responsive layout
/// - Zoom/pan toolbar
/// - Export to PNG/SVG buttons
pub fn generate_html(
    data: &TimeSeriesData,
    output: &Path,
) -> Result<()> {
    let json_data = serde_json::json!({
        "data": data.datasets.iter().map(|d| {
            serde_json::json!({
                "x": data.timestamps,
                "y": d.values,
                "type": "scatter",
                "mode": "lines",
                "name": d.label,
                "line": if let Some(color) = &d.color { serde_json::json!({"color": color}) } else { serde_json::json!(null) },
            })
        }).collect::<Vec<_>>(),
        "layout": {
            "title": "Time Series Visualization",
            "xaxis": {"title": "Hour"},
            "yaxis": {"title": "Value"},
            "hovermode": "closest",
            "dragmode": "zoom",
        },
        "config": {
            "modeBarButtonsToAdd": ["toImage"],
            "displayModeBar": true,
            "responsive": true,
        }
    });

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fluxion Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #chart {{ width: 100%; height: 100vh; }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <script>
        const data = {};
        const layout = data.layout;
        const config = data.config;
        Plotly.newPlot('chart', data.data, layout, config);
    </script>
</body>
</html>"#,
        serde_json::to_string_pretty(&json_data)?
    );

    std::fs::write(output, html)?;
    Ok(())
}

/// Generate an animated HTML chart with play/pause, speed control, and scrubber.
///
/// This creates a two-panel chart:
/// - Upper: temperature series (one or more zones)
/// - Lower: HVAC heating/cooling and solar gains
pub fn generate_animation(
    data: &TimeSeriesData,
    output: &Path,
) -> Result<()> {
    // Split datasets based on panel assignment
    let temp_datasets: Vec<&Dataset> = data
        .datasets
        .iter()
        .filter(|d| d.panel == Some(PlotPanel::Temperature))
        .collect();
    let mut bottom_datasets: Vec<&Dataset> = data
        .datasets
        .iter()
        .filter(|d| d.panel == Some(PlotPanel::HVAC) || d.panel == Some(PlotPanel::Solar))
        .collect();

    // Ensure we have at least some data
    if temp_datasets.is_empty() && bottom_datasets.is_empty() {
        return Err(anyhow!("No datasets provided for animation (set panel on datasets)"));
    }

    // Convert to Plotly traces
    let temp_traces: Vec<serde_json::Value> = temp_datasets
        .iter()
        .map(|d| dataset_to_trace(d, &data.timestamps))
        .collect();
    let bottom_traces: Vec<serde_json::Value> = bottom_datasets
        .iter()
        .map(|d| dataset_to_trace(d, &data.timestamps))
        .collect();

    let temp_traces_json = serde_json::to_string(&temp_traces)?;
    let bottom_traces_json = serde_json::to_string(&bottom_traces)?;

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fluxion Animated Visualization</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ margin: 0; padding: 0; font-family: sans-serif; }}
    #controls {{ padding: 10px; background: #f0f0f0; text-align: center; }}
    #tempChart, #bottomChart {{ width: 100%; }}
    #tempChart {{ height: 45vh; }}
    #bottomChart {{ height: 25vh; }}
    #scrubber {{ width: 80vw; max-width: 800px; }}
  </style>
</head>
<body>
  <div id="controls">
    <button id="playBtn">Play</button>
    <button id="pauseBtn">Pause</button>
    Speed: <input type="number" id="speedInput" value="1" min="0.1" max="100" step="0.1"> hours/sec
    Hour: <input type="range" id="scrubber" min="0" max="8759" value="0">
    <span id="hourDisplay">0</span> / 8759
  </div>
  <div id="tempChart"></div>
  <div id="bottomChart"></div>
  <script>
    const tempTraces = {0};
    const bottomTraces = {1};
    window.onload = function() {{
        const tempLayout = {{
            title: 'Zone Temperatures',
            xaxis: {{ title: 'Hour', range: [0, 100] }},
            yaxis: {{ title: 'Temperature (°C)' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }}
        }};
        const bottomLayout = {{
            title: 'HVAC & Solar',
            xaxis: {{ title: 'Hour', range: [0, 100] }},
            yaxis: {{ title: 'Power (W)' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }}
        }};
        const config = {{
            modeBarButtonsToAdd: ['toImage'],
            displayModeBar: true,
            responsive: true
        }};
        Plotly.newPlot('tempChart', tempTraces, tempLayout, config);
        Plotly.newPlot('bottomChart', bottomTraces, bottomLayout, config);

        let currentHour = 0;
        let speed = 1.0;
        let interval = null;
        const maxHour = 8759;
        const scrubber = document.getElementById('scrubber');
        const hourDisplay = document.getElementById('hourDisplay');

        function renderFrame(hour) {{
            currentHour = hour;
            if (currentHour > maxHour) currentHour = maxHour;
            Plotly.relayout('tempChart', {{'xaxis.range': [0, currentHour]}});
            Plotly.relayout('bottomChart', {{'xaxis.range': [0, currentHour]}});
            scrubber.value = currentHour;
            hourDisplay.textContent = currentHour;
        }}

        function play() {{
            if (interval) clearInterval(interval);
            interval = setInterval(() => {{
                let next = currentHour + speed;
                if (next > maxHour) {{
                    next = 0;
                }}
                renderFrame(next);
            }}, 1000 / speed);
        }}

        function pause() {{
            if (interval) {{
                clearInterval(interval);
                interval = null;
            }}
        }}

        document.getElementById('playBtn').addEventListener('click', play);
        document.getElementById('pauseBtn').addEventListener('click', pause);
        scrubber.addEventListener('input', (e) => {{
            renderFrame(parseInt(e.target.value));
        }});
        document.getElementById('speedInput').addEventListener('change', (e) => {{
            speed = parseFloat(e.target.value);
        }});
    }};
  </script>
</body>
</html>"#,
        temp_traces_json, bottom_traces_json
    );

    std::fs::write(output, html)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_html_generation() {
        let mut data = TimeSeriesData {
            timestamps: (0..8760).step_by(10).collect(),
            datasets: vec![
                Dataset {
                    label: "Temp Zone 1".to_string(),
                    values: vec![20.0, 21.0, 22.0],
                    color: None,
                    panel: None,
                },
                Dataset {
                    label: "Temp Zone 2".to_string(),
                    values: vec![19.0, 20.5, 21.5],
                    color: Some("#FF0000".to_string()),
                    panel: None,
                },
            ],
        };
        let temp_dir = env::temp_dir();
        let out_path = temp_dir.join("test_viz.html");
        generate_html(&data, &out_path).unwrap();
        assert!(out_path.exists());
        let content = std::fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("plotly"));
        assert!(content.contains("Temp Zone 1"));
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn test_animation_html() {
        use super::PlotPanel;
        let data = TimeSeriesData {
            timestamps: (0..8760).collect(),
            datasets: vec![
                Dataset {
                    label: "Temperature".to_string(),
                    values: vec![20.0; 8760],
                    color: None,
                    panel: Some(PlotPanel::Temperature),
                },
                Dataset {
                    label: "Heating".to_string(),
                    values: vec![0.0; 8760],
                    color: None,
                    panel: Some(PlotPanel::HVAC),
                },
            ],
        };
        let temp_dir = env::temp_dir();
        let out_path = temp_dir.join("test_anim.html");
        generate_animation(&data, &out_path).unwrap();
        assert!(out_path.exists());
        let content = std::fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("Play"));
        assert!(content.contains("Pause"));
        assert!(content.contains("type=\"range\""));
        assert!(content.contains("speedInput"));
        assert!(content.contains("function play()"));
        assert!(content.contains("function pause()"));
        assert!(content.contains("function renderFrame("));
        // Verify chart divs
        assert!(content.contains("tempChart"));
        assert!(content.contains("bottomChart"));
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn test_export_buttons() {
        let data = TimeSeriesData {
            timestamps: (0..8760).step_by(10).collect(),
            datasets: vec![Dataset {
                label: "Temp".to_string(),
                values: vec![20.0, 21.0, 22.0],
                color: None,
                panel: None,
            }],
        };
        let temp_dir = env::temp_dir();
        let out_path = temp_dir.join("test_export.html");
        generate_html(&data, &out_path).unwrap();
        let content = std::fs::read_to_string(&out_path).unwrap();
        assert!(content.contains("toImage"));
        let _ = std::fs::remove_file(&out_path);
    }
}
