// validation/automation/github/mod.rs
/// GitHub-specific automation module
///
/// This module provides GitHub Actions integration and workflow automation
/// for cross-validation testing and CI/CD pipelines.

pub mod workflow;

/// GitHub workflow configuration
#[derive(Debug, Clone)]
pub struct GitHubWorkflow {
    /// Workflow name
    pub name: String,
    /// Workflow description
    pub description: String,
    /// Trigger events
    pub triggers: Vec<String>,
    /// Environment variables
    pub env_vars: std::collections::HashMap<String, String>,
    /// Jobs configuration
    pub jobs: std::collections::HashMap<String, GitHubJob>,
}

impl GitHubWorkflow {
    /// Create a new GitHub workflow
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            triggers: vec!["push".to_string(), "pull_request".to_string()],
            env_vars: std::collections::HashMap::new(),
            jobs: std::collections::HashMap::new(),
        }
    }

    /// Add a trigger event
    pub fn add_trigger(&mut self, trigger: &str) -> &mut Self {
        self.triggers.push(trigger.to_string());
        self
    }

    /// Add an environment variable
    pub fn add_env_var(&mut self, key: &str, value: &str) -> &mut Self {
        self.env_vars.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a job to the workflow
    pub fn add_job(&mut self, name: &str, job: GitHubJob) -> &mut Self {
        self.jobs.insert(name.to_string(), job);
        self
    }
}

/// GitHub job configuration
#[derive(Debug, Clone)]
pub struct GitHubJob {
    /// Job name
    pub name: String,
    /// Runs on environment
    pub runs_on: String,
    /// Job steps
    pub steps: Vec<GitHubStep>,
    /// Job dependencies
    pub needs: Vec<String>,
}

impl GitHubJob {
    /// Create a new GitHub job
    pub fn new(name: &str, runs_on: &str) -> Self {
        Self {
            name: name.to_string(),
            runs_on: runs_on.to_string(),
            steps: Vec::new(),
            needs: Vec::new(),
        }
    }

    /// Add a step to the job
    pub fn add_step(&mut self, step: GitHubStep) -> &mut Self {
        self.steps.push(step);
        self
    }

    /// Add a job dependency
    pub fn add_dependency(&mut self, job_name: &str) -> &mut Self {
        self.needs.push(job_name.to_string());
        self
    }
}

/// GitHub workflow step
#[derive(Debug, Clone)]
pub struct GitHubStep {
    /// Step name
    pub name: String,
    /// Step uses (action or command)
    pub uses: Option<String>,
    /// Step run command
    pub run: Option<String>,
    /// Environment variables for this step
    pub env: std::collections::HashMap<String, String>,
}

impl GitHubStep {
    /// Create a new step with a command
    pub fn with_command(name: &str, command: &str) -> Self {
        Self {
            name: name.to_string(),
            uses: None,
            run: Some(command.to_string()),
            env: std::collections::HashMap::new(),
        }
    }

    /// Create a new step with an action
    pub fn with_action(name: &str, action: &str) -> Self {
        Self {
            name: name.to_string(),
            uses: Some(action.to_string()),
            run: None,
            env: std::collections::HashMap::new(),
        }
    }

    /// Add an environment variable to the step
    pub fn add_env(&mut self, key: &str, value: &str) -> &mut Self {
        self.env.insert(key.to_string(), value.to_string());
        self
    }
}

/// GitHub API interaction helpers
pub mod api {
    use reqwest::blocking::Client;
    use serde_json::Value;
    use std::error::Error;

    /// GitHub API client
    #[derive(Debug)]
    pub struct GitHubClient {
        client: Client,
        base_url: String,
        token: Option<String>,
    }

    impl GitHubClient {
        /// Create a new GitHub client
        pub fn new(token: Option<String>) -> Self {
            Self {
                client: Client::new(),
                base_url: "https://api.github.com".to_string(),
                token,
            }
        }

        /// Make a GET request to GitHub API
        pub fn get(&self, endpoint: &str) -> Result<Value, Box<dyn Error>> {
            let url = format!("{}{}", self.base_url, endpoint);
            let mut request = self.client.get(&url);

            if let Some(token) = &self.token {
                request = request.header("Authorization", format!("token {}", token));
            }

            let response = request.send()?;
            if !response.status().is_success() {
                return Err(format!("GitHub API request failed: {}", response.status()).into());
            }

            let body = response.text()?;
            let json: Value = serde_json::from_str(&body)?;
            Ok(json)
        }

        /// Make a POST request to GitHub API
        pub fn post(&self, endpoint: &str, data: &Value) -> Result<Value, Box<dyn Error>> {
            let url = format!("{}{}", self.base_url, endpoint);
            let mut request = self.client.post(&url);

            if let Some(token) = &self.token {
                request = request.header("Authorization", format!("token {}", token));
            }

            let response = request.json(data).send()?;
            if !response.status().is_success() {
                return Err(format!("GitHub API request failed: {}", response.status()).into());
            }

            let body = response.text()?;
            let json: Value = serde_json::from_str(&body)?;
            Ok(json)
        }
    }
}

/// Error handling for GitHub operations
#[derive(Debug, thiserror::Error)]
pub enum GitHubError {
    #[error("GitHub API error: {0}")]
    ApiError(String),
    #[error("Workflow validation error: {0}")]
    ValidationError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type for GitHub operations
pub type GitHubResult<T> = Result<T, GitHubError>;

/// Validate workflow configuration
pub fn validate_workflow(workflow: &GitHubWorkflow) -> GitHubResult<()> {
    if workflow.name.is_empty() {
        return Err(GitHubError::ValidationError("Workflow name cannot be empty".to_string()));
    }

    if workflow.jobs.is_empty() {
        return Err(GitHubError::ValidationError("Workflow must have at least one job".to_string()));
    }

    for (job_name, job) in &workflow.jobs {
        if job.steps.is_empty() {
            return Err(GitHubError::ValidationError(
                format!("Job '{}' must have at least one step", job_name),
            ));
        }
    }

    Ok(())
}

/// Generate workflow YAML content
pub fn generate_workflow_yaml(workflow: &GitHubWorkflow) -> GitHubResult<String> {
    let mut yaml = String::new();
    yaml.push_str(&format!("name: {}\n", workflow.name));
    yaml.push_str(&format!("description: {}\n", workflow.description));
    yaml.push_str("on:\n");

    for trigger in &workflow.triggers {
        yaml.push_str(&format!("  {}\n", trigger));
    }

    if !workflow.env_vars.is_empty() {
        yaml.push_str("env:\n");
        for (key, value) in &workflow.env_vars {
            yaml.push_str(&format!("  {}: {}\n", key, value));
        }
    }

    yaml.push_str("jobs:\n");
    for (job_name, job) in &workflow.jobs {
        yaml.push_str(&format!("  {}:\n", job_name));
        yaml.push_str(&format!("    name: {}\n", job.name));
        yaml.push_str(&format!("    runs-on: {}\n", job.runs_on));

        if !job.needs.is_empty() {
            yaml.push_str("    needs: [");
            for (i, need) in job.needs.iter().enumerate() {
                if i > 0 {
                    yaml.push_str(", ");
                }
                yaml.push_str(need);
            }
            yaml.push_str("]\n");
        }

        yaml.push_str("    steps:\n");
        for step in &job.steps {
            yaml.push_str("      - name: ");
            yaml.push_str(&step.name);
            yaml.push_str("\n");

            if let Some(uses) = &step.uses {
                yaml.push_str(&format!("        uses: {}\n", uses));
            }

            if let Some(run) = &step.run {
                yaml.push_str(&format!("        run: {}\n", run));
            }

            if !step.env.is_empty() {
                yaml.push_str("        env:\n");
                for (key, value) in &step.env {
                    yaml.push_str(&format!("          {}: {}\n", key, value));
                }
            }
        }
    }

    Ok(yaml)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let mut workflow = GitHubWorkflow::new("Test Workflow", "Test description");
        workflow.add_trigger("schedule");
        workflow.add_env_var("RUST_VERSION", "1.70.0");

        assert_eq!(workflow.name, "Test Workflow");
        assert_eq!(workflow.description, "Test description");
        assert!(workflow.triggers.contains(&"push".to_string()));
        assert!(workflow.triggers.contains(&"pull_request".to_string()));
        assert!(workflow.triggers.contains(&"schedule".to_string()));
        assert_eq!(workflow.env_vars.get("RUST_VERSION"), Some(&"1.70.0".to_string()));
    }

    #[test]
    fn test_job_creation() {
        let mut job = GitHubJob::new("Test Job", "ubuntu-latest");
        job.add_step(GitHubStep::with_command("Install dependencies", "cargo install --path ."));
        job.add_step(GitHubStep::with_action("Checkout", "actions/checkout@v4"));

        assert_eq!(job.name, "Test Job");
        assert_eq!(job.runs_on, "ubuntu-latest");
        assert_eq!(job.steps.len(), 2);
    }

    #[test]
    fn test_workflow_validation() {
        let mut workflow = GitHubWorkflow::new("Valid Workflow", "Valid description");
        let mut job = GitHubJob::new("Test Job", "ubuntu-latest");
        job.add_step(GitHubStep::with_command("Test", "echo 'test'"));
        workflow.add_job("test", job);

        assert!(validate_workflow(&workflow).is_ok());

        let empty_workflow = GitHubWorkflow::new("", "");
        assert!(validate_workflow(&empty_workflow).is_err());
    }

    #[test]
    fn test_yaml_generation() {
        let mut workflow = GitHubWorkflow::new("Test Workflow", "Test description");
        workflow.add_env_var("RUST_VERSION", "1.70.0");

        let mut job = GitHubJob::new("Build", "ubuntu-latest");
        job.add_step(GitHubStep::with_action("Checkout", "actions/checkout@v4"));
        job.add_step(GitHubStep::with_command("Build", "cargo build --release"));
        workflow.add_job("build", job);

        let yaml = generate_workflow_yaml(&workflow).unwrap();
        assert!(yaml.contains("name: Test Workflow"));
        assert!(yaml.contains("description: Test description"));
        assert!(yaml.contains("RUST_VERSION: 1.70.0"));
        assert!(yaml.contains("actions/checkout@v4"));
        assert!(yaml.contains("cargo build --release"));
    }
}
