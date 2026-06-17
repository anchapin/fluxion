//! Ollama client for local LLM inference
//!
//! This module provides a client for interacting with Ollama, a local LLM inference server.
//! It supports chat completions and is designed for privacy-preserving, offline-capable inference.

use crate::validation::copilot::types::BemIssue;
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Ollama API client
#[allow(dead_code)]
pub struct OllamaClient {
    base_url: String,
    client: Client,
    timeout: Duration,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(base_url: String, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url,
            client,
            timeout,
        }
    }

    /// Check if Ollama server is available
    pub async fn is_available(&self) -> bool {
        match self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Generate a chat completion using Ollama
    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let request = ChatRequest {
            model: "llama3.2:latest".to_string(),
            messages,
            stream: false,
        };

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("Ollama API error: {}", response.status()));
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response.message.content)
    }

    /// Analyze BEM configuration using LLM
    pub async fn analyze(&self, config_json: &str, rule_issues: &[BemIssue]) -> Result<String> {
        let prompt =
            crate::validation::copilot::prompt::build_analysis_prompt(config_json, rule_issues);

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: crate::validation::copilot::prompt::SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];

        self.chat(messages).await
    }

    /// Generate troubleshooting recommendations using LLM
    pub async fn troubleshoot(&self, issue: &BemIssue, context: &str) -> Result<String> {
        let prompt =
            crate::validation::copilot::prompt::build_troubleshooting_prompt(issue, context);

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: crate::validation::copilot::prompt::SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];

        self.chat(messages).await
    }

    /// List available models on the Ollama server
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("Failed to list models: {}", response.status()));
        }

        let tags: ModelTags = response.json().await?;
        Ok(tags.models.iter().map(|m| m.name.clone()).collect())
    }
}

// ============================================================================
// Ollama API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ModelTags {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::copilot::OLLAMA_DEFAULT_URL;

    #[test]
    fn test_ollama_client_creation() {
        let client = OllamaClient::new(OLLAMA_DEFAULT_URL.to_string(), Duration::from_secs(30));
        assert_eq!(client.base_url, OLLAMA_DEFAULT_URL);
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));
    }
}
