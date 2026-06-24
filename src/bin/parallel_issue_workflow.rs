//! Parallel Issue Workflow Tool
//!
//! Automates GitHub issue triage and PR creation by identifying the top 5
//! highest-priority open issues and working on them concurrently.
//!
//! Usage:
//!   cargo run --bin parallel-issue-workflow -- --repo owner/repo

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::process::Command;
use tokio::sync::mpsc;

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "parallel-issue-workflow")]
#[command(about = "Automates GitHub issue triage and PR creation in parallel", long_about = None)]
struct Args {
    /// Repository in format 'owner/repo'
    #[arg(short, long)]
    repo: String,

    /// Maximum number of issues to fetch
    #[arg(short, long, default_value = "50")]
    limit: usize,

    /// Maximum number of issues to process in parallel
    #[arg(short, long, default_value = "5")]
    max_parallel: usize,

    /// Comma-separated list of priority labels (highest priority first)
    #[arg(short, long, default_value = "critical,high-priority,urgent,priority")]
    priority_labels: String,

    /// Dry run mode - don't create PRs, just show what would be done
    #[arg(short, long, default_value = "false")]
    dry_run: bool,
}

// ============================================================================
// GitHub Issue Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubLabel {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubComment {
    #[serde(flatten)]
    _extra: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubIssue {
    pub number: u64,
    pub title: String,
    pub body: Option<String>,
    #[serde(default)]
    pub labels: Vec<GitHubLabel>,
    pub milestone: Option<GitHubMilestone>,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub comments: Vec<GitHubComment>,
    #[serde(default)]
    pub reactions: Reactions,
    #[serde(default)]
    pub state: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubMilestone {
    pub title: String,
    pub due_on: Option<String>,
    pub number: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionGroup {
    pub content: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Reactions {
    #[serde(rename = "reactionGroups")]
    pub reaction_groups: Vec<ReactionGroup>,
}

impl Reactions {
    pub fn thumbs_up(&self) -> u64 {
        self.reaction_groups
            .iter()
            .find(|r| r.content == "THUMBS_UP")
            .map(|r| r.count)
            .unwrap_or(0)
    }

    pub fn thumbs_down(&self) -> u64 {
        self.reaction_groups
            .iter()
            .find(|r| r.content == "THUMBS_DOWN")
            .map(|r| r.count)
            .unwrap_or(0)
    }

    pub fn eyes(&self) -> u64 {
        self.reaction_groups
            .iter()
            .find(|r| r.content == "EYES")
            .map(|r| r.count)
            .unwrap_or(0)
    }

    pub fn rocket(&self) -> u64 {
        self.reaction_groups
            .iter()
            .find(|r| r.content == "ROCKET")
            .map(|r| r.count)
            .unwrap_or(0)
    }

    pub fn hooray(&self) -> u64 {
        self.reaction_groups
            .iter()
            .find(|r| r.content == "HOORAY")
            .map(|r| r.count)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubUser {
    pub login: String,
    pub id: u64,
}

// ============================================================================
// Issue Ranking
// ============================================================================

#[derive(Debug, Clone)]
pub struct RankedIssue {
    pub issue: GitHubIssue,
    pub score: f64,
    pub priority_label: Option<String>,
}

impl RankedIssue {
    fn calculate_score(issue: &GitHubIssue, priority_labels: &[String]) -> (f64, Option<String>) {
        let mut score = 0.0;
        let mut matched_priority_label: Option<String> = None;

        // 1. Priority labels (highest weight)
        for label in &issue.labels {
            let label_lower = label.name.to_lowercase();
            for (idx, priority_label) in priority_labels.iter().enumerate() {
                if label_lower.contains(&priority_label.to_lowercase()) {
                    let weight = 1000.0 - (idx as f64 * 100.0); // First match gets highest weight
                    score += weight;
                    if matched_priority_label.is_none() {
                        matched_priority_label = Some(label.name.clone());
                    }
                }
            }
        }

        // 2. Milestone presence (due date proximity bonus)
        if let Some(ref milestone) = issue.milestone {
            score += 50.0;
            if let Some(ref due_on) = milestone.due_on {
                if let Ok(due_date) = chrono::NaiveDate::parse_from_str(&due_on[..10], "%Y-%m-%d") {
                    let today = chrono::Utc::now().date_naive();
                    let days_until_due = (due_date - today).num_days();
                    if days_until_due > 0 && days_until_due <= 7 {
                        score += 100.0; // Due soon bonus
                    } else if days_until_due < 0 {
                        score -= 50.0; // Overdue penalty
                    }
                }
            }
        }

        // 3. Community signal (reactions + comments)
        score += (issue.reactions.thumbs_up() as f64) * 5.0;
        score += (issue.reactions.rocket() as f64) * 10.0;
        score += (issue.reactions.hooray() as f64) * 8.0;
        score += (issue.comments.len() as f64) * 3.0;

        // 4. Issue age (older unresolved issues weighted higher)
        if !issue.created_at.is_empty() {
            if let Ok(created) =
                chrono::NaiveDate::parse_from_str(&issue.created_at[..10], "%Y-%m-%d")
            {
                let today = chrono::Utc::now().date_naive();
                let age_days = (today - created).num_days();
                score += (age_days as f64).sqrt() * 0.5; // Diminishing returns on age
            }
        }

        (score, matched_priority_label)
    }
}

// ============================================================================
// GitHub API Interaction
// ============================================================================

fn gh_exec(args: &[&str]) -> Result<String> {
    let output = Command::new("gh")
        .args(args)
        .output()
        .context("Failed to execute gh CLI")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("gh CLI error: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn git_exec(args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .context("Failed to execute git CLI")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git CLI error: {}", stderr);
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn fetch_issues(repo: &str, limit: usize) -> Result<Vec<GitHubIssue>> {
    println!(
        "Fetching issues from {}/{}...",
        repo.split('/').next().unwrap_or(""),
        repo.split('/').nth(1).unwrap_or("")
    );

    let output = gh_exec(&[
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        &limit.to_string(),
        "--json",
        "number,title,body,labels,milestone,createdAt,updatedAt,comments,reactionGroups,state,url",
    ])?;

    let issues: Vec<GitHubIssue> = serde_json::from_str(&output).map_err(|e| {
        anyhow::anyhow!(
            "JSON parse error at line {}, column {}: {}",
            e.line(),
            e.column(),
            e
        )
    })?;

    println!("Fetched {} open issues", issues.len());
    Ok(issues)
}

#[allow(dead_code)]
fn get_issue_details(repo: &str, issue_number: u64) -> Result<GitHubIssue> {
    let output = gh_exec(&[
        "issue",
        "view",
        &issue_number.to_string(),
        "--repo",
        repo,
        "--json",
        "number,title,body,labels,milestone,createdAt,updatedAt,comments,reactionGroups,state,url",
    ])?;

    let issue: GitHubIssue =
        serde_json::from_str(&output).context("Failed to parse gh issue view JSON")?;

    Ok(issue)
}

// ============================================================================
// Branch & PR Creation
// ============================================================================

#[allow(dead_code)]
fn create_branch_for_issue(issue: &GitHubIssue) -> Result<String> {
    let branch_name = format!(
        "fix/{}-{}",
        issue.number,
        issue
            .title
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
            .take(50)
            .collect::<String>()
    );

    println!("Creating branch: {}", branch_name);

    // Create branch
    git_exec(&["checkout", "-b", &branch_name])?;

    Ok(branch_name)
}

#[allow(dead_code)]
fn commit_and_push(issue: &GitHubIssue, branch_name: &str) -> Result<()> {
    // Stage all changes
    git_exec(&["add", "-A"])?;

    // Check if there are changes to commit
    let status_output = git_exec(&["status", "--porcelain"])?;
    if status_output.trim().is_empty() {
        println!("No changes to commit for issue #{}", issue.number);
        return Ok(());
    }

    // Commit with conventional format
    let commit_msg = format!(
        "fix: {} (#{})\n\nCloses #{}\n",
        issue.title, issue.number, issue.number
    );

    git_exec(&["commit", "-m", &commit_msg])?;
    println!("Committed changes for issue #{}", issue.number);

    // Push branch
    git_exec(&["push", "-u", "origin", &branch_name])?;

    Ok(())
}

/// Creates a placeholder fix file documenting the issue and proposed fix
fn create_fix_for_issue(issue: &GitHubIssue) -> Result<()> {
    let fix_content = format!(
        r#"# Fix for Issue #{number}: {title}

## Issue Summary
{note}

## Problem Description
{body}

## Proposed Solution
_TODO: Implement the actual fix here_

## Labels
{labels}

## Milestone
{milestone}

## Metadata
- Issue: #{number}
- URL: {url}
- Created: {created}

---
_This file was auto-generated by parallel-issue-workflow_
"#,
        number = issue.number,
        title = issue.title,
        note = "This is a placeholder fix file - implement actual fix before merging",
        body = issue.body.as_deref().unwrap_or("No description provided."),
        labels = issue
            .labels
            .iter()
            .map(|l| format!("- {}", l.name))
            .collect::<Vec<_>>()
            .join("\n"),
        milestone = issue
            .milestone
            .as_ref()
            .map(|m| m.title.as_str())
            .unwrap_or("None"),
        url = issue.url,
        created = issue.created_at,
    );

    let filename = format!("fixes/issue-{}.md", issue.number);
    std::fs::create_dir_all("fixes")?;
    std::fs::write(&filename, fix_content)?;

    println!("  Created fix file: {}", filename);
    Ok(())
}

#[allow(dead_code)]
fn create_pr(repo: &str, issue: &GitHubIssue, branch_name: &str) -> Result<String> {
    let output = gh_exec(&[
        "pr",
        "create",
        "--repo",
        repo,
        "--title",
        &format!("fix: {} (#{})", issue.title, issue.number),
        "--body",
        &format!(
            "## Summary\n{}\n\n## Closes\nCloses #{}\n\n---\nAuto-generated by parallel-issue-workflow",
            issue.body.as_deref().unwrap_or("No description provided."),
            issue.number
        ),
        "--base",
        "main",
        "--head",
        &branch_name,
        "--fill",
    ])?;

    Ok(output.trim().to_string())
}

// ============================================================================
// Issue Processing
// ============================================================================

#[derive(Debug)]
pub struct IssueResult {
    pub issue_number: u64,
    pub status: IssueStatus,
    pub pr_url: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum IssueStatus {
    Completed,
    Failed,
    Skipped,
    DryRun,
}

fn process_single_issue(
    repo: &str,
    issue: GitHubIssue,
    branch_name: String,
    dry_run: bool,
) -> IssueResult {
    println!("\nProcessing issue #{}: {}", issue.number, issue.title);

    if dry_run {
        println!("  [DRY RUN] Would create branch: {}", branch_name);
        println!("  [DRY RUN] Would implement fix for: {}", issue.title);
        println!("  [DRY RUN] Would create PR referencing #{}", issue.number);

        return IssueResult {
            issue_number: issue.number,
            status: IssueStatus::DryRun,
            pr_url: None,
            error: None,
        };
    }

    // Actual implementation mode
    println!("  Creating branch: {}", branch_name);

    // Step 1: Create branch
    if let Err(e) = create_branch_for_issue(&issue) {
        return IssueResult {
            issue_number: issue.number,
            status: IssueStatus::Failed,
            pr_url: None,
            error: Some(format!("Failed to create branch: {}", e)),
        };
    }

    // Step 2: Create fix placeholder
    println!("  Creating fix placeholder...");
    if let Err(e) = create_fix_for_issue(&issue) {
        return IssueResult {
            issue_number: issue.number,
            status: IssueStatus::Failed,
            pr_url: None,
            error: Some(format!("Failed to create fix: {}", e)),
        };
    }

    // Step 3: Commit and push changes
    println!("  Committing and pushing changes...");
    if let Err(e) = commit_and_push(&issue, &branch_name) {
        return IssueResult {
            issue_number: issue.number,
            status: IssueStatus::Failed,
            pr_url: None,
            error: Some(format!("Failed to commit/push: {}", e)),
        };
    }

    // Step 4: Create PR
    println!("  Creating pull request...");
    match create_pr(repo, &issue, &branch_name) {
        Ok(pr_url) => {
            println!("  ✅ PR created: {}", pr_url);
            IssueResult {
                issue_number: issue.number,
                status: IssueStatus::Completed,
                pr_url: Some(pr_url),
                error: None,
            }
        }
        Err(e) => IssueResult {
            issue_number: issue.number,
            status: IssueStatus::Failed,
            pr_url: None,
            error: Some(format!("Failed to create PR: {}", e)),
        },
    }
}

// ============================================================================
// Main Workflow
// ============================================================================

fn rank_issues(issues: Vec<GitHubIssue>, priority_labels: &[String]) -> Vec<RankedIssue> {
    issues
        .into_iter()
        .map(|issue| {
            let (score, priority_label) = RankedIssue::calculate_score(&issue, priority_labels);
            RankedIssue {
                issue,
                score,
                priority_label,
            }
        })
        .collect::<Vec<_>>()
}

async fn run_workflow(args: Args) -> Result<()> {
    println!("\n");
    println!("========================================");
    println!("PARALLEL ISSUE WORKFLOW");
    println!("========================================\n");
    println!("Repository: {}", args.repo);
    println!("Max parallel issues: {}", args.max_parallel);
    println!("Priority labels: {}", args.priority_labels);
    println!("Dry run: {}\n", args.dry_run);

    // Check gh auth
    println!("Checking gh CLI authentication...");
    if let Err(e) = gh_exec(&["auth", "status"]) {
        anyhow::bail!(
            "gh CLI not authenticated: {}. Run 'gh auth login' first.",
            e
        );
    }
    println!("gh CLI authenticated.\n");

    // Parse priority labels
    let priority_labels: Vec<String> = args
        .priority_labels
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Fetch issues
    let issues = fetch_issues(&args.repo, args.limit)?;

    if issues.is_empty() {
        println!("No open issues found. Exiting.");
        return Ok(());
    }

    // Rank issues
    println!("\nRanking {} issues by priority...", issues.len());
    let mut ranked_issues = rank_issues(issues, &priority_labels);
    ranked_issues.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // Select top N
    let top_issues: Vec<_> = ranked_issues.into_iter().take(args.max_parallel).collect();

    println!("\nTop {} Issues Selected:", top_issues.len());
    println!("----------------------------------------");
    for (idx, ranked) in top_issues.iter().enumerate() {
        println!(
            "{}. #{} [{}] - {} (score: {:.1})",
            idx + 1,
            ranked.issue.number,
            ranked.priority_label.as_deref().unwrap_or("no-priority"),
            ranked.issue.title,
            ranked.score
        );
        println!(
            "   Comments: {}, Reactions: {}",
            ranked.issue.comments.len(),
            ranked.issue.reactions.thumbs_up()
                + ranked.issue.reactions.rocket()
                + ranked.issue.reactions.hooray()
        );
    }
    println!();

    // Process issues in parallel
    println!("Spawning {} parallel threads...", top_issues.len());

    let (tx, mut rx) = mpsc::channel::<IssueResult>(top_issues.len());

    // Process each issue
    for ranked in top_issues {
        let tx = tx.clone();
        let repo = args.repo.clone();
        let issue = ranked.issue.clone();
        let dry_run = args.dry_run;
        let branch_name = format!(
            "fix/{}-{}",
            issue.number,
            issue
                .title
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .take(50)
                .collect::<String>()
        );

        tokio::spawn(async move {
            let result = process_single_issue(&repo, issue, branch_name, dry_run);
            let _ = tx.send(result).await;
        });
    }

    drop(tx);

    // Collect results
    let mut results: Vec<IssueResult> = Vec::new();
    while let Some(result) = rx.recv().await {
        results.push(result);
    }

    // Sort by issue number for consistent output
    results.sort_by_key(|r| r.issue_number);

    // Print summary
    println!("\n========================================");
    println!("WORKFLOW COMPLETED");
    println!("========================================\n");

    println!("Results:");
    for result in &results {
        let status_icon = match result.status {
            IssueStatus::Completed => "✅",
            IssueStatus::Failed => "❌",
            IssueStatus::Skipped => "⏭️",
            IssueStatus::DryRun => "🔄",
        };
        let error_msg = result.error.as_deref().unwrap_or("");
        let pr_info = if let Some(ref url) = result.pr_url {
            format!("PR: {}", url)
        } else {
            String::new()
        };
        println!(
            "  {} #{} - {}{}",
            status_icon, result.issue_number, pr_info, error_msg
        );
    }

    if args.dry_run {
        println!("\n[DRY RUN] No PRs were actually created.");
        println!("Run without --dry-run to create actual PRs.");
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Run the async workflow
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    if let Err(e) = runtime.block_on(run_workflow(args)) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ranked_issue_score_calculation() {
        let issue = GitHubIssue {
            number: 1,
            title: "Test issue".to_string(),
            body: Some("Description".to_string()),
            labels: vec![
                GitHubLabel {
                    name: "critical".to_string(),
                },
                GitHubLabel {
                    name: "bug".to_string(),
                },
            ],
            milestone: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-15T00:00:00Z".to_string(),
            comments: vec![],
            reactions: Reactions {
                reaction_groups: vec![
                    ReactionGroup {
                        content: "THUMBS_UP".to_string(),
                        count: 10,
                    },
                    ReactionGroup {
                        content: "THUMBS_DOWN".to_string(),
                        count: 0,
                    },
                    ReactionGroup {
                        content: "EYES".to_string(),
                        count: 2,
                    },
                    ReactionGroup {
                        content: "ROCKET".to_string(),
                        count: 3,
                    },
                    ReactionGroup {
                        content: "HOORAY".to_string(),
                        count: 1,
                    },
                ],
            },
            state: "open".to_string(),
            url: "https://github.com/owner/repo/issues/1".to_string(),
        };

        let priority_labels = vec!["critical".to_string(), "high-priority".to_string()];
        let (score, priority_label) = RankedIssue::calculate_score(&issue, &priority_labels);

        assert!(score > 0.0);
        assert_eq!(priority_label, Some("critical".to_string()));
    }

    #[test]
    fn test_branch_name_generation() {
        let issue = GitHubIssue {
            number: 123,
            title: "Fix: Database connection pooling issue!".to_string(),
            body: None,
            labels: vec![],
            milestone: None,
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            comments: vec![],
            reactions: Reactions {
                reaction_groups: vec![
                    ReactionGroup {
                        content: "THUMBS_UP".to_string(),
                        count: 0,
                    },
                    ReactionGroup {
                        content: "THUMBS_DOWN".to_string(),
                        count: 0,
                    },
                    ReactionGroup {
                        content: "EYES".to_string(),
                        count: 0,
                    },
                    ReactionGroup {
                        content: "ROCKET".to_string(),
                        count: 0,
                    },
                    ReactionGroup {
                        content: "HOORAY".to_string(),
                        count: 0,
                    },
                ],
            },
            state: "open".to_string(),
            url: "".to_string(),
        };

        let branch_name = format!(
            "fix/{}-{}",
            issue.number,
            issue
                .title
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .take(50)
                .collect::<String>()
        );

        assert_eq!(branch_name, "fix/123-fixdatabaseconnectionpoolingissue");
    }
}
