import os
import json
import requests

def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not found")
        return

    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("GITHUB_EVENT_NUMBER")
    
    if not pr_number:
        # Try to get from event payload if not in env
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path:
            with open(event_path, 'r') as f:
                event_data = json.load(f)
                pr_number = event_data.get("pull_request", {}).get("number")

    if not pr_number:
        print("PR number not found")
        return

    # Read validation report
    report_content = "### ASHRAE 140 Validation Results\n\n"
    if os.path.exists("validation_report.md"):
        with open("validation_report.md", 'r') as f:
            report_content += f.read()
    else:
        report_content += "⚠️ Validation report not found."

    # Post comment to GitHub
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "body": report_content
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Comment posted successfully")
    else:
        print(f"Failed to post comment: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
