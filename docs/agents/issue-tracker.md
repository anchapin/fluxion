# Issue tracker — GitHub

**Repo:** `anchapin/fluxion`
**CLI:** `gh`

## Workflow

- Issues are tracked in the repo's GitHub Issues.
- Skills call `gh issue create --title "..." --body "..." --label "..."` to create issues.
- Skills call `gh issue edit <number> --label "..."` to update labels.
- Skills call `gh issue list --label "..."` to query issues.

## Creating issues

```bash
gh issue create --title "<title>" --body "<body>" --label "<label>"
```

## Updating labels

```bash
gh issue edit <number> --label "<label>"
```

## Querying

```bash
gh issue list --label "needs-triage"
gh issue list --label "ready-for-agent"
```
