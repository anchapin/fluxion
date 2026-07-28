const fs = require('fs');

const outputFile = process.env.REGRESSION_OUTPUT_FILE || 'regression_output.txt';
const output = fs.existsSync(outputFile) ? fs.readFileSync(outputFile, 'utf8').slice(-4000) : '(output not found)';

const title = 'Nightly ASHRAE 140 Regression Test Failed';
const body = `## Nightly Regression Test Failure

The comprehensive ASHRAE 140 regression test failed in the nightly workflow.

### Test Output
\`\`\`
${output}
\`\`\`

### Actions Required
- Investigate the failing case(s)
- Check for recent changes that may have affected validation
- Run \`cargo test --test integration test_ashrae_140_comprehensive_regression --release\` locally
- Fix the regression or update baseline if appropriate

---
**Workflow:** ${process.env.GITHUB_WORKFLOW}
**Run:** ${process.env.GITHUB_RUN_NUMBER}
**Commit:** ${process.env.GITHUB_SHA}
**Timestamp:** ${process.env.GITHUB_EVENT_HEAD_COMMIT_TIMESTAMP}`;

async function main() {
  const github = require('@actions/github');
  const core = require('@actions/core');

  const client = github.getOctokit(process.env.GITHUB_TOKEN);

  const { data: issues } = await client.rest.issues.listForRepo({
    owner: context.repo.owner,
    repo: context.repo.repo,
    state: 'open',
    labels: 'regression,ashrae-140'
  });

  if (issues.length > 0) {
    console.log(`Found ${issues.length} existing regression issues, skipping issue creation`);
  } else {
    await client.rest.issues.create({
      owner: context.repo.owner,
      repo: context.repo.repo,
      title,
      body,
      labels: ['regression', 'ashrae-140']
    });
    console.log('Created regression issue');
  }
}

main().catch(err => {
  console.error('Error creating issue:', err.message);
  process.exit(0); // Non-critical
});
