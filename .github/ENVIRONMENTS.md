# GitHub Environments Configuration

This project uses GitHub Environments for deployment workflows.

## Required Environments

The following environments should be configured in your GitHub repository:
**Settings > Environments**

### 1. staging

- **Purpose**: Staging/development deployments
- **Branch**: `develop`
- **Required Secrets**: See secrets list below

### 2. production

- **Purpose**: Production deployments
- **Branch**: `main`
- **Required Secrets**: See secrets list below
- **Protection Rules**: Require approval before deployment

## Required Repository Secrets

Configure these in **Settings > Secrets and variables > Actions**

### AWS Configuration

- `AWS_ROLE_ARN` - IAM role for staging deployments
- `AWS_PROD_ROLE_ARN` - IAM role for production deployments
- `AWS_REGION` - AWS region (e.g., us-west-2)
- `EKS_CLUSTER_NAME` - Staging EKS cluster name
- `EKS_PROD_CLUSTER_NAME` - Production EKS cluster name

### Database & Cache

- `POSTGRES_PASSWORD` - PostgreSQL password
- `REDIS_PASSWORD` - Redis password

### API Keys

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOOGLE_API_KEY` - Google API key
- `MISTRAL_API_KEY` - Mistral API key
- `GROQ_API_KEY` - Groq API key

### Security

- `JWT_SECRET_KEY` - JWT signing secret
- `ENCRYPTION_KEY` - Data encryption key

### Testing

- `SMOKE_TEST_API_KEY` - API key for smoke tests
- `PERF_TEST_API_KEY` - API key for performance tests

### Notifications

- `SLACK_WEBHOOK_URL` - Slack webhook for deployment notifications

### Auto-Available

- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

## Setting Up Environments

1. Go to your repository on GitHub
2. Click **Settings** tab
3. Click **Environments** in the sidebar
4. Click **New environment**
5. Create environments named: `staging` and `production`
6. Configure protection rules as needed
7. Add environment-specific secrets if required

## Troubleshooting

If you see "Value 'staging' is not valid" in VS Code:

- This is a VS Code validation warning, not a real error
- The workflows will work correctly if environments are configured in GitHub
- You can ignore these warnings or disable YAML validation in VS Code settings
