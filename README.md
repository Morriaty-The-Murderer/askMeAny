# NL2SQL: Natural Language to SQL Query System

AI-powered data inspection and query generation tool that transforms natural language questions into SQL queries, supporting multiple LLM providers.

## Overview

NL2SQL is an advanced system that enables users to interact with databases using natural language. It combines modern LLM capabilities with database querying to provide an intuitive interface for data analysis and exploration.

Key Features:
- Natural language to SQL query conversion
- Support for OpenAI, Claude and Gemini models
- Automatic provider fallback handling
- Interactive data visualization interface
- Administrative dashboard for system monitoring
- Automated data collection and processing pipeline

## Project Structure

The project is organized into the following modules:

- `crawler/`: Web crawler for collecting SQL-related training data
  - Data extraction components
  - Data cleaning utilities
  - Storage management

- `dashboard/`: Admin interface for system management
  - Data visualization tools
  - Model performance monitoring
  - System configuration interface

- `frontend/`: User-facing web interface
  - Query input interface
  - Results visualization
  - Interactive components

- `common/`: Shared utilities and core components
  - Configuration management
  - ML model interfaces
  - Common data models

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Virtual environment tool (e.g., virtualenv, conda)
- Git
- OpenAI API Key

## Supported Language Models

NL2SQL supports multiple Language Model providers for query generation, each with unique strengths:

### OpenAI
- Models:
  * GPT-4 (default): Best for complex queries and schema understanding
  * GPT-4-turbo: Faster response times with similar quality
  * GPT-3.5-turbo: Cost-effective for simpler queries
- Features:
  * Robust SQL generation
  * Advanced schema comprehension
  * High accuracy for complex queries
- Setup:
  1. Get API key from [OpenAI Platform](https://platform.openai.com/)
  2. Set in .env: `OPENAI_API_KEY=sk-...`
  3. Optional: Configure organization ID: `OPENAI_ORG_ID=org-...`
- Limitations:
  * Token limit: 128k (GPT-4), 16k (GPT-3.5)
  * Rate limits vary by tier
  * Higher latency for GPT-4

### Claude (Anthropic)
- Models:
  * Claude-2.1 (recommended): Latest model with improved SQL capabilities
  * Claude-2: Stable model for production use
  * Claude-instant: Fast, cost-effective option
- Features:
  * Excellent at explaining query logic
  * Handles complex schema relationships
  * Good error correction
- Setup:
  1. Get API key from [Anthropic Console](https://console.anthropic.com/)
  2. Set in .env: `ANTHROPIC_API_KEY=sk-ant-...`
  3. Optional: Set token limit: `ANTHROPIC_MAX_TOKENS=100000`
- Limitations:
  * May require special access for production
  * Limited tool integration
  * Regional availability restrictions

### Google Gemini
- Models:
  * Gemini Pro: Primary model for SQL generation
  * Gemini Pro Vision: Supports visual query contexts
  * Text-bison: Legacy model for basic queries
- Features:
  * Strong multilingual support
  * Good performance on code generation
  * Integration with Google Cloud
- Setup:
  1. Get API key from [Google AI Studio](https://makersuite.google.com/)
  2. Set in .env: `GOOGLE_API_KEY=AIza...`
  3. Optional: Set project ID: `GOOGLE_PROJECT_ID=your-project-id`
- Limitations:
  * Regional availability varies
  * Limited historical query context
  * New platform, evolving features

### Module Setup

1. Crawler Module:
```
cd crawler
pip install -r requirements.txt
```

2. Dashboard Module:
```
cd dashboard
pip install -r requirements.txt
```

3. Frontend Module:
```
cd frontend
pip install -r requirements.txt
```

4. Common Utilities:
```
cd common
pip install -r requirements.txt
```

## Dependency Management

Each module maintains its own requirements.txt file for specific dependencies:

- crawler/requirements.txt: Web scraping and data processing
- dashboard/requirements.txt: Admin interface components
- frontend/requirements.txt: User interface elements
- common/requirements.txt: Core system dependencies

To install all dependencies:
```
pip install -r requirements.txt
```

## Configuration Setup

1. Environment Configuration:
   - Copy .env.example to .env
   - Configure OpenAI API credentials
   - Set database connection parameters
   - Configure model paths and parameters

   ## Model Configuration

   ### Provider Selection Guidelines

   1. Primary Use Case:
      * Complex Database Queries → OpenAI GPT-4
      * Detailed Query Explanations → Claude-2.1
      * Multilingual Support → Gemini Pro
      * Cost-Effective Solutions → GPT-3.5-turbo/Claude-instant

   2. Performance Considerations:
      * Response Time: GPT-3.5-turbo > Claude-instant > Gemini Pro > GPT-4
      * Accuracy: GPT-4 > Claude-2.1 > Gemini Pro > GPT-3.5-turbo
      * Cost Efficiency: Claude-instant > GPT-3.5-turbo > Gemini Pro > GPT-4

   3. Special Requirements:
      * Long Contexts (>32k tokens) → Claude-2.1
      * Visual Query Support → Gemini Pro Vision
      * Regulatory Compliance → Choose based on data residency
      * High Throughput → Setup provider pools and fallbacks

   ### Configuration Best Practices

   1. Provider Setup:
      ```yaml
      # config.yaml
      default_provider: "openai"  # Primary provider
      fallback_providers: ["anthropic", "google"]  # Backup options
      ```

   2. Model Selection:
      ```yaml
      openai:
        models:
          default: "gpt-4"  # Primary model
          alternatives: ["gpt-4-turbo", "gpt-3.5-turbo"]  # Fallbacks
      ```

   3. Performance Tuning:
      ```yaml
      parameters:
        max_tokens: 2048  # Adjust based on query complexity
        temperature: 0.7  # Lower for more deterministic output
        timeout: 30  # Adjust based on model response time
      ```
   1. Primary Provider Setup
      - Set API key in .env file
      - Configure model parameters in config.yaml
      - Adjust request timeouts and retries

   2. Fallback Providers (Optional)
      - Configure backup providers
      - Set automatic failover rules
      - Define model equivalence mappings

   3. Model Parameters
      - Adjust temperature (0.0-1.0)
      - Set token limits appropriate for provider
      - Configure response formats
      - Customize system prompts

   4. Performance Optimization
      - Enable response caching
      - Configure request batching
      - Set up connection pooling

3. Database Configuration:
   - Configure database connection strings
   - Set up access credentials
   - Initialize database schema

## Development Workflow

1. Code Development
   - Create feature branch from main
   - Implement changes following module structure
   - Add tests for new functionality
   - Update documentation

## Troubleshooting

### Common Issues

1. API Connection Issues
   - "Invalid API key"
     * Verify correct provider key in .env
     * Check key format matches provider requirements
     * Ensure key has required permissions/quotas
   - "Rate limit exceeded"
     * Implement request batching
     * Check usage quotas
     * Consider upgrading API tier
   - "Model not available"
     * Verify model availability in your region
     * Check if model requires special access
     * Consider fallback models

2. Environment Setup
   - Error: "Module not found"
     * Verify all requirements are installed
     * Check virtual environment is activated
     * Ensure PYTHONPATH includes project root

3. Database Connection
   - Error: "Could not connect to database"
     * Verify database credentials in .env
     * Check database is running and accessible
     * Confirm network/firewall settings

4. Model-Specific Issues
   - OpenAI
     * Token limit exceeded: Adjust max_tokens parameter
     * High latency: Use gpt-3.5-turbo for faster responses
   - Claude
     * Context window errors: Split long inputs
     * Tool use limitations: Check model capabilities
   - Gemini
     * Regional restrictions: Use appropriate endpoints
     * Version compatibility: Update API version

### Getting Help

- Check the logs in `nl2sql.log`
- Review error messages in console output
- Submit an issue on GitHub with:
  * Error message and stack trace
  * System configuration details
  * Steps to reproduce the issue

2. Testing
   - Run unit tests for modified components
   - Perform integration testing
   - Validate against test datasets

3. Deployment
   - Review changes and documentation
   - Merge to main branch
   - Deploy to staging environment
   - Perform final validation
   - Deploy to production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License

## Contact

NL2SQL Team - Contact details here