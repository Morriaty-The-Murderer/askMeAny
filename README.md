# NL2SQL: Natural Language to SQL Query System

AI-powered data inspection and query generation tool that transforms natural language questions into SQL queries.

## Overview

NL2SQL is an advanced system that enables users to interact with databases using natural language. It combines modern NLP techniques with database querying to provide an intuitive interface for data analysis and exploration.

Key Features:
- Natural language to SQL query conversion
- Interactive data visualization interface
- Administrative dashboard for system monitoring
- Support for both OpenAI API and local transformer models
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

2. Model Configuration:
   - Set up OpenAI API access
   - Configure local transformer models
   - Adjust model parameters in config.py

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