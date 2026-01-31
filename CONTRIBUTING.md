# Nexus AI - Contribution Guidelines

Thank you for your interest in contributing to Nexus Conversational AI! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account
- Basic understanding of NLP/ML concepts (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/nexus-conversational-ai.git
cd nexus-conversational-ai
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/original/nexus-conversational-ai.git
```

## Development Setup

### Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest tests/ -v

# Run linting
ruff check src/

# Check formatting
black --check src/
```

## Making Changes

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or improving tests
- `ci/` - CI/CD changes

### Commit Messages

Follow conventional commits:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `ci`, `chore`

Examples:
```
feat(nlu): add support for custom entity types
fix(api): handle empty message requests
docs(readme): update installation instructions
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for all public functions
- Use docstrings for all public modules, classes, and functions

### Formatting

```bash
# Format code
black src/ tests/

# Sort imports
ruff check --fix src/ tests/
```

### Type Checking

```bash
mypy src/
```

### Documentation Style

```python
def process_message(
    self,
    text: str,
    session_id: UUID | None = None,
) -> Response:
    """Process an incoming message and generate a response.

    Args:
        text: The user's input message.
        session_id: Optional session identifier for context tracking.

    Returns:
        Response object containing the generated reply and metadata.

    Raises:
        ValueError: If text is empty or exceeds maximum length.

    Example:
        >>> response = await engine.process_message("Hello!")
        >>> print(response.text)
        "Hello! How can I help you today?"
    """
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_nlu.py -v

# Specific test
pytest tests/test_nlu.py::TestIntentClassifier -v

# With coverage
pytest tests/ --cov=src/nexus --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures from `conftest.py`

```python
import pytest
from nexus.nlu.classifier import IntentClassifier

class TestIntentClassifier:
    @pytest.fixture
    def classifier(self, mock_config):
        return IntentClassifier(mock_config)

    @pytest.mark.asyncio
    async def test_classify_greeting(self, classifier):
        """Test that greetings are classified correctly."""
        result = await classifier.classify("hello")
        assert result.name == "greeting"
        assert result.confidence > 0.8
```

### Test Categories

- Unit tests: Test individual components
- Integration tests: Test component interactions
- E2E tests: Test full conversation flows

## Submitting Changes

### Before Submitting

1. ✅ All tests pass: `pytest tests/ -v`
2. ✅ Code is formatted: `black --check src/ tests/`
3. ✅ Linting passes: `ruff check src/ tests/`
4. ✅ Type checking passes: `mypy src/`
5. ✅ Documentation is updated
6. ✅ Commit messages follow convention

### Create Pull Request

1. Push your branch:
```bash
git push origin feature/your-feature-name
```

2. Create a Pull Request on GitHub

3. Fill out the PR template:
   - Describe your changes
   - Link related issues
   - List any breaking changes
   - Include screenshots if applicable

### PR Title Format

```
type(scope): short description
```

Example: `feat(api): add WebSocket heartbeat mechanism`

## Review Process

### What Reviewers Look For

- Code quality and readability
- Test coverage
- Documentation
- Performance implications
- Security considerations
- Backward compatibility

### Responding to Feedback

- Be open to suggestions
- Ask for clarification if needed
- Make requested changes promptly
- Mark conversations as resolved when addressed

### After Approval

- Your PR will be merged by a maintainer
- Delete your feature branch
- Celebrate your contribution! 🎉

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Join our community chat

Thank you for contributing to Nexus AI! 🚀
