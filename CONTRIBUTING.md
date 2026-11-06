# Contributing to FlowPredictor-Corrector

Thank you for your interest in contributing to FlowPredictor-Corrector! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](../../issues) section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, PyTorch version)

### Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Add tests if applicable
5. Update documentation if needed
6. Commit with clear messages
7. Push to your fork and submit a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular

### Commit Messages

Use clear and meaningful commit messages:
- `feat: Add new feature`
- `fix: Fix bug in...`
- `docs: Update documentation`
- `refactor: Refactor code for...`

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/FlowPredictor-Corrector.git
cd FlowPredictor-Corrector

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest
```

## Questions?

Feel free to open an issue for any questions or discussions.
