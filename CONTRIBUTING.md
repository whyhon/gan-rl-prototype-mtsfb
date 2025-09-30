# Contributing to GAN-RL Red Teaming Framework

We welcome contributions to the GAN-RL Red Teaming Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

This project is dedicated to providing a harassment-free experience for everyone. We expect all participants to abide by our Code of Conduct:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Issues

Before creating an issue, please:

1. **Search existing issues** to ensure your issue hasn't already been reported
2. **Use the issue template** if available
3. **Provide detailed information** including:
   - Operating system and version
   - Python version
   - Ollama version and model used
   - Steps to reproduce the issue
   - Expected vs actual behavior
   - Relevant log files or error messages

### Suggesting Enhancements

We welcome suggestions for new features or improvements:

1. **Check existing feature requests** to avoid duplicates
2. **Provide detailed description** of the enhancement
3. **Explain the use case** and how it benefits the project
4. **Consider implementation complexity** and potential impact

### Contributing Code

#### Prerequisites

- Python 3.8 or higher
- Ollama installed and running
- Familiarity with the framework architecture
- Understanding of AI safety and security principles

#### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/GAN-RL_prototype.git
   cd GAN-RL_prototype
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Making Changes

1. **Follow the existing code style**:
   - Use meaningful variable and function names
   - Add docstrings to functions and classes
   - Follow PEP 8 guidelines
   - Keep functions focused and modular

2. **Add tests** for new functionality:
   - Write unit tests for individual components
   - Add integration tests for framework interactions
   - Ensure tests cover edge cases

3. **Update documentation**:
   - Update README.md if adding new features
   - Add docstrings to new functions/classes
   - Update configuration examples if needed

4. **Test your changes**:
   - Run existing tests to ensure no regressions
   - Test with different models and configurations
   - Verify industry customization still works

#### Security Considerations

**IMPORTANT**: This framework is for defensive security research only.

- **Do not add malicious capabilities** that could be used for attacks
- **Ensure new features maintain ethical boundaries**
- **Test responsibly** on models you own or have permission to test
- **Document potential misuse** and include appropriate warnings

#### Pull Request Process

1. **Update version numbers** if applicable
2. **Update the README.md** with details of changes if needed
3. **Ensure all tests pass** and code follows style guidelines
4. **Create a clear pull request description**:
   - Describe what changes were made and why
   - Link to any related issues
   - Provide testing instructions
   - Include screenshots if UI changes are involved

5. **Wait for review** from maintainers
6. **Address feedback** and make requested changes
7. **Maintain commit history** with clear, descriptive commit messages

#### Commit Message Guidelines

Use clear, descriptive commit messages:

```
type(scope): brief description

Longer description if needed

- List any breaking changes
- Reference issues: Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Types of Contributions

### High Priority Areas

1. **Industry-specific configurations**:
   - New industry templates
   - Improved detection patterns
   - Better evaluation metrics

2. **Model compatibility**:
   - Support for new language models
   - Improved API integrations
   - Performance optimizations

3. **Detection improvements**:
   - New vulnerability patterns
   - Better evaluation heuristics
   - Reduced false positives

4. **Documentation**:
   - Usage examples
   - Tutorial content
   - API documentation

### Medium Priority Areas

1. **User interface improvements**:
   - Better command-line interface
   - Progress indicators
   - Error handling

2. **Testing and reliability**:
   - Unit test coverage
   - Integration tests
   - Performance benchmarks

3. **Internationalization**:
   - Additional language support
   - Cultural adaptation
   - Localized documentation

### Special Considerations

#### Industry Customization

When contributing industry-specific features:

- **Collaborate with domain experts** to ensure accuracy
- **Validate with real-world scenarios** when possible
- **Document regulatory compliance** considerations
- **Provide clear usage guidelines**

#### AI Safety and Ethics

All contributions must:

- **Prioritize defensive applications** over offensive capabilities
- **Include appropriate warnings** about potential misuse
- **Follow responsible disclosure** practices for vulnerabilities
- **Maintain focus on model improvement** rather than exploitation

## Recognition

Contributors will be recognized in:

- GitHub contributors list
- Release notes for significant contributions
- Project documentation for major features
- Academic publications when appropriate

## Getting Help

If you need help with contributing:

1. **Check the documentation** first
2. **Search existing issues** for similar questions
3. **Create a new issue** with the "question" label
4. **Join community discussions** if available

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

We appreciate all contributors who help make this framework better for the AI safety and security community. Every contribution, no matter how small, is valuable!