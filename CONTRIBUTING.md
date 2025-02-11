# Contributing to Zonos API

We love your input! We want to make contributing to Zonos API as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Git Submodules

This project uses git submodules to manage the Zonos model repository. When working with the codebase:

1. Clone with submodules:
```bash
git clone --recursive https://github.com/manascb1344/zonos-api
```

2. If you forgot --recursive:
```bash
git submodule update --init --recursive
```

3. Update submodules to latest:
```bash
git submodule update --remote
```

4. When making changes that require a specific version of Zonos:
```bash
cd Zonos
git checkout <specific-commit>
cd ..
git add Zonos
git commit -m "chore: update Zonos submodule to <version>"
```

## Development Process

1. Clone the repository and set up your environment:
```bash
# Clone the repo with submodules
git clone --recursive https://github.com/manascb1344/zonos-api
cd zonos-api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
# Install dependencies
pip install -r requirements.txt

# Install Zonos from submodule
cd Zonos
pip install -e .
cd ..

# Install GPU optimizations
pip install --no-build-isolation -e .[compile]

2. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

3. Make your changes and run the application:
```bash
# Run the application locally
uvicorn app.main:app --reload
```

4. Update documentation:
- Update docstrings for any modified functions
- Update README.md if adding new features
- Update CHANGELOG.md following the format
- Update API documentation if endpoints change

5. Commit your changes:
```bash
git add .
git commit -m "feat: your detailed commit message"
```

We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `chore:` for maintenance tasks
- `refactor:` for code refactoring

## Docker Development

For development with Docker:

```bash
# Build and run with docker-compose
docker-compose up --build

# Check logs
docker-compose logs -f api
```

## Monitoring and Debugging

The application includes Prometheus metrics and Grafana dashboards:

1. Access metrics at: http://localhost:8000/metrics
2. View Prometheus: http://localhost:9090
3. Access Grafana: http://localhost:3000 (admin/admin)

## Pull Request Process

1. Fork the repo and create your branch from `main`
2. Update the documentation
3. Update the CHANGELOG.md
4. Issue that pull request!

## License
By contributing, you agree that your contributions will be licensed under its Apache 2.0 License. 