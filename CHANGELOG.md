# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-02-10

### Added
- Git submodule support for Zonos repository
- Automatic submodule updates in Docker builds
- Documentation for submodule management

### Changed
- Updated Docker build process to use submodules
- Improved installation instructions for submodule handling
- Reorganized dependency installation process

### Security
- Improved version control with git submodules
- Better dependency tracking and updates

## [1.0.0] - 2025-02-11 - Initial Release (Unstable)

### Added
- Initial development setup
- Basic FastAPI structure
- Docker configuration
- CI/CD pipeline
- Initial release of Zonos API
- Support for both Transformer and Hybrid model variants
- FastAPI implementation with comprehensive API endpoints
- Docker and docker-compose deployment with NVIDIA GPU support
- Prometheus and Grafana monitoring integration
- Voice cloning capabilities
- Audio continuation support
- Fine-grained emotion control
- Health checks and logging
- CORS support
- Swagger documentation
- Production-ready configurations
- GPU optimizations with flash-attention and mamba-ssm
- Comprehensive documentation
