# Maintenance Guide

## System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **CPU**: 4 cores (8+ cores recommended)
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Recommended Requirements
- **RAM**: 32GB
- **Storage**: 50GB free space
- **CPU**: 8+ cores
- **GPU**: NVIDIA GPU with 10GB+ VRAM (optional)

## Regular Maintenance Tasks

### Daily
- [ ] Check system logs for errors
- [ ] Monitor disk space usage
- [ ] Verify all services are running
- [ ] Review analysis results for anomalies

### Weekly
- [ ] Update Docker images
- [ ] Clean up temporary files
- [ ] Backup configuration files
- [ ] Review security logs

### Monthly
- [ ] Update system dependencies
- [ ] Review and rotate logs
- [ ] Performance optimization
- [ ] Security audit

## Monitoring

### Health Checks

#### API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "services": {
    "api": "ok",
    "dmr": "ok"
  }
}
```

#### DMR Health
```bash
curl http://localhost:11434/health
```

#### UI Health
```bash
curl http://localhost:5173
```

### Log Monitoring

#### API Logs
```bash
docker logs rocky_api
```

#### DMR Logs
```bash
docker logs rocky_dmr
```

#### UI Logs
```bash
docker logs rocky_ui
```

### Resource Monitoring

#### Memory Usage
```bash
docker stats rocky_api rocky_dmr rocky_ui
```

#### Disk Usage
```bash
docker system df
```

## Troubleshooting

### Common Issues

#### Service Won't Start
1. Check Docker is running
2. Verify ports are available
3. Check system resources
4. Review error logs

#### Out of Memory
1. Increase system RAM
2. Reduce model size in config
3. Limit concurrent analyses
4. Restart services

#### Slow Performance
1. Check CPU usage
2. Monitor memory usage
3. Review disk I/O
4. Optimize Docker settings

#### Analysis Failures
1. Check data format
2. Verify required libraries
3. Review error messages
4. Test with sample data

### Debug Mode

#### Enable Debug Logging
```yaml
# docker-compose.yml
services:
  api:
    environment:
      - LOG_LEVEL=DEBUG
```

#### Verbose Output
```bash
docker-compose up --build --verbose
```

## Backup and Recovery

### Backup Strategy

#### Configuration Files
```bash
# Backup configuration
tar -czf rocky-config-$(date +%Y%m%d).tar.gz \
  services/dmr/config.yaml \
  docker-compose.yml \
  apps/api/requirements.txt
```

#### Data Files
```bash
# Backup datasets
tar -czf rocky-data-$(date +%Y%m%d).tar.gz \
  data/
```

#### Docker Images
```bash
# Save Docker images
docker save rocky_api:latest | gzip > rocky-api-$(date +%Y%m%d).tar.gz
docker save rocky_dmr:latest | gzip > rocky-dmr-$(date +%Y%m%d).tar.gz
```

### Recovery Procedures

#### Restore Configuration
```bash
# Restore configuration
tar -xzf rocky-config-YYYYMMDD.tar.gz
docker-compose up -d
```

#### Restore Data
```bash
# Restore data
tar -xzf rocky-data-YYYYMMDD.tar.gz
```

#### Restore Images
```bash
# Restore Docker images
gunzip -c rocky-api-YYYYMMDD.tar.gz | docker load
gunzip -c rocky-dmr-YYYYMMDD.tar.gz | docker load
```

## Updates and Upgrades

### Updating Rocky AI

#### Check Current Version
```bash
docker-compose exec api python -c "import app; print(app.__version__)"
```

#### Update Process
1. **Backup current installation**
2. **Pull latest changes**
   ```bash
   git pull origin main
   ```
3. **Update Docker images**
   ```bash
   docker-compose pull
   ```
4. **Rebuild services**
   ```bash
   docker-compose up --build -d
   ```
5. **Verify functionality**
   ```bash
   curl http://localhost:8000/health
   ```

#### Rollback Procedure
1. **Stop services**
   ```bash
   docker-compose down
   ```
2. **Restore previous version**
   ```bash
   git checkout previous-version-tag
   ```
3. **Restart services**
   ```bash
   docker-compose up -d
   ```

### Dependency Updates

#### Python Dependencies
```bash
# Update API dependencies
cd apps/api
pip install --upgrade -r requirements.txt
```

#### Node.js Dependencies
```bash
# Update UI dependencies
cd apps/ui
npm update
```

#### Docker Base Images
```bash
# Update base images
docker-compose pull
docker-compose build --no-cache
```

## Performance Optimization

### Docker Optimization

#### Resource Limits
```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

#### Volume Optimization
```yaml
# Use named volumes for better performance
volumes:
  rocky_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /path/to/data
```

### System Optimization

#### Memory Settings
```bash
# Increase Docker memory limit
# Edit Docker Desktop settings or daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

#### Network Optimization
```yaml
# Use host networking for better performance
services:
  api:
    network_mode: "host"
```

## Security Maintenance

### Regular Security Tasks

#### Update Dependencies
```bash
# Check for security vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image rocky_api:latest
```

#### Review Access Logs
```bash
# Check API access logs
docker logs rocky_api | grep -E "(ERROR|WARN|401|403)"
```

#### Rotate Secrets
```bash
# Rotate API keys and tokens
# Update in environment variables
docker-compose down
# Update secrets
docker-compose up -d
```

### Security Monitoring

#### Failed Login Attempts
```bash
# Monitor failed authentication
docker logs rocky_api | grep -i "failed\|unauthorized"
```

#### Suspicious Activity
```bash
# Check for unusual patterns
docker logs rocky_api | grep -E "(injection|malicious|suspicious)"
```

## Support and Documentation

### Getting Help

#### Documentation
- **README.md**: Basic usage and setup
- **API Documentation**: http://localhost:8000/docs
- **GitHub Wiki**: Detailed guides and tutorials

#### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community help
- **Discord**: Real-time community chat

#### Professional Support
- **Email**: support@rockyai.org
- **Response Time**: 24-48 hours
- **Priority Support**: Available for enterprise users

### Contributing

#### Bug Reports
1. Check existing issues
2. Provide detailed reproduction steps
3. Include system information
4. Attach relevant logs

#### Feature Requests
1. Check roadmap
2. Describe use case
3. Provide mockups if applicable
4. Consider contributing implementation

---

**Last Updated**: December 2024
