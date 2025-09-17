# Rocky AI - Research Assistant

> "The AI Research Assistant That Thinks Like a Statistician, Codes Like a Pro, and Explains Like a Teacher"

Rocky AI is a next-generation, open-source research assistant designed to revolutionize how researchers, students, data analysts, and scientists interact with data. It combines the ease of natural language interaction with the power and rigor of R and Python, enhanced by local AI capabilities and a commitment to transparency and reproducibility.

## 🚀 Features

### ✅ Core Capabilities
- **Conversational Data Analysis**: Ask questions in plain English
- **Dual-Language Powerhouse**: Native R & Python execution with full integration
- **Local AI with Privacy**: Docker Model Runner with ai/llama2/llama3
- **Comprehensive Statistics**: t-tests, ANOVA, regression, survival analysis, mixed models
- **Machine Learning Integration**: Scikit-learn, tidymodels, advanced ML workflows
- **Modern Web Interface**: React-based UI with real-time chat and enhanced UX
- **Reproducible Research**: Every result tied to verifiable code
- **Export Capabilities**: R Markdown, PDF, code generation

### 🆕 Enhanced Features (v0.2.0)
- **High-Performance Caching**: Redis-based caching for faster analysis
- **Database Integration**: PostgreSQL for metadata and analysis history
- **Advanced Error Handling**: Comprehensive error recovery and user feedback
- **Structured Logging**: Correlation IDs and detailed logging for debugging
- **Async Processing**: Non-blocking analysis execution
- **Health Monitoring**: Comprehensive health checks for all services
- **Configuration Management**: Environment-based configuration system
- **Background Tasks**: Celery integration for long-running analyses

### 🔬 Statistical Analysis Suite
- **Descriptive Statistics**: Summary statistics, distributions, correlations
- **Inferential Statistics**: t-tests, ANOVA, chi-square tests, non-parametric tests
- **Regression Analysis**: Linear, logistic, mixed effects models
- **Survival Analysis**: Kaplan-Meier curves, Cox proportional hazards
- **Advanced Methods**: PCA, mixed models, time series analysis
- **Machine Learning**: Classification, regression, clustering, feature selection

## 🏃‍♂️ Quick Start

### Prerequisites
- Docker and Docker Compose
- At least 8GB RAM (16GB+ recommended)
- Optional: NVIDIA GPU with 10GB+ VRAM for ai/llama3

### Quick Start

#### Windows
1. **Navigate to the project:**
   ```cmd
   cd C:\Users\user\RockyAi
   ```

2. **Run the startup script:**
   ```cmd
   start.bat
   ```

#### Linux/macOS
1. **Navigate to the project:**
   ```bash
   cd /path/to/RockyAi
   ```

2. **Run the startup script:**
   ```bash
   ./start.sh
   ```

#### Manual Start
1. **Create environment file:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

2. **Start all services:**
   ```bash
   docker-compose up --build -d
   ```

3. **Access the application:**
   - **Web UI**: http://localhost:5173
   - **API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs
   - **DMR**: http://localhost:11434
   - **Database**: localhost:5432
   - **Redis**: localhost:6379

### First Analysis

1. Open the web interface at http://localhost:5173
2. Upload a dataset or use the sample data
3. Ask questions like:
   - "Perform a t-test to compare groups A and B"
   - "Create a Kaplan-Meier survival curve"
   - "Run a mixed effects model for repeated measures"
   - "Show me the correlation matrix"

## 🏗️ Architecture

### Services
- **UI (React)**: Modern web interface with Tailwind CSS
- **API (FastAPI)**: Backend orchestration and code generation
- **DMR**: Docker Model Runner for local AI models
- **Python Executor**: Sandboxed Python code execution
- **R Executor**: Sandboxed R code execution

### Model Selection
The system automatically selects the best model based on your hardware:
- **CPU Only**: ai/llama2 (default)
- **GPU Available**: ai/llama3 (if 10GB+ VRAM)

## 📁 Project Structure

```
rocky/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── app/
│   │   │   ├── main.py         # API endpoints
│   │   │   └── orchestrator.py # Chat→plan→code orchestration
│   │   └── requirements.txt
│   └── ui/                     # React frontend
│       ├── src/
│       │   ├── components/     # UI components
│       │   ├── App.tsx         # Main app
│       │   └── main.tsx
│       └── package.json
├── services/
│   ├── dmr/                    # Docker Model Runner config
│   │   ├── config.yaml
│   │   └── model_detector.py
│   └── runners/
│       ├── python/             # Python executor
│       └── r/                  # R executor
├── packages/
│   └── stats/                  # Analysis templates
│       ├── analysis_templates.py
│       ├── survival_template.py
│       └── mixed_models_template.py
├── docker-compose.yml          # Local development setup
└── README.md
```

## 🛠️ Development

### API Development
```bash
cd apps/api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### UI Development
```bash
cd apps/ui
npm install
npm run dev
```

### Adding New Analysis Types
1. Create a new template in `packages/stats/`
2. Extend the `AnalysisTemplate` base class
3. Implement `get_python_code()` and `get_r_code()` methods
4. Register in `packages/stats/__init__.py`

## 🔒 Security & Privacy

- **Local Processing**: All data stays on your machine
- **Sandboxed Execution**: Code runs in isolated containers
- **No Network Access**: Executors have no internet connectivity
- **Resource Limits**: Memory and CPU constraints prevent abuse
- **Library Allowlists**: Only approved packages can be imported

## 📊 Supported Analyses

### Statistical Tests
- **t-tests**: Independent samples, paired, Welch's
- **ANOVA**: One-way, repeated measures, mixed models
- **Chi-square**: Independence tests, goodness of fit
- **Non-parametric**: Mann-Whitney U, Kruskal-Wallis, Wilcoxon

### Regression Analysis
- **Linear Regression**: Simple, multiple, polynomial
- **Logistic Regression**: Binary, multinomial
- **Mixed Effects**: Random intercepts, random slopes
- **Survival Analysis**: Cox proportional hazards, Kaplan-Meier

### Machine Learning
- **Classification**: Random Forest, SVM, Neural Networks
- **Regression**: Gradient Boosting, Elastic Net
- **Clustering**: K-means, Hierarchical, DBSCAN
- **Dimensionality Reduction**: PCA, LDA, t-SNE

## 🚀 Deployment

### Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/rocky-ai
gcloud run deploy rocky-ai --image gcr.io/PROJECT_ID/rocky-ai
```

### Alibaba Cloud
```bash
# Deploy to Alibaba Container Service
aliyun cs CreateCluster --name rocky-ai-cluster
aliyun cs CreateApplication --cluster-id CLUSTER_ID --template rocky-ai.yaml
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Out of Memory**
- Reduce model size in `services/dmr/config.yaml`
- Increase system RAM or use smaller datasets

**GPU Not Detected**
- Ensure NVIDIA drivers are installed
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

**Port Conflicts**
- Modify ports in `docker-compose.yml`
- Check for running services: `netstat -tulpn | grep :8000`

**Analysis Fails**
- Check data format and column names
- Verify required libraries are installed
- Review error messages in the Results pane

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-org/rocky-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rocky-ai/discussions)
- **Documentation**: [Wiki](https://github.com/your-org/rocky-ai/wiki)

## 🎯 Roadmap

### v0.2.0 (Next Release)
- [ ] R Markdown export to PDF/HTML
- [ ] Dataset ingestion and caching
- [ ] Advanced visualization module
- [ ] Cloud deployment automation

### v0.3.0 (Future)
- [ ] Multi-language UI support
- [ ] Collaborative features
- [ ] Live data connectors
- [ ] Advanced ML pipelines

---

**Built with ❤️ for the research community**
