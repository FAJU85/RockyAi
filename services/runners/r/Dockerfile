FROM r-base:4.3.2

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('tidyverse', 'broom', 'car', 'survival', 'survminer', 'lme4', 'nlme', 'renv', 'jsonlite'), repos='https://cran.rstudio.com/')"

# Copy executor
COPY executor.R .

# Create non-root user for security
RUN useradd -m -u 1001 rocky && \
    chown -R rocky:rocky /app

USER rocky

# Default command
CMD ["Rscript", "executor.R"]
