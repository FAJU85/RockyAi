"""
Principal Component Analysis (PCA) template
"""
from .analysis_templates import AnalysisTemplate
from typing import Dict, Any


class PCATemplate(AnalysisTemplate):
    """Principal Component Analysis template"""
    
    def __init__(self):
        super().__init__(
            name="pca",
            description="Principal Component Analysis",
            keywords=["pca", "principal component", "dimensionality reduction", "factor analysis"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
print("Dataset shape:", df.shape)
print("\\nFirst few rows:")
print(df.head())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Assuming numeric columns for PCA
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\\nNumeric columns: {{numeric_cols}}")

# Prepare data (remove missing values)
df_clean = df[numeric_cols].dropna()
print(f"\\nClean data shape: {{df_clean.shape}}")

if df_clean.shape[0] < 2:
    print("Error: Not enough data points for PCA")
    exit()

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

print("\\n=== PCA ANALYSIS ===")

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\\nExplained variance by component:")
for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"PC{{i+1}}: {{var:.4f}} ({{var*100:.2f}}%) - Cumulative: {{cum_var:.4f}} ({{cum_var*100:.2f}}%)")

# Determine number of components to retain
# Rule of thumb: retain components that explain 80-90% of variance
n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1

print(f"\\nComponents explaining 80% variance: {{n_components_80}}")
print(f"Components explaining 90% variance: {{n_components_90}}")

# Kaiser criterion: retain components with eigenvalue > 1
eigenvalues = pca.explained_variance_
n_components_kaiser = np.sum(eigenvalues > 1)
print(f"Components with eigenvalue > 1 (Kaiser): {{n_components_kaiser}}")

# Scree plot
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.axhline(y=0.1, color='r', linestyle='--', label='10% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.axhline(y=0.8, color='g', linestyle='--', label='80% threshold')
plt.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.legend()
plt.grid(True)

# Component loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

plt.subplot(1, 3, 3)
sns.heatmap(loadings[:, :min(5, loadings.shape[1])], 
            xticklabels=[f'PC{{i+1}}' for i in range(min(5, loadings.shape[1]))],
            yticklabels=numeric_cols,
            annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Component Loadings (first 5 PCs)')
plt.tight_layout()
plt.show()

# Create PCA dataframe
pca_df = pd.DataFrame(X_pca, columns=[f'PC{{i+1}}' for i in range(X_pca.shape[1])])

# Add original data info if available
if 'group' in df.columns:
    pca_df['group'] = df.loc[df_clean.index, 'group'].values

print("\\n=== PRINCIPAL COMPONENTS ===")
print("First 5 principal components:")
print(pca_df.iloc[:5, :min(5, pca_df.shape[1])])

# Biplot (first two components)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.xlabel(f'PC1 ({{explained_variance_ratio[0]*100:.1f}}%)')
plt.ylabel(f'PC2 ({{explained_variance_ratio[1]*100:.1f}}%)')
plt.title('PCA Biplot - Observations')
plt.grid(True)

# Add variable vectors
for i, var in enumerate(numeric_cols):
    plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
              head_width=0.1, head_length=0.1, fc='red', ec='red')
    plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, var, fontsize=8)

# If groups available, color by group
if 'group' in df.columns:
    plt.subplot(1, 2, 2)
    groups = df.loc[df_clean.index, 'group'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
    
    for i, group in enumerate(groups):
        mask = df.loc[df_clean.index, 'group'] == group
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=group, alpha=0.6)
    
    plt.xlabel(f'PC1 ({{explained_variance_ratio[0]*100:.1f}}%)')
    plt.ylabel(f'PC2 ({{explained_variance_ratio[1]*100:.1f}}%)')
    plt.title('PCA Biplot - Colored by Group')
    plt.legend()
    plt.grid(True)
else:
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.xlabel(f'PC1 ({{explained_variance_ratio[0]*100:.1f}}%)')
    plt.ylabel(f'PC2 ({{explained_variance_ratio[1]*100:.1f}}%)')
    plt.title('PCA Biplot - All Observations')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Component interpretation
print("\\n=== COMPONENT INTERPRETATION ===")
for i in range(min(3, loadings.shape[1])):
    print(f"\\nPC{{i+1}} ({{explained_variance_ratio[i]*100:.1f}}% variance):")
    
    # Get top contributing variables
    pc_loadings = loadings[:, i]
    top_vars = np.argsort(np.abs(pc_loadings))[-3:][::-1]
    
    for j, var_idx in enumerate(top_vars):
        var_name = numeric_cols[var_idx]
        loading = pc_loadings[var_idx]
        print(f"  {{j+1}}. {{var_name}}: {{loading:.3f}}")

# Summary
print("\\n=== SUMMARY ===")
print(f"• Total variance explained by first 2 components: {{cumulative_variance[1]*100:.1f}}%")
print(f"• Total variance explained by first 3 components: {{cumulative_variance[2]*100:.1f}}%")
print(f"• Recommended components (80% variance): {{n_components_80}}")
print(f"• Recommended components (90% variance): {{n_components_90}}")
print(f"• Kaiser criterion: {{n_components_kaiser}} components")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(factoextra)
library(corrplot)

# Load data
df <- read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
cat("Dataset dimensions:", dim(df), "\\n")
cat("\\nFirst few rows:\\n")
print(head(df))

# Check for missing values
cat("\\nMissing values:\\n")
print(colSums(is.na(df)))

# Assuming numeric columns for PCA
numeric_cols <- df %>% select_if(is.numeric) %>% names()
cat("\\nNumeric columns:", numeric_cols, "\\n")

# Prepare data (remove missing values)
df_clean <- df %>% 
  select(all_of(numeric_cols)) %>% 
  na.omit()

cat("\\nClean data dimensions:", dim(df_clean), "\\n")

if(nrow(df_clean) < 2) {{
  stop("Error: Not enough data points for PCA")
}}

# Standardize the data
df_scaled <- df_clean %>% 
  scale() %>% 
  as.data.frame()

cat("\\n=== PCA ANALYSIS ===\\n")

# Perform PCA
pca_result <- prcomp(df_scaled, center = FALSE, scale. = FALSE)

# Explained variance
explained_variance <- pca_result$sdev^2
explained_variance_ratio <- explained_variance / sum(explained_variance)
cumulative_variance <- cumsum(explained_variance_ratio)

cat("\\nExplained variance by component:\\n")
for(i in 1:length(explained_variance_ratio)) {{
  cat(sprintf("PC%d: %.4f (%.2f%%) - Cumulative: %.4f (%.2f%%)\\n", 
              i, explained_variance_ratio[i], explained_variance_ratio[i]*100,
              cumulative_variance[i], cumulative_variance[i]*100))
}}

# Determine number of components to retain
n_components_80 <- which(cumulative_variance >= 0.8)[1]
n_components_90 <- which(cumulative_variance >= 0.9)[1]

cat("\\nComponents explaining 80% variance:", n_components_80, "\\n")
cat("Components explaining 90% variance:", n_components_90, "\\n")

# Kaiser criterion: retain components with eigenvalue > 1
n_components_kaiser <- sum(explained_variance > 1)
cat("Components with eigenvalue > 1 (Kaiser):", n_components_kaiser, "\\n")

# Scree plot
par(mfrow = c(1, 3))

# Scree plot
plot(1:length(explained_variance_ratio), explained_variance_ratio, 
     type = "b", pch = 19, col = "blue",
     xlab = "Principal Component", ylab = "Explained Variance Ratio",
     main = "Scree Plot")
abline(h = 0.1, col = "red", lty = 2)
legend("topright", "10% threshold", col = "red", lty = 2)

# Cumulative variance
plot(1:length(cumulative_variance), cumulative_variance, 
     type = "b", pch = 19, col = "red",
     xlab = "Principal Component", ylab = "Cumulative Explained Variance",
     main = "Cumulative Variance Explained")
abline(h = 0.8, col = "green", lty = 2)
abline(h = 0.9, col = "orange", lty = 2)
legend("bottomright", c("80% threshold", "90% threshold"), 
       col = c("green", "orange"), lty = 2)

# Component loadings heatmap
loadings <- pca_result$rotation[, 1:min(5, ncol(pca_result$rotation))]
colnames(loadings) <- paste0("PC", 1:ncol(loadings))
rownames(loadings) <- numeric_cols

# Create heatmap
library(pheatmap)
pheatmap(loadings, 
         cluster_rows = FALSE, cluster_cols = FALSE,
         display_numbers = TRUE, number_format = "%.2f",
         main = "Component Loadings (first 5 PCs)")

# Create PCA dataframe
pca_df <- as.data.frame(pca_result$x)
colnames(pca_df) <- paste0("PC", 1:ncol(pca_df))

# Add original data info if available
if("group" %in% names(df)) {{
  pca_df$group <- df[complete.cases(df[numeric_cols]), "group"]
}}

cat("\\n=== PRINCIPAL COMPONENTS ===\\n")
cat("First 5 principal components:\\n")
print(head(pca_df[, 1:min(5, ncol(pca_df))]))

# Biplot
par(mfrow = c(1, 2))

# Basic biplot
biplot(pca_result, main = "PCA Biplot", cex = 0.8)

# If groups available, create colored biplot
if("group" %in% names(df)) {{
  groups <- unique(df[complete.cases(df[numeric_cols]), "group"])
  colors <- rainbow(length(groups))
  
  plot(pca_result$x[, 1], pca_result$x[, 2], 
       xlab = paste0("PC1 (", round(explained_variance_ratio[1]*100, 1), "%)"),
       ylab = paste0("PC2 (", round(explained_variance_ratio[2]*100, 1), "%)"),
       main = "PCA Biplot - Colored by Group", pch = 19)
  
  for(i in 1:length(groups)) {{
    group_data <- pca_result$x[df[complete.cases(df[numeric_cols]), "group"] == groups[i], ]
    points(group_data[, 1], group_data[, 2], col = colors[i], pch = 19)
  }}
  
  legend("topright", groups, col = colors, pch = 19)
}} else {{
  plot(pca_result$x[, 1], pca_result$x[, 2], 
       xlab = paste0("PC1 (", round(explained_variance_ratio[1]*100, 1), "%)"),
       ylab = paste0("PC2 (", round(explained_variance_ratio[2]*100, 1), "%)"),
       main = "PCA Biplot - All Observations", pch = 19)
}}

# Component interpretation
cat("\\n=== COMPONENT INTERPRETATION ===\\n")
for(i in 1:min(3, ncol(pca_result$rotation))) {{
  cat("\\nPC", i, "(", round(explained_variance_ratio[i]*100, 1), "% variance):\\n")
  
  # Get top contributing variables
  pc_loadings <- pca_result$rotation[, i]
  top_vars <- order(abs(pc_loadings), decreasing = TRUE)[1:3]
  
  for(j in 1:length(top_vars)) {{
    var_name <- numeric_cols[top_vars[j]]
    loading <- pc_loadings[top_vars[j]]
    cat("  ", j, ". ", var_name, ": ", round(loading, 3), "\\n")
  }}
}}

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• Total variance explained by first 2 components:", round(cumulative_variance[2]*100, 1), "%\\n")
cat("• Total variance explained by first 3 components:", round(cumulative_variance[3]*100, 1), "%\\n")
cat("• Recommended components (80% variance):", n_components_80, "\\n")
cat("• Recommended components (90% variance):", n_components_90, "\\n")
cat("• Kaiser criterion:", n_components_kaiser, "components\\n")
'''
