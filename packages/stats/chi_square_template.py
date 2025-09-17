"""
Chi-square test analysis template
"""
from .analysis_templates import AnalysisTemplate
from typing import Dict, Any


class ChiSquareTemplate(AnalysisTemplate):
    """Chi-square test of independence template"""
    
    def __init__(self):
        super().__init__(
            name="chi_square",
            description="Chi-square test of independence",
            keywords=["chi-square", "chi square", "contingency", "categorical", "independence"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
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

# Assuming columns are: {data_info.get('columns', ['var1', 'var2'])}
var1_col = "{data_info.get('var1_column', 'var1')}"
var2_col = "{data_info.get('var2_column', 'var2')}"

# Create contingency table
contingency_table = pd.crosstab(df[var1_col], df[var2_col])
print("\\nContingency Table:")
print(contingency_table)

# Check assumptions
print("\\n=== ASSUMPTION CHECKS ===")

# Expected frequencies
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Expected frequencies (all should be ≥ 5):")
expected_df = pd.DataFrame(expected, 
                          index=contingency_table.index, 
                          columns=contingency_table.columns)
print(expected_df)

# Check if all expected frequencies are ≥ 5
min_expected = expected.min()
print(f"\\nMinimum expected frequency: {{min_expected:.2f}}")
if min_expected < 5:
    print("  → Warning: Some expected frequencies < 5. Consider combining categories.")
else:
    print("  → All expected frequencies ≥ 5. Assumption met.")

# Perform chi-square test
print("\\n=== CHI-SQUARE TEST RESULTS ===")
print(f"Chi-square statistic: {{chi2_stat:.4f}}")
print(f"Degrees of freedom: {{dof}}")
print(f"p-value: {{p_value:.4f}}")

# Critical value
alpha = 0.05
critical_value = chi2.ppf(1 - alpha, dof)
print(f"Critical value (α = {{alpha}}): {{critical_value:.4f}}")

# Effect size (Cramer's V)
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
print(f"Cramer's V (effect size): {{cramers_v:.4f}}")

# Interpretation
if p_value < alpha:
    print(f"\\nResult: Significant association between variables (p < {{alpha}})")
    if cramers_v < 0.1:
        effect_size = "negligible"
    elif cramers_v < 0.3:
        effect_size = "small"
    elif cramers_v < 0.5:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size: {{effect_size}} (Cramer's V = {{cramers_v:.4f}})")
else:
    print(f"\\nResult: No significant association between variables (p ≥ {{alpha}})")

# Standardized residuals
residuals = (contingency_table - expected) / np.sqrt(expected)
print("\\nStandardized residuals (|residual| > 2 indicates significant cell):")
print(residuals.round(3))

# Visualization
plt.figure(figsize=(15, 5))

# Contingency table heatmap
plt.subplot(1, 3, 1)
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title("Observed Frequencies")

# Expected frequencies heatmap
plt.subplot(1, 3, 2)
sns.heatmap(expected_df, annot=True, fmt='.1f', cmap='Greens')
plt.title("Expected Frequencies")

# Standardized residuals heatmap
plt.subplot(1, 3, 3)
sns.heatmap(residuals, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title("Standardized Residuals")

plt.tight_layout()
plt.show()

# Proportions by row
print("\\n=== ROW PROPORTIONS ===")
row_props = contingency_table.div(contingency_table.sum(axis=1), axis=0)
print(row_props.round(3))

# Proportions by column
print("\\n=== COLUMN PROPORTIONS ===")
col_props = contingency_table.div(contingency_table.sum(axis=0), axis=1)
print(col_props.round(3))

# Summary
print("\\n=== SUMMARY ===")
print(f"• Chi-square statistic: {{chi2_stat:.4f}}")
print(f"• p-value: {{p_value:.4f}}")
print(f"• Effect size (Cramer's V): {{cramers_v:.4f}}")
print(f"• Conclusion: {{'Significant' if p_value < alpha else 'Not significant'}} association between variables")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(broom)

# Load data
df <- read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
cat("Dataset dimensions:", dim(df), "\\n")
cat("\\nFirst few rows:\\n")
print(head(df))

# Check for missing values
cat("\\nMissing values:\\n")
print(colSums(is.na(df)))

# Assuming columns are: {data_info.get('columns', ['var1', 'var2'])}
var1_col <- "{data_info.get('var1_column', 'var1')}"
var2_col <- "{data_info.get('var2_column', 'var2')}"

# Create contingency table
contingency_table <- table(df[[var1_col]], df[[var2_col]])
cat("\\nContingency Table:\\n")
print(contingency_table)

# Check assumptions
cat("\\n=== ASSUMPTION CHECKS ===\\n")

# Perform chi-square test
chi_test <- chisq.test(contingency_table)
expected <- chi_test$expected

cat("Expected frequencies (all should be ≥ 5):\\n")
print(expected)

# Check if all expected frequencies are ≥ 5
min_expected <- min(expected)
cat("\\nMinimum expected frequency:", min_expected, "\\n")
if(min_expected < 5) {{
  cat("  → Warning: Some expected frequencies < 5. Consider combining categories.\\n")
}} else {{
  cat("  → All expected frequencies ≥ 5. Assumption met.\\n")
}}

# Test results
cat("\\n=== CHI-SQUARE TEST RESULTS ===\\n")
cat("Chi-square statistic:", chi_test$statistic, "\\n")
cat("Degrees of freedom:", chi_test$parameter, "\\n")
cat("p-value:", chi_test$p.value, "\\n")

# Critical value
alpha <- 0.05
critical_value <- qchisq(1 - alpha, chi_test$parameter)
cat("Critical value (α =", alpha, "):", critical_value, "\\n")

# Effect size (Cramer's V)
n <- sum(contingency_table)
cramers_v <- sqrt(chi_test$statistic / (n * (min(dim(contingency_table)) - 1)))
cat("Cramer's V (effect size):", cramers_v, "\\n")

# Interpretation
if(chi_test$p.value < alpha) {{
  cat("\\nResult: Significant association between variables (p <", alpha, ")\\n")
  if(cramers_v < 0.1) {{
    effect_size <- "negligible"
  }} else if(cramers_v < 0.3) {{
    effect_size <- "small"
  }} else if(cramers_v < 0.5) {{
    effect_size <- "medium"
  }} else {{
    effect_size <- "large"
  }}
  cat("Effect size:", effect_size, "(Cramer's V =", cramers_v, ")\\n")
}} else {{
  cat("\\nResult: No significant association between variables (p >=", alpha, ")\\n")
}}

# Standardized residuals
residuals <- chi_test$residuals
cat("\\nStandardized residuals (|residual| > 2 indicates significant cell):\\n")
print(round(residuals, 3))

# Visualization
library(ggplot2)
library(gridExtra)

# Convert to data frame for plotting
contingency_df <- as.data.frame(contingency_table)
names(contingency_df) <- c("Var1", "Var2", "Count")

expected_df <- as.data.frame(expected)
names(expected_df) <- c("Var1", "Var2", "Expected")

residuals_df <- as.data.frame(residuals)
names(residuals_df) <- c("Var1", "Var2", "Residual")

# Observed frequencies
p1 <- ggplot(contingency_df, aes(x = Var1, y = Var2, fill = Count)) +
  geom_tile() +
  geom_text(aes(label = Count), color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Observed Frequencies") +
  theme_minimal()

# Expected frequencies
p2 <- ggplot(expected_df, aes(x = Var1, y = Var2, fill = Expected)) +
  geom_tile() +
  geom_text(aes(label = round(Expected, 1)), color = "white") +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
  labs(title = "Expected Frequencies") +
  theme_minimal()

# Standardized residuals
p3 <- ggplot(residuals_df, aes(x = Var1, y = Var2, fill = Residual)) +
  geom_tile() +
  geom_text(aes(label = round(Residual, 2)), color = "white") +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0) +
  labs(title = "Standardized Residuals") +
  theme_minimal()

# Display plots
grid.arrange(p1, p2, p3, ncol = 3)

# Proportions by row
cat("\\n=== ROW PROPORTIONS ===\\n")
row_props <- prop.table(contingency_table, margin = 1)
print(round(row_props, 3))

# Proportions by column
cat("\\n=== COLUMN PROPORTIONS ===\\n")
col_props <- prop.table(contingency_table, margin = 2)
print(round(col_props, 3))

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• Chi-square statistic:", chi_test$statistic, "\\n")
cat("• p-value:", chi_test$p.value, "\\n")
cat("• Effect size (Cramer's V):", cramers_v, "\\n")
cat("• Conclusion:", ifelse(chi_test$p.value < alpha, "Significant", "Not significant"), "association between variables\\n")
'''
