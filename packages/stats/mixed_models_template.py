"""
Mixed Effects Models (LME/GLMM) templates
"""
from .analysis_templates import AnalysisTemplate
from typing import Dict, Any


class MixedModelsTemplate(AnalysisTemplate):
    """Mixed effects models template"""
    
    def __init__(self):
        super().__init__(
            name="mixed_models",
            description="Mixed effects models (LME/GLMM) for repeated measures and hierarchical data",
            keywords=["mixed model", "lme", "glmm", "repeated measures", "random effects", "hierarchical"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Note: For mixed models in Python, we'll use statsmodels
# which has limited mixed model support. For full functionality,
# consider using R or specialized packages like pymer4

try:
    from statsmodels.formula.api import mixedlm
    from statsmodels.stats.anova import anova_lm
    from statsmodels.regression.mixed_linear_model import MixedLM
    MIXED_MODELS_AVAILABLE = True
except ImportError:
    MIXED_MODELS_AVAILABLE = False
    print("Warning: Mixed models not fully available in Python.")
    print("Consider using R for comprehensive mixed model analysis.")

# Load data
df = pd.read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
print("Dataset shape:", df.shape)
print("\\nFirst few rows:")
print(df.head())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Assuming columns are: {data_info.get('columns', ['subject', 'time', 'outcome', 'group'])}
subject_col = "{data_info.get('subject_column', 'subject')}"
time_col = "{data_info.get('time_column', 'time')}"
outcome_col = "{data_info.get('outcome_column', 'outcome')}"
group_col = "{data_info.get('group_column', 'group')}" if "{data_info.get('group_column', 'group')}" in df.columns else None

# Check data structure
print(f"\\nData structure:")
print(f"  Subjects: {{df[subject_col].nunique()}}")
print(f"  Time points: {{df[time_col].nunique()}}")
print(f"  Time range: {{df[time_col].min()}} to {{df[time_col].max()}}")

if group_col:
    print(f"  Groups: {{df[group_col].unique()}}")
    print(f"  Group sizes: {{df[group_col].value_counts().to_dict()}}")

# Check for balanced design
subject_counts = df[subject_col].value_counts()
print(f"\\nObservations per subject:")
print(f"  Min: {{subject_counts.min()}}, Max: {{subject_counts.max()}}")
print(f"  Mean: {{subject_counts.mean():.1f}}, Std: {{subject_counts.std():.1f}}")

if subject_counts.std() == 0:
    print("  → Balanced design")
else:
    print("  → Unbalanced design")

# Data exploration
print("\\n=== DATA EXPLORATION ===")

# Plot individual trajectories
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for subject in df[subject_col].unique()[:10]:  # Show first 10 subjects
    subject_data = df[df[subject_col] == subject]
    plt.plot(subject_data[time_col], subject_data[outcome_col], 
             alpha=0.7, marker='o', markersize=3)
plt.xlabel(time_col)
plt.ylabel(outcome_col)
plt.title('Individual Trajectories (first 10 subjects)')
plt.grid(True, alpha=0.3)

# Group means over time
if group_col:
    plt.subplot(2, 3, 2)
    group_means = df.groupby([group_col, time_col])[outcome_col].mean().reset_index()
    for group in df[group_col].unique():
        group_data = group_means[group_means[group_col] == group]
        plt.plot(group_data[time_col], group_data[outcome_col], 
                marker='o', label=f'Group {group}', linewidth=2)
    plt.xlabel(time_col)
    plt.ylabel(f'Mean {outcome_col}')
    plt.title('Group Means Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Distribution of outcomes
plt.subplot(2, 3, 3)
plt.hist(df[outcome_col], bins=30, alpha=0.7, edgecolor='black')
plt.xlabel(outcome_col)
plt.ylabel('Frequency')
plt.title('Distribution of Outcome')
plt.grid(True, alpha=0.3)

# Box plot by time
plt.subplot(2, 3, 4)
df.boxplot(column=outcome_col, by=time_col, ax=plt.gca())
plt.title(f'{outcome_col} by {time_col}')
plt.suptitle('')  # Remove default title
plt.grid(True, alpha=0.3)

# Box plot by group (if available)
if group_col:
    plt.subplot(2, 3, 5)
    df.boxplot(column=outcome_col, by=group_col, ax=plt.gca())
    plt.title(f'{outcome_col} by {group_col}')
    plt.suptitle('')  # Remove default title
    plt.grid(True, alpha=0.3)

# Correlation matrix (if multiple time points)
if df[time_col].nunique() > 2:
    plt.subplot(2, 3, 6)
    pivot_data = df.pivot(index=subject_col, columns=time_col, values=outcome_col)
    correlation_matrix = pivot_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Across Time')

plt.tight_layout()
plt.show()

# Basic statistics
print("\\n=== DESCRIPTIVE STATISTICS ===")
desc_stats = df.groupby([time_col] + ([group_col] if group_col else []))[outcome_col].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(3)
print(desc_stats)

# Intraclass correlation coefficient (ICC)
print("\\n=== INTRACLASS CORRELATION COEFFICIENT ===")

# Calculate ICC using variance components
from scipy.stats import f_oneway

# One-way ANOVA to get variance components
subjects = df[subject_col].unique()
subject_means = [df[df[subject_col] == subj][outcome_col].mean() for subj in subjects]
subject_data = [df[df[subject_col] == subj][outcome_col].values for subj in subjects]

# Between-subjects variance
between_var = np.var(subject_means) * len(subject_data[0])  # Approximate
# Within-subjects variance
within_var = np.mean([np.var(subj_data) for subj_data in subject_data])

icc = between_var / (between_var + within_var)
print(f"ICC: {{icc:.4f}}")

if icc > 0.75:
    print("  → High correlation within subjects")
elif icc > 0.5:
    print("  → Moderate correlation within subjects")
elif icc > 0.25:
    print("  → Low correlation within subjects")
else:
    print("  → Very low correlation within subjects")

# Mixed model analysis
if MIXED_MODELS_AVAILABLE:
    print("\\n=== MIXED MODEL ANALYSIS ===")
    
    # Model 1: Random intercept only
    print("\\nModel 1: Random intercept")
    try:
        if group_col:
            model1 = MixedLM.from_formula(
                f"{outcome_col} ~ {time_col} + {group_col}",
                data=df, 
                groups=df[subject_col]
            )
        else:
            model1 = MixedLM.from_formula(
                f"{outcome_col} ~ {time_col}",
                data=df, 
                groups=df[subject_col]
            )
        
        result1 = model1.fit()
        print(result1.summary())
        
        # Model 2: Random intercept and slope
        print("\\nModel 2: Random intercept and slope")
        try:
            if group_col:
                model2 = MixedLM.from_formula(
                    f"{outcome_col} ~ {time_col} + {group_col}",
                    data=df, 
                    groups=df[subject_col],
                    re_formula=f"1 + {time_col}"
                )
            else:
                model2 = MixedLM.from_formula(
                    f"{outcome_col} ~ {time_col}",
                    data=df, 
                    groups=df[subject_col],
                    re_formula=f"1 + {time_col}"
                )
            
            result2 = model2.fit()
            print(result2.summary())
            
            # Model comparison
            print("\\nModel Comparison:")
            print(f"Model 1 AIC: {{result1.aic:.2f}}")
            print(f"Model 2 AIC: {{result2.aic:.2f}}")
            
            if result2.aic < result1.aic:
                print("  → Model 2 (random slope) preferred")
                best_model = result2
            else:
                print("  → Model 1 (random intercept only) preferred")
                best_model = result1
                
        except Exception as e:
            print(f"  → Model 2 failed: {{e}}")
            print("  → Using Model 1")
            best_model = result1
            
    except Exception as e:
        print(f"Mixed model fitting failed: {{e}}")
        print("Consider using R for mixed model analysis")
        best_model = None

else:
    print("\\n=== ALTERNATIVE ANALYSIS (Repeated Measures ANOVA) ===")
    
    # Repeated measures ANOVA as alternative
    from scipy.stats import f_oneway
    
    # Prepare data for repeated measures ANOVA
    time_points = sorted(df[time_col].unique())
    subject_data_by_time = []
    
    for time_point in time_points:
        time_data = df[df[time_col] == time_point][outcome_col].values
        subject_data_by_time.append(time_data)
    
    # One-way ANOVA across time points
    f_stat, p_value = f_oneway(*subject_data_by_time)
    
    print(f"Repeated Measures ANOVA:")
    print(f"  F-statistic: {{f_stat:.4f}}")
    print(f"  p-value: {{p_value:.4f}}")
    
    if p_value < 0.05:
        print("  → Significant effect of time")
    else:
        print("  → No significant effect of time")

# Post-hoc analysis
print("\\n=== POST-HOC ANALYSIS ===")

# Pairwise comparisons between time points
from scipy.stats import ttest_rel
from itertools import combinations

time_pairs = list(combinations(time_points, 2))
print("\\nPairwise comparisons between time points:")

for time1, time2 in time_pairs:
    data1 = df[df[time_col] == time1][outcome_col].values
    data2 = df[df[time_col] == time2][outcome_col].values
    
    # Paired t-test
    t_stat, p_val = ttest_rel(data1, data2)
    
    print(f"  {time1} vs {time2}: t = {{t_stat:.3f}}, p = {{p_val:.4f}}")
    
    if p_val < 0.05:
        mean_diff = np.mean(data2) - np.mean(data1)
        print(f"    → Significant difference (mean change: {{mean_diff:.3f}})")

# Summary
print("\\n=== SUMMARY ===")
print(f"• Sample size: {{len(df)}} observations from {{df[subject_col].nunique()}} subjects")
print(f"• Time points: {{df[time_col].nunique()}}")
print(f"• ICC: {{icc:.4f}}")
print(f"• Design: {{'Balanced' if subject_counts.std() == 0 else 'Unbalanced'}}")

if MIXED_MODELS_AVAILABLE and best_model:
    print(f"• Best model AIC: {{best_model.aic:.2f}}")
    print("• Mixed model analysis completed")
else:
    print("• Repeated measures ANOVA used as alternative")
    print("• Consider R for full mixed model capabilities")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(lme4)
library(nlme)
library(broom)
library(emmeans)
library(performance)
library(ggplot2)

# Load data
df <- read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
cat("Dataset dimensions:", dim(df), "\\n")
cat("\\nFirst few rows:\\n")
print(head(df))

# Check for missing values
cat("\\nMissing values:\\n")
print(colSums(is.na(df)))

# Assuming columns are: {data_info.get('columns', ['subject', 'time', 'outcome', 'group'])}
subject_col <- "{data_info.get('subject_column', 'subject')}"
time_col <- "{data_info.get('time_column', 'time')}"
outcome_col <- "{data_info.get('outcome_column', 'outcome')}"
group_col <- "{data_info.get('group_column', 'group')}"

# Check if group column exists
has_group <- group_col %in% names(df)
if(!has_group) group_col <- NULL

# Check data structure
cat("\\nData structure:\\n")
cat("  Subjects:", length(unique(df[[subject_col]])), "\\n")
cat("  Time points:", length(unique(df[[time_col]])), "\\n")
cat("  Time range:", min(df[[time_col]]), "to", max(df[[time_col]]), "\\n")

if(has_group) {{
  cat("  Groups:", unique(df[[group_col]]), "\\n")
  cat("  Group sizes:", table(df[[group_col]]), "\\n")
}}

# Check for balanced design
subject_counts <- table(df[[subject_col]])
cat("\\nObservations per subject:\\n")
cat("  Min:", min(subject_counts), "Max:", max(subject_counts), "\\n")
cat("  Mean:", round(mean(subject_counts), 1), "Std:", round(sd(subject_counts), 1), "\\n")

if(sd(subject_counts) == 0) {{
  cat("  → Balanced design\\n")
}} else {{
  cat("  → Unbalanced design\\n")
}}

# Data exploration
cat("\\n=== DATA EXPLORATION ===\\n")

# Plot individual trajectories
p1 <- ggplot(df, aes(x = !!sym(time_col), y = !!sym(outcome_col), group = !!sym(subject_col))) +
  geom_line(alpha = 0.3) +
  geom_point(alpha = 0.5, size = 0.8) +
  labs(title = "Individual Trajectories", x = time_col, y = outcome_col) +
  theme_minimal()

# Group means over time
if(has_group) {{
  p2 <- df %>% 
    group_by(!!sym(group_col), !!sym(time_col)) %>% 
    summarise(mean_outcome = mean(!!sym(outcome_col), na.rm = TRUE), .groups = 'drop') %>% 
    ggplot(aes(x = !!sym(time_col), y = mean_outcome, color = !!sym(group_col))) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(title = "Group Means Over Time", x = time_col, y = paste("Mean", outcome_col)) +
    theme_minimal()
}} else {{
  p2 <- df %>% 
    group_by(!!sym(time_col)) %>% 
    summarise(mean_outcome = mean(!!sym(outcome_col), na.rm = TRUE), .groups = 'drop') %>% 
    ggplot(aes(x = !!sym(time_col), y = mean_outcome)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    labs(title = "Mean Over Time", x = time_col, y = paste("Mean", outcome_col)) +
    theme_minimal()
}}

# Distribution of outcomes
p3 <- ggplot(df, aes(x = !!sym(outcome_col))) +
  geom_histogram(bins = 30, alpha = 0.7, fill = "skyblue") +
  labs(title = "Distribution of Outcome", x = outcome_col, y = "Frequency") +
  theme_minimal()

# Box plot by time
p4 <- ggplot(df, aes(x = factor(!!sym(time_col)), y = !!sym(outcome_col))) +
  geom_boxplot() +
  labs(title = paste(outcome_col, "by", time_col), x = time_col, y = outcome_col) +
  theme_minimal()

# Box plot by group (if available)
if(has_group) {{
  p5 <- ggplot(df, aes(x = factor(!!sym(group_col)), y = !!sym(outcome_col))) +
    geom_boxplot() +
    labs(title = paste(outcome_col, "by", group_col), x = group_col, y = outcome_col) +
    theme_minimal()
  
  # Combine plots
  library(gridExtra)
  grid.arrange(p1, p2, p3, p4, p5, ncol = 3)
}} else {{
  library(gridExtra)
  grid.arrange(p1, p2, p3, p4, ncol = 2)
}}

# Basic statistics
cat("\\n=== DESCRIPTIVE STATISTICS ===\\n")
if(has_group) {{
  desc_stats <- df %>% 
    group_by(!!sym(time_col), !!sym(group_col)) %>% 
    summarise(
      n = n(),
      mean = mean(!!sym(outcome_col), na.rm = TRUE),
      sd = sd(!!sym(outcome_col), na.rm = TRUE),
      min = min(!!sym(outcome_col), na.rm = TRUE),
      max = max(!!sym(outcome_col), na.rm = TRUE),
      .groups = 'drop'
    )
}} else {{
  desc_stats <- df %>% 
    group_by(!!sym(time_col)) %>% 
    summarise(
      n = n(),
      mean = mean(!!sym(outcome_col), na.rm = TRUE),
      sd = sd(!!sym(outcome_col), na.rm = TRUE),
      min = min(!!sym(outcome_col), na.rm = TRUE),
      max = max(!!sym(outcome_col), na.rm = TRUE),
      .groups = 'drop'
    )
}}
print(desc_stats)

# Intraclass correlation coefficient (ICC)
cat("\\n=== INTRACLASS CORRELATION COEFFICIENT ===\\n")

# Calculate ICC using lme4
if(has_group) {{
  icc_model <- lmer(!!sym(outcome_col) ~ 1 + (1|!!sym(subject_col)), data = df)
}} else {{
  icc_model <- lmer(!!sym(outcome_col) ~ 1 + (1|!!sym(subject_col)), data = df)
}}

icc_result <- icc(icc_model)
cat("ICC:", round(icc_result, 4), "\\n")

if(icc_result > 0.75) {{
  cat("  → High correlation within subjects\\n")
}} else if(icc_result > 0.5) {{
  cat("  → Moderate correlation within subjects\\n")
}} else if(icc_result > 0.25) {{
  cat("  → Low correlation within subjects\\n")
}} else {{
  cat("  → Very low correlation within subjects\\n")
}}

# Mixed model analysis
cat("\\n=== MIXED MODEL ANALYSIS ===\\n")

# Model 1: Random intercept only
cat("\\nModel 1: Random intercept\\n")
if(has_group) {{
  model1 <- lmer(!!sym(outcome_col) ~ !!sym(time_col) + !!sym(group_col) + (1|!!sym(subject_col)), 
                 data = df)
}} else {{
  model1 <- lmer(!!sym(outcome_col) ~ !!sym(time_col) + (1|!!sym(subject_col)), 
                 data = df)
}}

print(summary(model1))

# Model 2: Random intercept and slope
cat("\\nModel 2: Random intercept and slope\\n")
if(has_group) {{
  model2 <- lmer(!!sym(outcome_col) ~ !!sym(time_col) + !!sym(group_col) + (1 + !!sym(time_col)|!!sym(subject_col)), 
                 data = df)
}} else {{
  model2 <- lmer(!!sym(outcome_col) ~ !!sym(time_col) + (1 + !!sym(time_col)|!!sym(subject_col)), 
                 data = df)
}}

print(summary(model2))

# Model comparison
cat("\\nModel Comparison:\\n")
aic1 <- AIC(model1)
aic2 <- AIC(model2)
cat("Model 1 AIC:", round(aic1, 2), "\\n")
cat("Model 2 AIC:", round(aic2, 2), "\\n")

if(aic2 < aic1) {{
  cat("  → Model 2 (random slope) preferred\\n")
  best_model <- model2
}} else {{
  cat("  → Model 1 (random intercept only) preferred\\n")
  best_model <- model1
}}

# Model diagnostics
cat("\\n=== MODEL DIAGNOSTICS ===\\n")

# Check model assumptions
assumptions <- check_model(best_model)
print(assumptions)

# Random effects
cat("\\nRandom Effects:\\n")
ranef_summary <- ranef(best_model)
print(ranef_summary)

# Fixed effects
cat("\\nFixed Effects:\\n")
fixed_effects <- tidy(best_model, effects = "fixed")
print(fixed_effects)

# Post-hoc analysis
cat("\\n=== POST-HOC ANALYSIS ===\\n")

# Pairwise comparisons between time points
emmeans_result <- emmeans(best_model, pairwise ~ !!sym(time_col))
cat("\\nPairwise comparisons between time points:\\n")
print(emmeans_result$contrasts)

# If groups available, test group differences
if(has_group) {{
  group_emmeans <- emmeans(best_model, pairwise ~ !!sym(group_col))
  cat("\\nGroup comparisons:\\n")
  print(group_emmeans$contrasts)
}}

# Model performance
cat("\\n=== MODEL PERFORMANCE ===\\n")
r2_marginal <- r2_nakagawa(best_model)$R2_marginal
r2_conditional <- r2_nakagawa(best_model)$R2_conditional

cat("Marginal R² (fixed effects):", round(r2_marginal, 3), "\\n")
cat("Conditional R² (fixed + random):", round(r2_conditional, 3), "\\n")

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• Sample size:", nrow(df), "observations from", length(unique(df[[subject_col]])), "subjects\\n")
cat("• Time points:", length(unique(df[[time_col]])), "\\n")
cat("• ICC:", round(icc_result, 4), "\\n")
cat("• Design:", ifelse(sd(subject_counts) == 0, "Balanced", "Unbalanced"), "\\n")
cat("• Best model AIC:", round(AIC(best_model), 2), "\\n")
cat("• Marginal R²:", round(r2_marginal, 3), "\\n")
cat("• Conditional R²:", round(r2_conditional, 3), "\\n")
'''
