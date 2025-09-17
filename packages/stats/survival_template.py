"""
Survival Analysis and Kaplan-Meier templates
"""
from .analysis_templates import AnalysisTemplate
from typing import Dict, Any


class SurvivalTemplate(AnalysisTemplate):
    """Survival analysis template"""
    
    def __init__(self):
        super().__init__(
            name="survival",
            description="Survival analysis with Kaplan-Meier curves and Cox regression",
            keywords=["survival", "kaplan-meier", "cox", "time to event", "hazard", "mortality"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
print("Dataset shape:", df.shape)
print("\\nFirst few rows:")
print(df.head())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Assuming columns are: {data_info.get('columns', ['time', 'event', 'group'])}
time_col = "{data_info.get('time_column', 'time')}"
event_col = "{data_info.get('event_column', 'event')}"
group_col = "{data_info.get('group_column', 'group')}" if "{data_info.get('group_column', 'group')}" in df.columns else None

# Check data types and values
print(f"\\nTime column ({time_col}) info:")
print(f"  Min: {{df[time_col].min()}}, Max: {{df[time_col].max()}}")
print(f"  Mean: {{df[time_col].mean():.2f}}, Median: {{df[time_col].median():.2f}}")

print(f"\\nEvent column ({event_col}) info:")
print(f"  Unique values: {{df[event_col].unique()}}")
print(f"  Event rate: {{df[event_col].mean():.3f}}")

if group_col:
    print(f"\\nGroup column ({group_col}) info:")
    print(f"  Groups: {{df[group_col].unique()}}")
    print(f"  Group sizes: {{df[group_col].value_counts().to_dict()}}")

# Data preparation
# Ensure event column is binary (0/1)
if df[event_col].nunique() > 2:
    print(f"\\nWarning: Event column has {{df[event_col].nunique()}} unique values. Converting to binary.")
    # Assume 0 = censored, anything else = event
    df[event_col] = (df[event_col] != 0).astype(int)

# Remove missing values
df_clean = df[[time_col, event_col]].dropna()
if group_col:
    df_clean[group_col] = df[group_col]

print(f"\\nClean data shape: {{df_clean.shape}}")

# Basic survival statistics
print("\\n=== SURVIVAL STATISTICS ===")

# Overall survival
kmf = KaplanMeierFitter()
kmf.fit(df_clean[time_col], df_clean[event_col], label='Overall')

print(f"\\nOverall survival:")
print(f"  Median survival time: {{kmf.median_survival_time_:.2f}}")
print(f"  Survival at 25th percentile: {{kmf.percentile(0.25):.3f}}")
print(f"  Survival at 75th percentile: {{kmf.percentile(0.75):.3f}}")

# Survival by groups (if available)
if group_col:
    print(f"\\nSurvival by {group_col}:")
    groups = df_clean[group_col].unique()
    
    # Kaplan-Meier curves for each group
    plt.figure(figsize=(12, 8))
    
    for i, group in enumerate(groups):
        group_data = df_clean[df_clean[group_col] == group]
        kmf_group = KaplanMeierFitter()
        kmf_group.fit(group_data[time_col], group_data[event_col], label=f'Group {group}')
        
        # Plot survival curve
        kmf_group.plot_survival_function(ax=plt.gca())
        
        # Print statistics
        median_time = kmf_group.median_survival_time_
        print(f"  Group {group}:")
        print(f"    Sample size: {{len(group_data)}}")
        print(f"    Events: {{group_data[event_col].sum()}}")
        print(f"    Median survival: {{median_time:.2f}}")
        print(f"    Survival at 25th percentile: {{kmf_group.percentile(0.25):.3f}}")
        print(f"    Survival at 75th percentile: {{kmf_group.percentile(0.75):.3f}}")
    
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Survival Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Log-rank test
    if len(groups) == 2:
        group1_data = df_clean[df_clean[group_col] == groups[0]]
        group2_data = df_clean[df_clean[group_col] == groups[1]]
        
        logrank_result = logrank_test(
            group1_data[time_col], group2_data[time_col],
            group1_data[event_col], group2_data[event_col]
        )
        
        print(f"\\n=== LOG-RANK TEST ===")
        print(f"Chi-square statistic: {{logrank_result.test_statistic:.4f}}")
        print(f"p-value: {{logrank_result.p_value:.4f}}")
        
        if logrank_result.p_value < 0.05:
            print("Result: Significant difference in survival between groups")
        else:
            print("Result: No significant difference in survival between groups")
    
    # Cox proportional hazards model
    print(f"\\n=== COX PROPORTIONAL HAZARDS MODEL ===")
    
    # Prepare data for Cox model
    cox_data = df_clean.copy()
    
    # Create dummy variables for categorical groups
    if group_col and cox_data[group_col].dtype == 'object':
        cox_data = pd.get_dummies(cox_data, columns=[group_col], prefix=group_col)
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col=time_col, event_col=event_col)
    
    print("\\nCox model summary:")
    print(cph.summary)
    
    # Hazard ratios
    print("\\nHazard Ratios (95% CI):")
    for var in cph.params.index:
        hr = np.exp(cph.params[var])
        ci_lower = np.exp(cph.confidence_intervals_.loc[var, 'lower_bound'])
        ci_upper = np.exp(cph.confidence_intervals_.loc[var, 'upper_bound'])
        print(f"  {var}: {{hr:.3f}} ({{ci_lower:.3f}} - {{ci_upper:.3f}})")
    
    # Model diagnostics
    print("\\n=== MODEL DIAGNOSTICS ===")
    
    # Proportional hazards assumption
    cph.check_assumptions(cox_data, p_value_threshold=0.05)
    
    # Concordance index
    concordance = cph.concordance_index_
    print(f"\\nConcordance index: {{concordance:.3f}}")
    if concordance > 0.7:
        print("  → Good model discrimination")
    elif concordance > 0.6:
        print("  → Moderate model discrimination")
    else:
        print("  → Poor model discrimination")
    
    # Plot hazard ratios
    plt.figure(figsize=(10, 6))
    cph.plot()
    plt.title('Hazard Ratios with 95% Confidence Intervals')
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.show()

else:
    # Single group analysis
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Survival Curve')
    plt.grid(True, alpha=0.3)
    plt.show()

# Risk table
print("\\n=== RISK TABLE ===")
if group_col:
    for group in groups:
        group_data = df_clean[df_clean[group_col] == group]
        kmf_group = KaplanMeierFitter()
        kmf_group.fit(group_data[time_col], group_data[event_col])
        
        print(f"\\nGroup {group}:")
        risk_table = kmf_group.event_table
        print(risk_table.head(10))
else:
    risk_table = kmf.event_table
    print(risk_table.head(10))

# Summary
print("\\n=== SUMMARY ===")
print(f"• Total sample size: {{len(df_clean)}}")
print(f"• Total events: {{df_clean[event_col].sum()}}")
print(f"• Event rate: {{df_clean[event_col].mean():.3f}}")
print(f"• Median survival time: {{kmf.median_survival_time_:.2f}}")

if group_col and len(groups) == 2:
    print(f"• Log-rank test p-value: {{logrank_result.p_value:.4f}}")
    print(f"• Cox model concordance: {{concordance:.3f}}")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(survival)
library(survminer)
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

# Assuming columns are: {data_info.get('columns', ['time', 'event', 'group'])}
time_col <- "{data_info.get('time_column', 'time')}"
event_col <- "{data_info.get('event_column', 'event')}"
group_col <- "{data_info.get('group_column', 'group')}"

# Check if group column exists
has_group <- group_col %in% names(df)
if(!has_group) group_col <- NULL

# Check data types and values
cat("\\nTime column (", time_col, ") info:\\n")
cat("  Min:", min(df[[time_col]], na.rm = TRUE), "Max:", max(df[[time_col]], na.rm = TRUE), "\\n")
cat("  Mean:", round(mean(df[[time_col]], na.rm = TRUE), 2), "Median:", round(median(df[[time_col]], na.rm = TRUE), 2), "\\n")

cat("\\nEvent column (", event_col, ") info:\\n")
cat("  Unique values:", unique(df[[event_col]]), "\\n")
cat("  Event rate:", round(mean(df[[event_col]], na.rm = TRUE), 3), "\\n")

if(has_group) {{
  cat("\\nGroup column (", group_col, ") info:\\n")
  cat("  Groups:", unique(df[[group_col]]), "\\n")
  cat("  Group sizes:", table(df[[group_col]]), "\\n")
}}

# Data preparation
# Ensure event column is binary (0/1)
if(length(unique(df[[event_col]])) > 2) {{
  cat("\\nWarning: Event column has", length(unique(df[[event_col]])), "unique values. Converting to binary.\\n")
  # Assume 0 = censored, anything else = event
  df[[event_col]] <- as.numeric(df[[event_col]] != 0)
}}

# Remove missing values
df_clean <- df %>% 
  select(all_of(c(time_col, event_col, if(has_group) group_col else NULL))) %>% 
  na.omit()

cat("\\nClean data dimensions:", dim(df_clean), "\\n")

# Create survival object
surv_obj <- Surv(df_clean[[time_col]], df_clean[[event_col]])

# Basic survival statistics
cat("\\n=== SURVIVAL STATISTICS ===\\n")

# Overall survival
km_fit <- survfit(surv_obj ~ 1, data = df_clean)
print(summary(km_fit))

# Median survival time
median_survival <- median(km_fit)
cat("\\nMedian survival time:", median_survival, "\\n")

# Survival by groups (if available)
if(has_group) {{
  cat("\\nSurvival by", group_col, ":\\n")
  groups <- unique(df_clean[[group_col]])
  
  # Kaplan-Meier curves for each group
  km_fit_groups <- survfit(surv_obj ~ df_clean[[group_col]], data = df_clean)
  
  # Plot survival curves
  p1 <- ggsurvplot(km_fit_groups, 
                   data = df_clean,
                   pval = TRUE,
                   conf.int = TRUE,
                   risk.table = TRUE,
                   risk.table.height = 0.3,
                   palette = "Set1",
                   title = "Kaplan-Meier Survival Curves",
                   xlab = "Time",
                   ylab = "Survival Probability")
  
  print(p1)
  
  # Print statistics for each group
  for(group in groups) {{
    group_data <- df_clean[df_clean[[group_col]] == group, ]
    group_surv <- Surv(group_data[[time_col]], group_data[[event_col]])
    group_km <- survfit(group_surv ~ 1)
    
    cat("\\nGroup", group, ":\\n")
    cat("  Sample size:", nrow(group_data), "\\n")
    cat("  Events:", sum(group_data[[event_col]]), "\\n")
    cat("  Median survival:", median(group_km), "\\n")
  }}
  
  # Log-rank test
  if(length(groups) == 2) {{
    logrank_test <- survdiff(surv_obj ~ df_clean[[group_col]], data = df_clean)
    
    cat("\\n=== LOG-RANK TEST ===\\n")
    print(logrank_test)
    
    p_value <- 1 - pchisq(logrank_test$chisq, length(logrank_test$n) - 1)
    cat("\\np-value:", p_value, "\\n")
    
    if(p_value < 0.05) {{
      cat("Result: Significant difference in survival between groups\\n")
    }} else {{
      cat("Result: No significant difference in survival between groups\\n")
    }}
  }}
  
  # Cox proportional hazards model
  cat("\\n=== COX PROPORTIONAL HAZARDS MODEL ===\\n")
  
  # Fit Cox model
  cox_model <- coxph(surv_obj ~ df_clean[[group_col]], data = df_clean)
  
  cat("\\nCox model summary:\\n")
  print(summary(cox_model))
  
  # Hazard ratios
  cat("\\nHazard Ratios (95% CI):\\n")
  cox_summary <- tidy(cox_model, exponentiate = TRUE, conf.int = TRUE)
  print(cox_summary)
  
  # Model diagnostics
  cat("\\n=== MODEL DIAGNOSTICS ===\\n")
  
  # Concordance index
  concordance <- cox_model$concordance["C"]
  cat("\\nConcordance index:", round(concordance, 3), "\\n")
  if(concordance > 0.7) {{
    cat("  → Good model discrimination\\n")
  }} else if(concordance > 0.6) {{
    cat("  → Moderate model discrimination\\n")
  }} else {{
    cat("  → Poor model discrimination\\n")
  }}
  
  # Plot hazard ratios
  cox_plot <- ggforest(cox_model, data = df_clean)
  print(cox_plot)
  
  # Check proportional hazards assumption
  ph_test <- cox.zph(cox_model)
  cat("\\nProportional Hazards Test:\\n")
  print(ph_test)
  
  if(ph_test$table[1, "p"] < 0.05) {{
    cat("Warning: Proportional hazards assumption may be violated\\n")
  }} else {{
    cat("Proportional hazards assumption appears to be met\\n")
  }}

}} else {{
  # Single group analysis
  p1 <- ggsurvplot(km_fit, 
                   data = df_clean,
                   conf.int = TRUE,
                   risk.table = TRUE,
                   risk.table.height = 0.3,
                   title = "Kaplan-Meier Survival Curve",
                   xlab = "Time",
                   ylab = "Survival Probability")
  
  print(p1)
}}

# Risk table
cat("\\n=== RISK TABLE ===\\n")
if(has_group) {{
  risk_table <- survfit(surv_obj ~ df_clean[[group_col]], data = df_clean)
  print(summary(risk_table))
}} else {{
  print(summary(km_fit))
}}

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• Total sample size:", nrow(df_clean), "\\n")
cat("• Total events:", sum(df_clean[[event_col]]), "\\n")
cat("• Event rate:", round(mean(df_clean[[event_col]]), 3), "\\n")
cat("• Median survival time:", median_survival, "\\n")

if(has_group && length(unique(df_clean[[group_col]])) == 2) {{
  cat("• Log-rank test p-value:", round(p_value, 4), "\\n")
  cat("• Cox model concordance:", round(concordance, 3), "\\n")
}}
'''
