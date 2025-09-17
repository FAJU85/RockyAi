"""
Statistical Analysis Templates for Rocky AI
Provides code templates for common statistical analyses in Python and R
"""
from typing import Dict, List, Any
from enum import Enum


class AnalysisTemplate:
    """Base class for analysis templates"""
    
    def __init__(self, name: str, description: str, keywords: List[str]):
        self.name = name
        self.description = description
        self.keywords = keywords
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        """Generate Python code for this analysis"""
        raise NotImplementedError
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        """Generate R code for this analysis"""
        raise NotImplementedError


class TTestTemplate(AnalysisTemplate):
    """T-test analysis template"""
    
    def __init__(self):
        super().__init__(
            name="t_test",
            description="Independent samples t-test",
            keywords=["t-test", "t test", "compare means", "difference between groups"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
from scipy import stats
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

# Assuming columns are: {data_info.get('columns', ['group', 'value'])}
group_col = "{data_info.get('group_column', 'group')}"
value_col = "{data_info.get('value_column', 'value')}"

# Get groups
groups = df[group_col].unique()
print(f"\\nGroups: {{groups}}")

# Check assumptions
print("\\n=== ASSUMPTION CHECKS ===")

# Normality test for each group
for group in groups:
    group_data = df[df[group_col] == group][value_col].dropna()
    stat, p_value = stats.shapiro(group_data)
    print(f"Group {{group}} - Shapiro-Wilk test: p = {{p_value:.4f}}")
    if p_value < 0.05:
        print("  → Data may not be normally distributed")
    else:
        print("  → Data appears normally distributed")

# Levene's test for equal variances
group1_data = df[df[group_col] == groups[0]][value_col].dropna()
group2_data = df[df[group_col] == groups[1]][value_col].dropna()
levene_stat, levene_p = stats.levene(group1_data, group2_data)
print(f"\\nLevene's test for equal variances: p = {{levene_p:.4f}}")
if levene_p < 0.05:
    print("  → Variances may not be equal (use Welch's t-test)")
else:
    print("  → Variances appear equal")

# Perform t-test
print("\\n=== T-TEST RESULTS ===")

# Independent samples t-test
if levene_p < 0.05:
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    test_type = "Welch's t-test (unequal variances)"
else:
    # Standard t-test (equal variances)
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)
    test_type = "Independent samples t-test (equal variances)"

print(f"Test: {{test_type}}")
print(f"t-statistic: {{t_stat:.4f}}")
print(f"p-value: {{p_value:.4f}}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                     (len(group2_data) - 1) * group2_data.var()) / 
                    (len(group1_data) + len(group2_data) - 2))
cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
print(f"Cohen's d: {{cohens_d:.4f}}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print(f"\\nResult: Significant difference between groups (p < {{alpha}})")
    if cohens_d < 0.2:
        effect_size = "small"
    elif cohens_d < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size: {{effect_size}} (Cohen's d = {{cohens_d:.4f}})")
else:
    print(f"\\nResult: No significant difference between groups (p ≥ {{alpha}})")

# Descriptive statistics
print("\\n=== DESCRIPTIVE STATISTICS ===")
desc_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
print(desc_stats)

# Visualization
plt.figure(figsize=(12, 5))

# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x=group_col, y=value_col)
plt.title("Box Plot by Group")
plt.ylabel(value_col)

# Histogram
plt.subplot(1, 2, 2)
for group in groups:
    group_data = df[df[group_col] == group][value_col].dropna()
    plt.hist(group_data, alpha=0.7, label=f"Group {{group}}", bins=20)
plt.xlabel(value_col)
plt.ylabel("Frequency")
plt.title("Distribution by Group")
plt.legend()

plt.tight_layout()
plt.show()

# Summary
print("\\n=== SUMMARY ===")
print(f"• Test: {{test_type}}")
print(f"• t-statistic: {{t_stat:.4f}}")
print(f"• p-value: {{p_value:.4f}}")
print(f"• Effect size (Cohen's d): {{cohens_d:.4f}}")
print(f"• Conclusion: {{'Significant' if p_value < alpha else 'Not significant'}} difference between groups")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(broom)
library(car)

# Load data
df <- read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
cat("Dataset dimensions:", dim(df), "\\n")
cat("\\nFirst few rows:\\n")
print(head(df))

# Check for missing values
cat("\\nMissing values:\\n")
print(colSums(is.na(df)))

# Assuming columns are: {data_info.get('columns', ['group', 'value'])}
group_col <- "{data_info.get('group_column', 'group')}"
value_col <- "{data_info.get('value_column', 'value')}"

# Get groups
groups <- unique(df[[group_col]])
cat("\\nGroups:", groups, "\\n")

# Check assumptions
cat("\\n=== ASSUMPTION CHECKS ===\\n")

# Normality test for each group
for(group in groups) {{
  group_data <- df %>% 
    filter(!!sym(group_col) == group) %>% 
    pull(!!sym(value_col)) %>% 
    na.omit()
  
  shapiro_test <- shapiro.test(group_data)
  cat("Group", group, "- Shapiro-Wilk test: p =", shapiro_test$p.value, "\\n")
  
  if(shapiro_test$p.value < 0.05) {{
    cat("  → Data may not be normally distributed\\n")
  }} else {{
    cat("  → Data appears normally distributed\\n")
  }}
}}

# Levene's test for equal variances
group1_data <- df %>% 
  filter(!!sym(group_col) == groups[1]) %>% 
  pull(!!sym(value_col)) %>% 
  na.omit()

group2_data <- df %>% 
  filter(!!sym(group_col) == groups[2]) %>% 
  pull(!!sym(value_col)) %>% 
  na.omit()

levene_test <- leveneTest(!!sym(value_col) ~ !!sym(group_col), data = df)
cat("\\nLevene's test for equal variances: p =", levene_test$`Pr(>F)`[1], "\\n")

if(levene_test$`Pr(>F)`[1] < 0.05) {{
  cat("  → Variances may not be equal (use Welch's t-test)\\n")
}} else {{
  cat("  → Variances appear equal\\n")
}}

# Perform t-test
cat("\\n=== T-TEST RESULTS ===\\n")

# Independent samples t-test
if(levene_test$`Pr(>F)`[1] < 0.05) {{
  # Welch's t-test (unequal variances)
  t_test <- t.test(!!sym(value_col) ~ !!sym(group_col), data = df, var.equal = FALSE)
  test_type <- "Welch's t-test (unequal variances)"
}} else {{
  # Standard t-test (equal variances)
  t_test <- t.test(!!sym(value_col) ~ !!sym(group_col), data = df, var.equal = TRUE)
  test_type <- "Independent samples t-test (equal variances)"
}}

cat("Test:", test_type, "\\n")
cat("t-statistic:", t_test$statistic, "\\n")
cat("p-value:", t_test$p.value, "\\n")

# Effect size (Cohen's d)
pooled_sd <- sqrt(((length(group1_data) - 1) * var(group1_data) + 
                  (length(group2_data) - 1) * var(group2_data)) / 
                 (length(group1_data) + length(group2_data) - 2))
cohens_d <- (mean(group1_data) - mean(group2_data)) / pooled_sd
cat("Cohen's d:", cohens_d, "\\n")

# Interpretation
alpha <- 0.05
if(t_test$p.value < alpha) {{
  cat("\\nResult: Significant difference between groups (p <", alpha, ")\\n")
  if(cohens_d < 0.2) {{
    effect_size <- "small"
  }} else if(cohens_d < 0.8) {{
    effect_size <- "medium"
  }} else {{
    effect_size <- "large"
  }}
  cat("Effect size:", effect_size, "(Cohen's d =", cohens_d, ")\\n")
}} else {{
  cat("\\nResult: No significant difference between groups (p >=", alpha, ")\\n")
}}

# Descriptive statistics
cat("\\n=== DESCRIPTIVE STATISTICS ===\\n")
desc_stats <- df %>% 
  group_by(!!sym(group_col)) %>% 
  summarise(
    n = n(),
    mean = mean(!!sym(value_col), na.rm = TRUE),
    sd = sd(!!sym(value_col), na.rm = TRUE),
    min = min(!!sym(value_col), na.rm = TRUE),
    max = max(!!sym(value_col), na.rm = TRUE),
    .groups = 'drop'
  )
print(desc_stats)

# Visualization
p1 <- ggplot(df, aes(x = !!sym(group_col), y = !!sym(value_col))) +
  geom_boxplot() +
  labs(title = "Box Plot by Group", y = value_col) +
  theme_minimal()

p2 <- ggplot(df, aes(x = !!sym(value_col), fill = !!sym(group_col))) +
  geom_histogram(alpha = 0.7, bins = 20, position = "identity") +
  labs(title = "Distribution by Group", x = value_col, y = "Frequency") +
  theme_minimal()

# Display plots
print(p1)
print(p2)

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• Test:", test_type, "\\n")
cat("• t-statistic:", t_test$statistic, "\\n")
cat("• p-value:", t_test$p.value, "\\n")
cat("• Effect size (Cohen's d):", cohens_d, "\\n")
cat("• Conclusion:", ifelse(t_test$p.value < alpha, "Significant", "Not significant"), "difference between groups\\n")
'''


class AnovaTemplate(AnalysisTemplate):
    """ANOVA analysis template"""
    
    def __init__(self):
        super().__init__(
            name="anova",
            description="One-way Analysis of Variance",
            keywords=["anova", "analysis of variance", "multiple groups", "f-test"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# Load data
df = pd.read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
print("Dataset shape:", df.shape)
print("\\nFirst few rows:")
print(df.head())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Assuming columns are: {data_info.get('columns', ['group', 'value'])}
group_col = "{data_info.get('group_column', 'group')}"
value_col = "{data_info.get('value_column', 'value')}"

# Get groups
groups = df[group_col].unique()
print(f"\\nGroups: {{groups}}")
print(f"Number of groups: {{len(groups)}}")

# Check assumptions
print("\\n=== ASSUMPTION CHECKS ===")

# Normality test for each group
for group in groups:
    group_data = df[df[group_col] == group][value_col].dropna()
    stat, p_value = stats.shapiro(group_data)
    print(f"Group {{group}} - Shapiro-Wilk test: p = {{p_value:.4f}}")
    if p_value < 0.05:
        print("  → Data may not be normally distributed")
    else:
        print("  → Data appears normally distributed")

# Levene's test for equal variances
group_data_list = [df[df[group_col] == group][value_col].dropna().values for group in groups]
levene_stat, levene_p = stats.levene(*group_data_list)
print(f"\\nLevene's test for equal variances: p = {{levene_p:.4f}}")
if levene_p < 0.05:
    print("  → Variances may not be equal")
else:
    print("  → Variances appear equal")

# Perform ANOVA
print("\\n=== ANOVA RESULTS ===")

# One-way ANOVA using scipy
f_stat, p_value = f_oneway(*group_data_list)
print(f"F-statistic: {{f_stat:.4f}}")
print(f"p-value: {{p_value:.4f}}")

# Effect size (eta-squared)
ss_between = sum(len(group_data) * (group_data.mean() - df[value_col].mean())**2 
                for group_data in group_data_list)
ss_total = sum((df[value_col] - df[value_col].mean())**2)
eta_squared = ss_between / ss_total
print(f"Eta-squared (effect size): {{eta_squared:.4f}}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print(f"\\nResult: Significant difference between groups (p < {{alpha}})")
    if eta_squared < 0.01:
        effect_size = "negligible"
    elif eta_squared < 0.06:
        effect_size = "small"
    elif eta_squared < 0.14:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size: {{effect_size}} (eta-squared = {{eta_squared:.4f}})")
else:
    print(f"\\nResult: No significant difference between groups (p ≥ {{alpha}})")

# Post-hoc tests (if significant)
if p_value < alpha and len(groups) > 2:
    print("\\n=== POST-HOC TESTS ===")
    from scipy.stats import tukey_hsd
    tukey_result = tukey_hsd(*group_data_list)
    print("Tukey's HSD test:")
    print(tukey_result)

# Descriptive statistics
print("\\n=== DESCRIPTIVE STATISTICS ===")
desc_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
print(desc_stats)

# Visualization
plt.figure(figsize=(15, 5))

# Box plot
plt.subplot(1, 3, 1)
sns.boxplot(data=df, x=group_col, y=value_col)
plt.title("Box Plot by Group")
plt.xticks(rotation=45)
plt.ylabel(value_col)

# Violin plot
plt.subplot(1, 3, 2)
sns.violinplot(data=df, x=group_col, y=value_col)
plt.title("Violin Plot by Group")
plt.xticks(rotation=45)
plt.ylabel(value_col)

# Histogram by group
plt.subplot(1, 3, 3)
for group in groups:
    group_data = df[df[group_col] == group][value_col].dropna()
    plt.hist(group_data, alpha=0.7, label=f"Group {{group}}", bins=15)
plt.xlabel(value_col)
plt.ylabel("Frequency")
plt.title("Distribution by Group")
plt.legend()

plt.tight_layout()
plt.show()

# Summary
print("\\n=== SUMMARY ===")
print(f"• F-statistic: {{f_stat:.4f}}")
print(f"• p-value: {{p_value:.4f}}")
print(f"• Effect size (eta-squared): {{eta_squared:.4f}}")
print(f"• Conclusion: {{'Significant' if p_value < alpha else 'Not significant'}} difference between groups")
if p_value < alpha and len(groups) > 2:
    print("• Post-hoc tests recommended to identify which groups differ")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(broom)
library(car)

# Load data
df <- read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
cat("Dataset dimensions:", dim(df), "\\n")
cat("\\nFirst few rows:\\n")
print(head(df))

# Check for missing values
cat("\\nMissing values:\\n")
print(colSums(is.na(df)))

# Assuming columns are: {data_info.get('columns', ['group', 'value'])}
group_col <- "{data_info.get('group_column', 'group')}"
value_col <- "{data_info.get('value_column', 'value')}"

# Get groups
groups <- unique(df[[group_col]])
cat("\\nGroups:", groups, "\\n")
cat("Number of groups:", length(groups), "\\n")

# Check assumptions
cat("\\n=== ASSUMPTION CHECKS ===\\n")

# Normality test for each group
for(group in groups) {{
  group_data <- df %>% 
    filter(!!sym(group_col) == group) %>% 
    pull(!!sym(value_col)) %>% 
    na.omit()
  
  shapiro_test <- shapiro.test(group_data)
  cat("Group", group, "- Shapiro-Wilk test: p =", shapiro_test$p.value, "\\n")
  
  if(shapiro_test$p.value < 0.05) {{
    cat("  → Data may not be normally distributed\\n")
  }} else {{
    cat("  → Data appears normally distributed\\n")
  }}
}}

# Levene's test for equal variances
levene_test <- leveneTest(!!sym(value_col) ~ !!sym(group_col), data = df)
cat("\\nLevene's test for equal variances: p =", levene_test$`Pr(>F)`[1], "\\n")

if(levene_test$`Pr(>F)`[1] < 0.05) {{
  cat("  → Variances may not be equal\\n")
}} else {{
  cat("  → Variances appear equal\\n")
}}

# Perform ANOVA
cat("\\n=== ANOVA RESULTS ===\\n")

# One-way ANOVA
anova_model <- aov(!!sym(value_col) ~ !!sym(group_col), data = df)
anova_summary <- summary(anova_model)
print(anova_summary)

# Extract F-statistic and p-value
f_stat <- anova_summary[[1]]$`F value`[1]
p_value <- anova_summary[[1]]$`Pr(>F)`[1]

# Effect size (eta-squared)
ss_between <- anova_summary[[1]]$`Sum Sq`[1]
ss_total <- sum(anova_summary[[1]]$`Sum Sq`)
eta_squared <- ss_between / ss_total
cat("\\nEta-squared (effect size):", eta_squared, "\\n")

# Interpretation
alpha <- 0.05
if(p_value < alpha) {{
  cat("\\nResult: Significant difference between groups (p <", alpha, ")\\n")
  if(eta_squared < 0.01) {{
    effect_size <- "negligible"
  }} else if(eta_squared < 0.06) {{
    effect_size <- "small"
  }} else if(eta_squared < 0.14) {{
    effect_size <- "medium"
  }} else {{
    effect_size <- "large"
  }}
  cat("Effect size:", effect_size, "(eta-squared =", eta_squared, ")\\n")
}} else {{
  cat("\\nResult: No significant difference between groups (p >=", alpha, ")\\n")
}}

# Post-hoc tests (if significant)
if(p_value < alpha && length(groups) > 2) {{
  cat("\\n=== POST-HOC TESTS ===\\n")
  tukey_result <- TukeyHSD(anova_model)
  print(tukey_result)
}}

# Descriptive statistics
cat("\\n=== DESCRIPTIVE STATISTICS ===\\n")
desc_stats <- df %>% 
  group_by(!!sym(group_col)) %>% 
  summarise(
    n = n(),
    mean = mean(!!sym(value_col), na.rm = TRUE),
    sd = sd(!!sym(value_col), na.rm = TRUE),
    min = min(!!sym(value_col), na.rm = TRUE),
    max = max(!!sym(value_col), na.rm = TRUE),
    .groups = 'drop'
  )
print(desc_stats)

# Visualization
p1 <- ggplot(df, aes(x = !!sym(group_col), y = !!sym(value_col))) +
  geom_boxplot() +
  labs(title = "Box Plot by Group", y = value_col) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ggplot(df, aes(x = !!sym(group_col), y = !!sym(value_col))) +
  geom_violin() +
  labs(title = "Violin Plot by Group", y = value_col) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p3 <- ggplot(df, aes(x = !!sym(value_col), fill = !!sym(group_col))) +
  geom_histogram(alpha = 0.7, bins = 15, position = "identity") +
  labs(title = "Distribution by Group", x = value_col, y = "Frequency") +
  theme_minimal()

# Display plots
print(p1)
print(p2)
print(p3)

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• F-statistic:", f_stat, "\\n")
cat("• p-value:", p_value, "\\n")
cat("• Effect size (eta-squared):", eta_squared, "\\n")
cat("• Conclusion:", ifelse(p_value < alpha, "Significant", "Not significant"), "difference between groups\\n")
if(p_value < alpha && length(groups) > 2) {{
  cat("• Post-hoc tests recommended to identify which groups differ\\n")
}}
'''


class RegressionTemplate(AnalysisTemplate):
    """Linear regression analysis template"""
    
    def __init__(self):
        super().__init__(
            name="regression",
            description="Linear regression analysis",
            keywords=["regression", "predict", "correlation", "relationship", "linear"]
        )
    
    def get_python_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.api import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
print("Dataset shape:", df.shape)
print("\\nFirst few rows:")
print(df.head())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Assuming columns are: {data_info.get('columns', ['x', 'y'])}
x_col = "{data_info.get('x_column', 'x')}"
y_col = "{data_info.get('y_column', 'y')}"

# Prepare data
X = df[[x_col]].values
y = df[y_col].values

# Remove any missing values
mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X_clean = X[mask]
y_clean = y[mask]

print(f"\\nClean data points: {{len(X_clean)}}")

# Check assumptions
print("\\n=== ASSUMPTION CHECKS ===")

# 1. Linearity - scatter plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X_clean, y_clean, alpha=0.6)
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title("Scatter Plot - Linearity Check")

# 2. Normality of residuals (will check after fitting model)
# 3. Homoscedasticity (will check after fitting model)
# 4. Independence (assumed for now)

# Fit linear regression
print("\\n=== REGRESSION RESULTS ===")

# Using sklearn
model = LinearRegression()
model.fit(X_clean, y_clean)

# Predictions
y_pred = model.predict(X_clean)

# Calculate metrics
r2 = r2_score(y_clean, y_pred)
mse = mean_squared_error(y_clean, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared: {{r2:.4f}}")
print(f"RMSE: {{rmse:.4f}}")
print(f"Intercept: {{model.intercept_:.4f}}")
print(f"Slope: {{model.coef_[0]:.4f}}")

# Statistical significance using statsmodels
import statsmodels.api as sm
X_with_const = sm.add_constant(X_clean)
ols_model = OLS(y_clean, X_with_const).fit()
print("\\nDetailed statistics:")
print(ols_model.summary())

# Check residuals
residuals = y_clean - y_pred

# Normality of residuals
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"\\nResiduals normality (Shapiro-Wilk): p = {{shapiro_p:.4f}}")
if shapiro_p < 0.05:
    print("  → Residuals may not be normally distributed")
else:
    print("  → Residuals appear normally distributed")

# Homoscedasticity (Breusch-Pagan test)
try:
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"Homoscedasticity (Breusch-Pagan): p = {{bp_p:.4f}}")
    if bp_p < 0.05:
        print("  → Heteroscedasticity detected")
    else:
        print("  → Homoscedasticity assumption met")
except:
    print("  → Could not perform Breusch-Pagan test")

# Residual plots
plt.subplot(1, 3, 2)
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")

plt.subplot(1, 3, 3)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")

plt.tight_layout()
plt.show()

# Correlation
correlation = np.corrcoef(X_clean.flatten(), y_clean)[0, 1]
print(f"\\nCorrelation coefficient: {{correlation:.4f}}")

# Confidence intervals for coefficients
conf_int = ols_model.conf_int()
print("\\n95% Confidence Intervals:")
print(f"Intercept: [{{conf_int.iloc[0, 0]:.4f}}, {{conf_int.iloc[0, 1]:.4f}}]")
print(f"Slope: [{{conf_int.iloc[1, 0]:.4f}}, {{conf_int.iloc[1, 1]:.4f}}]")

# Prediction for new values (example)
new_x = np.array([[df[x_col].mean()], [df[x_col].mean() + df[x_col].std()]])
new_predictions = model.predict(new_x)
print(f"\\nPredictions for x = {{new_x[0, 0]:.2f}}: y = {{new_predictions[0]:.4f}}")
print(f"Predictions for x = {{new_x[1, 0]:.2f}}: y = {{new_predictions[1]:.4f}}")

# Summary
print("\\n=== SUMMARY ===")
print(f"• R-squared: {{r2:.4f}} ({{r2*100:.1f}}% of variance explained)")
print(f"• Correlation: {{correlation:.4f}}")
print(f"• RMSE: {{rmse:.4f}}")
print(f"• Equation: y = {{model.intercept_:.4f}} + {{model.coef_[0]:.4f}} * x")
print(f"• Slope p-value: {{ols_model.pvalues[1]:.4f}}")
if ols_model.pvalues[1] < 0.05:
    print("• Significant linear relationship detected")
else:
    print("• No significant linear relationship")
'''
    
    def get_r_code(self, data_info: Dict[str, Any]) -> str:
        return f'''
# Load required libraries
library(tidyverse)
library(broom)
library(car)

# Load data
df <- read_csv("{data_info.get('path', 'data.csv')}")

# Display basic info
cat("Dataset dimensions:", dim(df), "\\n")
cat("\\nFirst few rows:\\n")
print(head(df))

# Check for missing values
cat("\\nMissing values:\\n")
print(colSums(is.na(df)))

# Assuming columns are: {data_info.get('columns', ['x', 'y'])}
x_col <- "{data_info.get('x_column', 'x')}"
y_col <- "{data_info.get('y_column', 'y')}"

# Remove missing values
df_clean <- df %>% 
  filter(!is.na(!!sym(x_col)) & !is.na(!!sym(y_col)))

cat("\\nClean data points:", nrow(df_clean), "\\n")

# Check assumptions
cat("\\n=== ASSUMPTION CHECKS ===\\n")

# 1. Linearity - scatter plot
p1 <- ggplot(df_clean, aes(x = !!sym(x_col), y = !!sym(y_col))) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Scatter Plot - Linearity Check", x = x_col, y = y_col) +
  theme_minimal()

print(p1)

# Fit linear regression
cat("\\n=== REGRESSION RESULTS ===\\n")

# Linear regression model
model <- lm(!!sym(y_col) ~ !!sym(x_col), data = df_clean)

# Model summary
model_summary <- summary(model)
print(model_summary)

# Extract key metrics
r_squared <- model_summary$r.squared
adj_r_squared <- model_summary$adj.r.squared
f_stat <- model_summary$fstatistic[1]
f_p_value <- pf(f_stat, model_summary$fstatistic[2], model_summary$fstatistic[3], lower.tail = FALSE)

cat("\\nR-squared:", r_squared, "\\n")
cat("Adjusted R-squared:", adj_r_squared, "\\n")
cat("F-statistic:", f_stat, "p-value:", f_p_value, "\\n")

# Coefficients
coef_summary <- tidy(model)
print("\\nCoefficients:")
print(coef_summary)

# Check residuals
residuals <- residuals(model)
fitted_values <- fitted(model)

# Normality of residuals
shapiro_test <- shapiro.test(residuals)
cat("\\nResiduals normality (Shapiro-Wilk): p =", shapiro_test$p.value, "\\n")
if(shapiro_test$p.value < 0.05) {{
  cat("  → Residuals may not be normally distributed\\n")
}} else {{
  cat("  → Residuals appear normally distributed\\n")
}}

# Homoscedasticity (Breusch-Pagan test)
bp_test <- bptest(model)
cat("Homoscedasticity (Breusch-Pagan): p =", bp_test$p.value, "\\n")
if(bp_test$p.value < 0.05) {{
  cat("  → Heteroscedasticity detected\\n")
}} else {{
  cat("  → Homoscedasticity assumption met\\n")
}}

# Residual plots
p2 <- ggplot(data.frame(fitted = fitted_values, residuals = residuals), 
             aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linestyle = "dashed") +
  labs(title = "Residuals vs Fitted", x = "Fitted values", y = "Residuals") +
  theme_minimal()

p3 <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Q-Q Plot of Residuals") +
  theme_minimal()

print(p2)
print(p3)

# Correlation
correlation <- cor(df_clean[[x_col]], df_clean[[y_col]])
cat("\\nCorrelation coefficient:", correlation, "\\n")

# Confidence intervals
conf_int <- confint(model)
cat("\\n95% Confidence Intervals:\\n")
print(conf_int)

# Predictions for new values (example)
new_data <- data.frame(
  !!sym(x_col) := c(mean(df_clean[[x_col]]), mean(df_clean[[x_col]]) + sd(df_clean[[x_col]]))
)
predictions <- predict(model, new_data, interval = "confidence")
cat("\\nPredictions for new values:\\n")
print(cbind(new_data, predictions))

# Summary
cat("\\n=== SUMMARY ===\\n")
cat("• R-squared:", r_squared, "(", round(r_squared * 100, 1), "% of variance explained)\\n")
cat("• Correlation:", correlation, "\\n")
cat("• RMSE:", sqrt(mean(residuals^2)), "\\n")
cat("• Equation: y =", coef(model)[1], "+", coef(model)[2], "* x\\n")
cat("• Slope p-value:", coef_summary$p.value[2], "\\n")
if(coef_summary$p.value[2] < 0.05) {{
  cat("• Significant linear relationship detected\\n")
}} else {{
  cat("• No significant linear relationship\\n")
}}
'''


# Template registry
ANALYSIS_TEMPLATES = {
    "t_test": TTestTemplate(),
    "anova": AnovaTemplate(),
    "regression": RegressionTemplate(),
}


def get_template(analysis_type: str) -> AnalysisTemplate:
    """Get analysis template by type"""
    return ANALYSIS_TEMPLATES.get(analysis_type)


def list_available_templates() -> List[str]:
    """List all available analysis templates"""
    return list(ANALYSIS_TEMPLATES.keys())


def generate_analysis_code(analysis_type: str, language: str, data_info: Dict[str, Any]) -> str:
    """Generate analysis code for given type and language"""
    template = get_template(analysis_type)
    if not template:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    if language.lower() == "python":
        return template.get_python_code(data_info)
    elif language.lower() == "r":
        return template.get_r_code(data_info)
    else:
        raise ValueError(f"Unsupported language: {language}")
