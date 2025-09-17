"""
Advanced Analytics Engine for Rocky AI
ML pipelines, custom templates, and advanced statistical methods
"""
import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from lifelines import KaplanMeierFitter, CoxPHFitter
from apps.api.app.logging_config import get_logger
from apps.api.app.metrics import get_metrics_collector

logger = get_logger(__name__)


class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    EXPLORATORY = "exploratory"
    TIME_SERIES = "time_series"
    SURVIVAL = "survival"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"


class MLAlgorithm(Enum):
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    PCA = "pca"
    ARIMA = "arima"
    COX_REGRESSION = "cox_regression"


@dataclass
class AnalysisConfig:
    """Configuration for advanced analytics"""
    analysis_type: AnalysisType
    algorithm: Optional[MLAlgorithm] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    test_size: float = 0.2
    random_state: int = 42
    n_clusters: int = 3
    max_features: int = 10
    cv_folds: int = 5
    hyperparameters: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """Result of advanced analytics"""
    analysis_type: str
    algorithm: str
    metrics: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    model_info: Optional[Dict[str, Any]] = None
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    recommendations: List[str] = []


class AdvancedAnalyticsEngine:
    """Advanced analytics engine with ML capabilities"""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    async def analyze_data(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform advanced analytics on the dataset"""
        try:
            logger.info(f"Starting {config.analysis_type.value} analysis with {config.algorithm}")
            
            # Validate data
            self._validate_data(data, config)
            
            # Preprocess data
            processed_data = await self._preprocess_data(data, config)
            
            # Perform analysis based on type
            if config.analysis_type == AnalysisType.DESCRIPTIVE:
                result = await self._descriptive_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.INFERENTIAL:
                result = await self._inferential_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.PREDICTIVE:
                result = await self._predictive_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.EXPLORATORY:
                result = await self._exploratory_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.TIME_SERIES:
                result = await self._time_series_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.SURVIVAL:
                result = await self._survival_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.CLUSTERING:
                result = await self._clustering_analysis(processed_data, config)
            elif config.analysis_type == AnalysisType.DIMENSIONALITY_REDUCTION:
                result = await self._dimensionality_reduction_analysis(processed_data, config)
            else:
                raise ValueError(f"Unsupported analysis type: {config.analysis_type}")
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(result, config)
            
            logger.info(f"Completed {config.analysis_type.value} analysis successfully")
            return result
            
        except Exception as e:
            logger.error(f"Advanced analytics failed: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame, config: AnalysisConfig):
        """Validate data for analysis"""
        if data.empty:
            raise ValueError("Dataset is empty")
        
        if config.target_column and config.target_column not in data.columns:
            raise ValueError(f"Target column '{config.target_column}' not found in dataset")
        
        if config.feature_columns:
            missing_cols = [col for col in config.feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found: {missing_cols}")
    
    async def _preprocess_data(self, data: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        """Preprocess data for analysis"""
        processed_data = data.copy()
        
        # Handle missing values
        if processed_data.isnull().any().any():
            # For numerical columns, fill with median
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                processed_data[numeric_cols].median()
            )
            
            # For categorical columns, fill with mode
            categorical_cols = processed_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
        
        # Encode categorical variables
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != config.target_column:
                processed_data[col] = self.label_encoder.fit_transform(processed_data[col].astype(str))
        
        return processed_data
    
    async def _descriptive_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform descriptive analysis"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Basic statistics
        descriptive_stats = data[numeric_cols].describe()
        
        # Correlation analysis
        correlation_matrix = data[numeric_cols].corr()
        
        # Distribution analysis
        distributions = {}
        for col in numeric_cols:
            distributions[col] = {
                'skewness': stats.skew(data[col].dropna()),
                'kurtosis': stats.kurtosis(data[col].dropna()),
                'normality_p_value': stats.shapiro(data[col].dropna())[1] if len(data[col].dropna()) <= 5000 else None
            }
        
        # Generate visualizations
        visualizations = []
        
        # Correlation heatmap
        fig_corr = px.imshow(correlation_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Correlation Matrix")
        visualizations.append({
            'type': 'heatmap',
            'title': 'Correlation Matrix',
            'data': fig_corr.to_json()
        })
        
        # Distribution plots
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            fig_dist = px.histogram(data, x=col, title=f"Distribution of {col}")
            visualizations.append({
                'type': 'histogram',
                'title': f"Distribution of {col}",
                'data': fig_dist.to_json()
            })
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm="descriptive",
            metrics={
                'descriptive_stats': descriptive_stats.to_dict(),
                'correlation_matrix': correlation_matrix.to_dict(),
                'distributions': distributions
            },
            visualizations=visualizations
        )
    
    async def _inferential_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform inferential analysis"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        results = {}
        
        # T-tests for each numeric column
        for col in numeric_cols:
            if data[col].nunique() > 1:  # Skip if only one unique value
                # One-sample t-test
                t_stat, p_value = stats.ttest_1samp(data[col].dropna(), data[col].mean())
                results[f'{col}_one_sample_ttest'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        # Chi-square test for categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) >= 2:
            # Create contingency table for first two categorical variables
            col1, col2 = categorical_cols[:2]
            contingency_table = pd.crosstab(data[col1], data[col2])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            results['chi_square_test'] = {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm="inferential",
            metrics=results,
            visualizations=[]
        )
    
    async def _predictive_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform predictive analysis"""
        if not config.target_column:
            raise ValueError("Target column required for predictive analysis")
        
        # Prepare features and target
        if config.feature_columns:
            X = data[config.feature_columns]
        else:
            X = data.drop(columns=[config.target_column])
        
        y = data[config.target_column]
        
        # Determine if classification or regression
        is_classification = y.dtype == 'object' or y.nunique() < 10
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model based on algorithm
        if config.algorithm == MLAlgorithm.RANDOM_FOREST:
            if is_classification:
                model = RandomForestClassifier(random_state=config.random_state)
            else:
                model = RandomForestRegressor(random_state=config.random_state)
        elif config.algorithm == MLAlgorithm.LOGISTIC_REGRESSION:
            if not is_classification:
                raise ValueError("Logistic regression requires classification target")
            model = LogisticRegression(random_state=config.random_state)
        elif config.algorithm == MLAlgorithm.LINEAR_REGRESSION:
            if is_classification:
                raise ValueError("Linear regression requires continuous target")
            model = LinearRegression()
        else:
            raise ValueError(f"Unsupported algorithm for predictive analysis: {config.algorithm}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        if is_classification:
            accuracy = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=config.cv_folds)
            metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=config.cv_folds, scoring='r2')
            metrics = {
                'mse': mse,
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, model.coef_[0]))
        
        # Generate visualizations
        visualizations = []
        
        # Feature importance plot
        if feature_importance:
            fig_importance = px.bar(
                x=list(feature_importance.keys()),
                y=list(feature_importance.values()),
                title="Feature Importance"
            )
            visualizations.append({
                'type': 'bar',
                'title': 'Feature Importance',
                'data': fig_importance.to_json()
            })
        
        # Prediction vs actual plot
        if not is_classification:
            fig_pred = px.scatter(
                x=y_test,
                y=y_pred,
                title="Predicted vs Actual",
                labels={'x': 'Actual', 'y': 'Predicted'}
            )
            visualizations.append({
                'type': 'scatter',
                'title': 'Predicted vs Actual',
                'data': fig_pred.to_json()
            })
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm=config.algorithm.value,
            metrics=metrics,
            visualizations=visualizations,
            model_info={
                'model_type': type(model).__name__,
                'is_classification': is_classification,
                'n_features': X.shape[1],
                'n_samples': X.shape[0]
            },
            predictions=y_pred,
            feature_importance=feature_importance
        )
    
    async def _clustering_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform clustering analysis"""
        # Use numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found for clustering")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        # Perform clustering
        if config.algorithm == MLAlgorithm.KMEANS:
            model = KMeans(n_clusters=config.n_clusters, random_state=config.random_state)
            clusters = model.fit_predict(scaled_data)
        elif config.algorithm == MLAlgorithm.DBSCAN:
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(scaled_data)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {config.algorithm}")
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(scaled_data, clusters)
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters
        
        # Generate visualizations
        visualizations = []
        
        # PCA visualization if more than 2 dimensions
        if scaled_data.shape[1] > 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(scaled_data)
            
            fig_pca = px.scatter(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                color=clusters,
                title="Clusters (PCA Visualization)",
                labels={'x': 'PC1', 'y': 'PC2'}
            )
            visualizations.append({
                'type': 'scatter',
                'title': 'Clusters (PCA Visualization)',
                'data': fig_pca.to_json()
            })
        
        # Cluster size distribution
        cluster_counts = pd.Series(clusters).value_counts()
        fig_counts = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Cluster Size Distribution"
        )
        visualizations.append({
            'type': 'bar',
            'title': 'Cluster Size Distribution',
            'data': fig_counts.to_json()
        })
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm=config.algorithm.value,
            metrics={
                'silhouette_score': silhouette_avg,
                'n_clusters': len(set(clusters)),
                'cluster_sizes': cluster_counts.to_dict()
            },
            visualizations=visualizations,
            model_info={
                'model_type': type(model).__name__,
                'n_features': scaled_data.shape[1],
                'n_samples': scaled_data.shape[0]
            }
        )
    
    async def _time_series_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform time series analysis"""
        if not config.target_column:
            raise ValueError("Target column required for time series analysis")
        
        # Assume first column is time index if not specified
        time_col = data.columns[0] if not config.feature_columns else config.feature_columns[0]
        ts_data = data.set_index(time_col)[config.target_column].dropna()
        
        # Decompose time series
        decomposition = seasonal_decompose(ts_data, model='additive', period=12)
        
        # ARIMA model
        if config.algorithm == MLAlgorithm.ARIMA:
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=12)
            
            # Generate visualizations
            visualizations = []
            
            # Time series plot with forecast
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, name='Actual'))
            fig_ts.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name='Forecast'))
            fig_ts.update_layout(title="Time Series with Forecast")
            
            visualizations.append({
                'type': 'line',
                'title': 'Time Series with Forecast',
                'data': fig_ts.to_json()
            })
            
            # Decomposition plot
            fig_decomp = make_subplots(rows=4, cols=1, subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'])
            fig_decomp.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values, name='Original'), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal'), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual'), row=4, col=1)
            fig_decomp.update_layout(title="Time Series Decomposition")
            
            visualizations.append({
                'type': 'subplot',
                'title': 'Time Series Decomposition',
                'data': fig_decomp.to_json()
            })
            
            return AnalysisResult(
                analysis_type=config.analysis_type.value,
                algorithm=config.algorithm.value,
                metrics={
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'forecast_mean': forecast.mean(),
                    'forecast_std': forecast.std()
                },
                visualizations=visualizations,
                model_info={
                    'model_type': 'ARIMA',
                    'order': (1, 1, 1),
                    'n_observations': len(ts_data)
                }
            )
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm="time_series",
            metrics={},
            visualizations=[]
        )
    
    async def _survival_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform survival analysis"""
        # This would require specific survival analysis libraries
        # For now, return a basic implementation
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm="survival",
            metrics={},
            visualizations=[]
        )
    
    async def _exploratory_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform exploratory data analysis"""
        # Combine descriptive and inferential analysis
        descriptive_result = await self._descriptive_analysis(data, config)
        inferential_result = await self._inferential_analysis(data, config)
        
        # Combine results
        combined_metrics = {
            **descriptive_result.metrics,
            **inferential_result.metrics
        }
        
        combined_visualizations = descriptive_result.visualizations + inferential_result.visualizations
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm="exploratory",
            metrics=combined_metrics,
            visualizations=combined_visualizations
        )
    
    async def _dimensionality_reduction_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
        """Perform dimensionality reduction analysis"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric columns found for dimensionality reduction")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data)
        
        if config.algorithm == MLAlgorithm.PCA:
            # PCA
            pca = PCA(n_components=min(config.max_features, scaled_data.shape[1]))
            pca_data = pca.fit_transform(scaled_data)
            
            # Explained variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Generate visualizations
            visualizations = []
            
            # Scree plot
            fig_scree = px.line(
                x=range(1, len(explained_variance_ratio) + 1),
                y=explained_variance_ratio,
                title="PCA Scree Plot",
                labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
            )
            visualizations.append({
                'type': 'line',
                'title': 'PCA Scree Plot',
                'data': fig_scree.to_json()
            })
            
            # Cumulative variance plot
            fig_cumvar = px.line(
                x=range(1, len(cumulative_variance) + 1),
                y=cumulative_variance,
                title="Cumulative Explained Variance",
                labels={'x': 'Principal Component', 'y': 'Cumulative Variance'}
            )
            visualizations.append({
                'type': 'line',
                'title': 'Cumulative Explained Variance',
                'data': fig_cumvar.to_json()
            })
            
            return AnalysisResult(
                analysis_type=config.analysis_type.value,
                algorithm=config.algorithm.value,
                metrics={
                    'explained_variance_ratio': explained_variance_ratio.tolist(),
                    'cumulative_variance': cumulative_variance.tolist(),
                    'n_components': pca.n_components_,
                    'total_variance_explained': cumulative_variance[-1]
                },
                visualizations=visualizations,
                model_info={
                    'model_type': 'PCA',
                    'n_features': scaled_data.shape[1],
                    'n_samples': scaled_data.shape[0]
                }
            )
        
        return AnalysisResult(
            analysis_type=config.analysis_type.value,
            algorithm="dimensionality_reduction",
            metrics={},
            visualizations=[]
        )
    
    def _generate_recommendations(self, result: AnalysisResult, config: AnalysisConfig) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if config.analysis_type == AnalysisType.PREDICTIVE:
            if 'accuracy' in result.metrics:
                accuracy = result.metrics['accuracy']
                if accuracy < 0.7:
                    recommendations.append("Model accuracy is low. Consider feature engineering or trying different algorithms.")
                elif accuracy > 0.9:
                    recommendations.append("Excellent model performance! Consider validating on unseen data.")
            
            if 'cv_std' in result.metrics:
                cv_std = result.metrics['cv_std']
                if cv_std > 0.1:
                    recommendations.append("High variance in cross-validation. Consider regularization or more data.")
        
        elif config.analysis_type == AnalysisType.CLUSTERING:
            if 'silhouette_score' in result.metrics:
                silhouette = result.metrics['silhouette_score']
                if silhouette < 0.3:
                    recommendations.append("Low silhouette score. Consider different number of clusters or algorithm.")
                elif silhouette > 0.7:
                    recommendations.append("Good cluster separation. Consider analyzing cluster characteristics.")
        
        elif config.analysis_type == AnalysisType.DIMENSIONALITY_REDUCTION:
            if 'total_variance_explained' in result.metrics:
                total_var = result.metrics['total_variance_explained']
                if total_var < 0.8:
                    recommendations.append("Consider increasing number of components to explain more variance.")
                elif total_var > 0.95:
                    recommendations.append("High variance explained. Consider reducing components for simplicity.")
        
        return recommendations


# Global analytics engine
analytics_engine = AdvancedAnalyticsEngine()


def get_analytics_engine() -> AdvancedAnalyticsEngine:
    """Get the global analytics engine"""
    return analytics_engine
