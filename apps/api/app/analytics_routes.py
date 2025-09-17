"""
Advanced Analytics API routes for Rocky AI
ML pipelines, custom templates, and advanced statistical methods
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import json
from apps.api.app.analytics_engine import (
    get_analytics_engine, 
    AnalysisConfig, 
    AnalysisType, 
    MLAlgorithm,
    AdvancedAnalyticsEngine
)
from apps.api.app.logging_config import get_logger
from apps.api.app.metrics import get_metrics_collector

logger = get_logger(__name__)
router = APIRouter()


class AnalyticsRequest(BaseModel):
    """Request model for advanced analytics"""
    data: List[Dict[str, Any]] = Field(..., description="Dataset as list of dictionaries")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    algorithm: Optional[str] = Field(None, description="ML algorithm to use")
    target_column: Optional[str] = Field(None, description="Target column for supervised learning")
    feature_columns: Optional[List[str]] = Field(None, description="Feature columns to use")
    test_size: float = Field(0.2, description="Test set size for train/test split")
    random_state: int = Field(42, description="Random state for reproducibility")
    n_clusters: int = Field(3, description="Number of clusters for clustering algorithms")
    max_features: int = Field(10, description="Maximum features for dimensionality reduction")
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Hyperparameters for the algorithm")


class AnalyticsResponse(BaseModel):
    """Response model for advanced analytics"""
    analysis_type: str
    algorithm: str
    metrics: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    model_info: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []
    execution_time: float
    success: bool
    error: Optional[str] = None


@router.post("/analytics/analyze", response_model=AnalyticsResponse)
async def analyze_data(
    request: AnalyticsRequest,
    background_tasks: BackgroundTasks,
    analytics_engine: AdvancedAnalyticsEngine = Depends(get_analytics_engine)
):
    """Perform advanced analytics on the provided dataset"""
    metrics_collector = get_metrics_collector()
    start_time = time.time()
    
    try:
        logger.info(f"Starting {request.analysis_type} analysis with {request.algorithm}")
        
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate analysis type
        try:
            analysis_type = AnalysisType(request.analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis type: {request.analysis_type}. "
                       f"Valid types: {[t.value for t in AnalysisType]}"
            )
        
        # Validate algorithm if provided
        algorithm = None
        if request.algorithm:
            try:
                algorithm = MLAlgorithm(request.algorithm)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid algorithm: {request.algorithm}. "
                           f"Valid algorithms: {[a.value for a in MLAlgorithm]}"
                )
        
        # Create analysis configuration
        config = AnalysisConfig(
            analysis_type=analysis_type,
            algorithm=algorithm,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            test_size=request.test_size,
            random_state=request.random_state,
            n_clusters=request.n_clusters,
            max_features=request.max_features,
            cv_folds=request.cv_folds,
            hyperparameters=request.hyperparameters
        )
        
        # Perform analysis
        result = await analytics_engine.analyze_data(df, config)
        
        execution_time = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_analysis_request(
            analysis_type=result.analysis_type,
            language="python",
            status="completed"
        )
        
        logger.info(f"Completed {request.analysis_type} analysis in {execution_time:.2f}s")
        
        return AnalyticsResponse(
            analysis_type=result.analysis_type,
            algorithm=result.algorithm,
            metrics=result.metrics,
            visualizations=result.visualizations,
            model_info=result.model_info,
            recommendations=result.recommendations,
            execution_time=execution_time,
            success=True
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Advanced analytics failed: {e}")
        
        # Record error metrics
        metrics_collector.record_analysis_error(
            error_type=type(e).__name__,
            analysis_type=request.analysis_type
        )
        
        return AnalyticsResponse(
            analysis_type=request.analysis_type,
            algorithm=request.algorithm or "unknown",
            metrics={},
            visualizations=[],
            recommendations=[],
            execution_time=execution_time,
            success=False,
            error=str(e)
        )


@router.get("/analytics/algorithms")
async def get_available_algorithms():
    """Get list of available algorithms for each analysis type"""
    algorithms_by_type = {
        analysis_type.value: [
            {
                "value": algorithm.value,
                "name": algorithm.name.replace("_", " ").title(),
                "description": _get_algorithm_description(algorithm)
            }
            for algorithm in MLAlgorithm
            if _is_algorithm_suitable_for_analysis_type(algorithm, analysis_type)
        ]
        for analysis_type in AnalysisType
    }
    
    return {
        "analysis_types": [
            {
                "value": analysis_type.value,
                "name": analysis_type.name.replace("_", " ").title(),
                "description": _get_analysis_type_description(analysis_type)
            }
            for analysis_type in AnalysisType
        ],
        "algorithms_by_type": algorithms_by_type
    }


@router.get("/analytics/templates")
async def get_analysis_templates():
    """Get predefined analysis templates"""
    templates = [
        {
            "id": "customer_segmentation",
            "name": "Customer Segmentation",
            "description": "Segment customers using clustering analysis",
            "analysis_type": "clustering",
            "algorithm": "kmeans",
            "required_columns": ["age", "income", "spending_score"],
            "optional_columns": ["gender", "education"],
            "parameters": {
                "n_clusters": 5,
                "max_features": 10
            }
        },
        {
            "id": "sales_forecasting",
            "name": "Sales Forecasting",
            "description": "Forecast sales using time series analysis",
            "analysis_type": "time_series",
            "algorithm": "arima",
            "required_columns": ["date", "sales"],
            "optional_columns": ["product", "region"],
            "parameters": {
                "max_features": 5
            }
        },
        {
            "id": "churn_prediction",
            "name": "Churn Prediction",
            "description": "Predict customer churn using machine learning",
            "analysis_type": "predictive",
            "algorithm": "random_forest",
            "required_columns": ["churn"],
            "optional_columns": ["tenure", "monthly_charges", "total_charges"],
            "parameters": {
                "test_size": 0.2,
                "cv_folds": 5
            }
        },
        {
            "id": "market_basket_analysis",
            "name": "Market Basket Analysis",
            "description": "Analyze product associations and patterns",
            "analysis_type": "exploratory",
            "algorithm": "descriptive",
            "required_columns": ["transaction_id", "product"],
            "optional_columns": ["quantity", "price"],
            "parameters": {}
        },
        {
            "id": "feature_importance",
            "name": "Feature Importance Analysis",
            "description": "Identify most important features using dimensionality reduction",
            "analysis_type": "dimensionality_reduction",
            "algorithm": "pca",
            "required_columns": [],
            "optional_columns": [],
            "parameters": {
                "max_features": 10
            }
        }
    ]
    
    return {"templates": templates}


@router.post("/analytics/templates/{template_id}/execute")
async def execute_template(
    template_id: str,
    data: List[Dict[str, Any]],
    analytics_engine: AdvancedAnalyticsEngine = Depends(get_analytics_engine)
):
    """Execute a predefined analysis template"""
    # Get template
    templates_response = await get_analysis_templates()
    template = next((t for t in templates_response["templates"] if t["id"] == template_id), None)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Create request from template
    request = AnalyticsRequest(
        data=data,
        analysis_type=template["analysis_type"],
        algorithm=template["algorithm"],
        **template["parameters"]
    )
    
    # Execute analysis
    return await analyze_data(request, BackgroundTasks(), analytics_engine)


@router.get("/analytics/health")
async def analytics_health_check(analytics_engine: AdvancedAnalyticsEngine = Depends(get_analytics_engine)):
    """Health check for analytics engine"""
    try:
        # Test with simple data
        test_data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        config = AnalysisConfig(
            analysis_type=AnalysisType.DESCRIPTIVE,
            algorithm=None
        )
        
        result = await analytics_engine.analyze_data(test_data, config)
        
        return {
            "status": "healthy",
            "analytics_engine": "operational",
            "test_analysis": "completed",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Analytics health check failed: {e}")
        return {
            "status": "unhealthy",
            "analytics_engine": "error",
            "error": str(e),
            "timestamp": time.time()
        }


def _get_algorithm_description(algorithm: MLAlgorithm) -> str:
    """Get description for an algorithm"""
    descriptions = {
        MLAlgorithm.RANDOM_FOREST: "Ensemble method for classification and regression",
        MLAlgorithm.LOGISTIC_REGRESSION: "Linear model for binary classification",
        MLAlgorithm.LINEAR_REGRESSION: "Linear model for regression",
        MLAlgorithm.KMEANS: "K-means clustering algorithm",
        MLAlgorithm.DBSCAN: "Density-based clustering algorithm",
        MLAlgorithm.PCA: "Principal Component Analysis for dimensionality reduction",
        MLAlgorithm.ARIMA: "AutoRegressive Integrated Moving Average for time series",
        MLAlgorithm.COX_REGRESSION: "Cox proportional hazards model for survival analysis"
    }
    return descriptions.get(algorithm, "Machine learning algorithm")


def _get_analysis_type_description(analysis_type: AnalysisType) -> str:
    """Get description for an analysis type"""
    descriptions = {
        AnalysisType.DESCRIPTIVE: "Basic statistical summaries and distributions",
        AnalysisType.INFERENTIAL: "Statistical hypothesis testing and inference",
        AnalysisType.PREDICTIVE: "Machine learning for prediction and classification",
        AnalysisType.EXPLORATORY: "Comprehensive data exploration and visualization",
        AnalysisType.TIME_SERIES: "Time series analysis and forecasting",
        AnalysisType.SURVIVAL: "Survival analysis and time-to-event modeling",
        AnalysisType.CLUSTERING: "Unsupervised learning for data grouping",
        AnalysisType.DIMENSIONALITY_REDUCTION: "Feature reduction and selection"
    }
    return descriptions.get(analysis_type, "Data analysis type")


def _is_algorithm_suitable_for_analysis_type(algorithm: MLAlgorithm, analysis_type: AnalysisType) -> bool:
    """Check if an algorithm is suitable for an analysis type"""
    suitability = {
        AnalysisType.PREDICTIVE: [
            MLAlgorithm.RANDOM_FOREST,
            MLAlgorithm.LOGISTIC_REGRESSION,
            MLAlgorithm.LINEAR_REGRESSION
        ],
        AnalysisType.CLUSTERING: [
            MLAlgorithm.KMEANS,
            MLAlgorithm.DBSCAN
        ],
        AnalysisType.DIMENSIONALITY_REDUCTION: [
            MLAlgorithm.PCA
        ],
        AnalysisType.TIME_SERIES: [
            MLAlgorithm.ARIMA
        ],
        AnalysisType.SURVIVAL: [
            MLAlgorithm.COX_REGRESSION
        ],
        AnalysisType.DESCRIPTIVE: [],
        AnalysisType.INFERENTIAL: [],
        AnalysisType.EXPLORATORY: []
    }
    
    return algorithm in suitability.get(analysis_type, [])
