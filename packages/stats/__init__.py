"""
Rocky AI Statistical Analysis Package
Provides templates and utilities for common statistical analyses
"""

from .analysis_templates import (
    AnalysisTemplate,
    TTestTemplate,
    AnovaTemplate,
    RegressionTemplate,
    ANALYSIS_TEMPLATES,
    get_template,
    list_available_templates,
    generate_analysis_code
)

from .chi_square_template import ChiSquareTemplate
from .pca_template import PCATemplate
from .survival_template import SurvivalTemplate
from .mixed_models_template import MixedModelsTemplate

# Add new templates to registry
ANALYSIS_TEMPLATES.update({
    "chi_square": ChiSquareTemplate(),
    "pca": PCATemplate(),
    "survival": SurvivalTemplate(),
    "mixed_models": MixedModelsTemplate(),
})

__all__ = [
    "AnalysisTemplate",
    "TTestTemplate", 
    "AnovaTemplate",
    "RegressionTemplate",
    "ChiSquareTemplate",
    "PCATemplate",
    "SurvivalTemplate",
    "MixedModelsTemplate",
    "ANALYSIS_TEMPLATES",
    "get_template",
    "list_available_templates", 
    "generate_analysis_code"
]
