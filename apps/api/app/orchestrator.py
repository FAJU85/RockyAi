"""
Rocky AI Orchestrator - Core chat to code generation system
Handles natural language queries → analysis planning → code generation → execution
Enhanced with caching, database integration, and comprehensive error handling
"""
import asyncio
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx
import os
from datetime import datetime

from apps.api.app.config import get_settings
from apps.api.app.logging_config import get_logger, generate_correlation_id
from apps.api.app.cache import get_cache
from apps.api.app.database import get_db, DatabaseService
from apps.api.app.executor_service import get_executor_service
from apps.api.app.websocket import get_connection_manager

# Configure logging
logger = get_logger(__name__)


class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    EXPLORATORY = "exploratory"


@dataclass
class AnalysisPlan:
    """Structured analysis plan generated from user query"""
    query: str
    analysis_type: AnalysisType
    language: str  # "python" or "r"
    steps: List[str]
    required_libraries: List[str]
    expected_outputs: List[str]
    data_requirements: Dict[str, Any]


@dataclass
class CodeResult:
    """Result of code execution"""
    code: str
    language: str
    output: str
    error: Optional[str]
    execution_time: float
    success: bool


class RockyOrchestrator:
    """Enhanced orchestrator for Rocky AI research assistant"""
    
    def __init__(self, dmr_base_url: str = None):
        self.settings = get_settings()
        self.dmr_base_url = dmr_base_url or self.settings.dmr.base_url
        self.session = httpx.AsyncClient(timeout=self.settings.dmr.timeout)
        
        # Initialize services
        self.cache = None
        self.db_service = None
        self.executor_service = None
        
        # Analysis templates and patterns
        self.analysis_patterns = self._load_analysis_patterns()
        self.library_mappings = self._load_library_mappings()
    
    async def initialize(self):
        """Initialize orchestrator with all services"""
        try:
            # Initialize cache
            self.cache = await get_cache()
            
            # Initialize database service
            db = next(get_db())
            self.db_service = DatabaseService(db)
            
        # Initialize executor service
        self.executor_service = await get_executor_service()
        
        # Initialize WebSocket connection manager
        self.websocket_manager = await get_connection_manager()
        
        logger.info("Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    def _load_analysis_patterns(self) -> Dict[str, Dict]:
        """Load analysis patterns for different types of queries"""
        return {
            "t_test": {
                "keywords": ["t-test", "t test", "compare means", "difference between groups"],
                "analysis_type": AnalysisType.INFERENTIAL,
                "libraries": ["scipy.stats", "statsmodels"],
                "r_libraries": ["t.test", "broom"]
            },
            "anova": {
                "keywords": ["anova", "analysis of variance", "multiple groups", "f-test"],
                "analysis_type": AnalysisType.INFERENTIAL,
                "libraries": ["scipy.stats", "statsmodels"],
                "r_libraries": ["aov", "car", "broom"]
            },
            "regression": {
                "keywords": ["regression", "predict", "correlation", "relationship"],
                "analysis_type": AnalysisType.PREDICTIVE,
                "libraries": ["sklearn", "statsmodels", "scipy.stats"],
                "r_libraries": ["lm", "glm", "broom", "tidymodels"]
            },
            "chi_square": {
                "keywords": ["chi-square", "chi square", "contingency", "categorical"],
                "analysis_type": AnalysisType.INFERENTIAL,
                "libraries": ["scipy.stats"],
                "r_libraries": ["chisq.test", "broom"]
            },
            "survival": {
                "keywords": ["survival", "kaplan-meier", "cox", "time to event"],
                "analysis_type": AnalysisType.INFERENTIAL,
                "libraries": ["lifelines"],
                "r_libraries": ["survival", "survminer"]
            },
            "pca": {
                "keywords": ["pca", "principal component", "dimensionality reduction"],
                "analysis_type": AnalysisType.EXPLORATORY,
                "libraries": ["sklearn", "scipy"],
                "r_libraries": ["prcomp", "factoextra"]
            }
        }
    
    def _load_library_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load library mappings between Python and R"""
        return {
            "pandas": "dplyr",
            "numpy": "base",
            "matplotlib": "ggplot2",
            "seaborn": "ggplot2",
            "scipy.stats": "stats",
            "sklearn": "tidymodels",
            "statsmodels": "broom"
        }
    
    async def analyze_query(self, query: str, context: Dict = None) -> AnalysisPlan:
        """Analyze user query and create analysis plan"""
        logger.info(f"Analyzing query: {query}")
        
        # Use AI to determine analysis type and requirements
        analysis_prompt = self._create_analysis_prompt(query, context)
        ai_response = await self._call_ai(analysis_prompt)
        
        # Parse AI response and create structured plan
        plan = self._parse_analysis_response(query, ai_response)
        
        # Enhance plan with pattern matching
        plan = self._enhance_plan_with_patterns(plan)
        
        logger.info(f"Generated plan: {plan.analysis_type.value} in {plan.language}")
        return plan
    
    def _create_analysis_prompt(self, query: str, context: Dict = None) -> str:
        """Create prompt for AI analysis planning"""
        context_str = ""
        if context:
            context_str = f"\n\nContext:\n{json.dumps(context, indent=2)}"
        
        return f"""You are Rocky AI, a research assistant that helps with statistical analysis and data science.

User Query: "{query}"{context_str}

Analyze this query and determine:
1. What type of analysis is needed (descriptive, inferential, predictive, exploratory)
2. Which programming language would be best (python or r)
3. What libraries/packages are needed
4. What are the main steps to perform this analysis
5. What outputs should be generated

Respond in JSON format:
{{
    "analysis_type": "descriptive|inferential|predictive|exploratory",
    "language": "python|r",
    "libraries": ["lib1", "lib2"],
    "steps": ["step1", "step2", "step3"],
    "expected_outputs": ["output1", "output2"],
    "data_requirements": {{
        "min_rows": 10,
        "required_columns": ["col1", "col2"],
        "data_types": {{"col1": "numeric", "col2": "categorical"}}
    }}
}}"""
    
    def _parse_analysis_response(self, query: str, ai_response: str) -> AnalysisPlan:
        """Parse AI response into structured AnalysisPlan"""
        try:
            # Extract JSON from AI response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                response_data = json.loads(ai_response[json_start:json_end])
            else:
                raise ValueError("No valid JSON found in response")
            
            return AnalysisPlan(
                query=query,
                analysis_type=AnalysisType(response_data.get("analysis_type", "descriptive")),
                language=response_data.get("language", "python"),
                steps=response_data.get("steps", []),
                required_libraries=response_data.get("libraries", []),
                expected_outputs=response_data.get("expected_outputs", []),
                data_requirements=response_data.get("data_requirements", {})
            )
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            # Fallback to basic plan
            return AnalysisPlan(
                query=query,
                analysis_type=AnalysisType.DESCRIPTIVE,
                language="python",
                steps=["Load data", "Perform analysis", "Generate results"],
                required_libraries=["pandas", "numpy"],
                expected_outputs=["Summary statistics", "Visualization"],
                data_requirements={}
            )
    
    def _enhance_plan_with_patterns(self, plan: AnalysisPlan) -> AnalysisPlan:
        """Enhance plan using pattern matching"""
        query_lower = plan.query.lower()
        
        for pattern_name, pattern_info in self.analysis_patterns.items():
            if any(keyword in query_lower for keyword in pattern_info["keywords"]):
                # Update analysis type
                plan.analysis_type = pattern_info["analysis_type"]
                
                # Add specific libraries
                if plan.language == "python":
                    plan.required_libraries.extend(pattern_info["libraries"])
                else:  # R
                    plan.required_libraries.extend(pattern_info["r_libraries"])
                
                break
        
        return plan
    
    async def generate_code(self, plan: AnalysisPlan, data_info: Dict = None) -> str:
        """Generate executable code based on analysis plan"""
        logger.info(f"Generating {plan.language} code for {plan.analysis_type.value} analysis")
        
        code_prompt = self._create_code_prompt(plan, data_info)
        ai_response = await self._call_ai(code_prompt)
        
        # Extract code from AI response
        code = self._extract_code_from_response(ai_response, plan.language)
        
        return code
    
    def _create_code_prompt(self, plan: AnalysisPlan, data_info: Dict = None) -> str:
        """Create prompt for code generation"""
        data_context = ""
        if data_info:
            data_context = f"\n\nData Information:\n{json.dumps(data_info, indent=2)}"
        
        return f"""You are Rocky AI, a research assistant. Generate {plan.language.upper()} code for the following analysis:

Query: {plan.query}
Analysis Type: {plan.analysis_type.value}
Language: {plan.language}
Required Libraries: {', '.join(plan.required_libraries)}
Steps: {plan.steps}
Expected Outputs: {plan.expected_outputs}{data_context}

Requirements:
1. Use only the specified libraries
2. Include data loading/import statements
3. Add proper error handling
4. Include comments explaining each step
5. Generate visualizations where appropriate
6. Return only the executable code, no explanations

Generate clean, production-ready {plan.language} code:"""
    
    def _extract_code_from_response(self, response: str, language: str) -> str:
        """Extract code from AI response"""
        # Look for code blocks
        if language == "python":
            code_blocks = []
            lines = response.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.strip().startswith('```python') or line.strip().startswith('```'):
                    if in_code_block:
                        code_blocks.append('\n'.join(current_block))
                        current_block = []
                    in_code_block = not in_code_block
                elif in_code_block and not line.strip().startswith('```'):
                    current_block.append(line)
            
            if current_block:
                code_blocks.append('\n'.join(current_block))
            
            return '\n\n'.join(code_blocks) if code_blocks else response
        
        else:  # R
            code_blocks = []
            lines = response.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.strip().startswith('```r') or line.strip().startswith('```'):
                    if in_code_block:
                        code_blocks.append('\n'.join(current_block))
                        current_block = []
                    in_code_block = not in_code_block
                elif in_code_block and not line.strip().startswith('```'):
                    current_block.append(line)
            
            if current_block:
                code_blocks.append('\n'.join(current_block))
            
            return '\n\n'.join(code_blocks) if code_blocks else response
    
    async def _call_ai(self, prompt: str) -> str:
        """Call the AI model via DMR"""
        payload = {
            "model": "ai/llama2",  # Will be dynamically selected
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = await self.session.post(
                f"{self.dmr_base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"AI call failed: {e}")
            raise
    
    async def execute_analysis(self, query: str, data_path: str = None, 
                             context: Dict = None, user_id: str = None,
                             dataset_id: str = None) -> Dict[str, Any]:
        """Enhanced method to execute complete analysis pipeline"""
        correlation_id = generate_correlation_id()
        logger.set_correlation_id(correlation_id)
        
        try:
            logger.log_analysis_start(query, user_id)
            start_time = datetime.now()
            
            # Check cache first
            if self.cache and self.settings.enable_caching:
                cache_key = f"analysis:{hashlib.sha256(f'{query}:{data_path}'.encode()).hexdigest()}"
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Using cached analysis result for query: {query[:50]}...")
                    return cached_result
            
            # Create analysis record in database
            analysis_id = None
            if self.db_service and self.settings.enable_database:
                analysis = self.db_service.create_analysis(
                    query=query,
                    analysis_type="pending",
                    language="python",  # Will be updated after planning
                    user_id=user_id,
                    dataset_id=dataset_id,
                    correlation_id=correlation_id
                )
                analysis_id = str(analysis.id)
            
            # Send real-time update for analysis start
            if self.websocket_manager and analysis_id:
                await self.websocket_manager.send_analysis_update(
                    analysis_id, "start", {
                        "query": query,
                        "status": "pending",
                        "progress": 0,
                        "currentStep": "Initializing analysis..."
                    }
                )
            
            # Step 1: Analyze query and create plan
            plan = await self.analyze_query(query, context)
            
            # Send real-time update for planning completion
            if self.websocket_manager and analysis_id:
                await self.websocket_manager.send_analysis_update(
                    analysis_id, "progress", {
                        "status": "running",
                        "progress": 25,
                        "currentStep": f"Planning {plan.analysis_type.value} analysis in {plan.language}",
                        "estimatedTimeRemaining": 30
                    }
                )
            
            # Update analysis record with plan
            if self.db_service and analysis_id:
                self.db_service.update_analysis(
                    analysis_id,
                    analysis_type=plan.analysis_type.value,
                    language=plan.language,
                    plan={
                        "analysis_type": plan.analysis_type.value,
                        "language": plan.language,
                        "steps": plan.steps,
                        "libraries": plan.required_libraries,
                        "expected_outputs": plan.expected_outputs
                    }
                )
            
            # Step 2: Generate code
            data_info = {"path": data_path} if data_path else None
            code = await self.generate_code(plan, data_info)
            
            # Send real-time update for code generation
            if self.websocket_manager and analysis_id:
                await self.websocket_manager.send_analysis_update(
                    analysis_id, "progress", {
                        "status": "running",
                        "progress": 50,
                        "currentStep": "Generating analysis code",
                        "estimatedTimeRemaining": 20
                    }
                )
            
            # Update analysis record with code
            if self.db_service and analysis_id:
                self.db_service.update_analysis(analysis_id, code=code, status="running")
            
            # Step 3: Execute code
            execution_result = None
            if self.executor_service and self.settings.enable_executors:
                # Send real-time update for execution start
                if self.websocket_manager and analysis_id:
                    await self.websocket_manager.send_analysis_update(
                        analysis_id, "progress", {
                            "status": "running",
                            "progress": 75,
                            "currentStep": f"Executing {plan.language} code",
                            "estimatedTimeRemaining": 10
                        }
                    )
                
                execution_result = await self.executor_service.execute_code(
                    code=code,
                    language=plan.language,
                    analysis_id=analysis_id,
                    correlation_id=correlation_id
                )
            
            # Prepare final result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "query": query,
                "analysis_id": analysis_id,
                "correlation_id": correlation_id,
                "plan": {
                    "analysis_type": plan.analysis_type.value,
                    "language": plan.language,
                    "steps": plan.steps,
                    "libraries": plan.required_libraries,
                    "expected_outputs": plan.expected_outputs
                },
                "code": code,
                "status": "completed" if execution_result and execution_result.get("success") else "generated",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add execution results if available
            if execution_result:
                result.update({
                    "output": execution_result.get("output", ""),
                    "error": execution_result.get("error", ""),
                    "execution_details": {
                        "success": execution_result.get("success", False),
                        "execution_time": execution_result.get("execution_time", 0),
                        "memory_used_mb": execution_result.get("memory_used_mb", 0),
                        "return_code": execution_result.get("return_code", 0)
                    }
                })
                
                # Update analysis record with execution results
                if self.db_service and analysis_id:
                    self.db_service.update_analysis(
                        analysis_id,
                        status="completed" if execution_result.get("success") else "failed",
                        output=execution_result.get("output", ""),
                        error=execution_result.get("error", ""),
                        execution_time=execution_result.get("execution_time", 0),
                        memory_used=execution_result.get("memory_used_mb", 0)
                    )
            
            # Send real-time update for completion
            if self.websocket_manager and analysis_id:
                await self.websocket_manager.send_analysis_update(
                    analysis_id, "complete", {
                        "status": result["status"],
                        "progress": 100,
                        "currentStep": "Analysis completed",
                        "executionTime": execution_time,
                        "success": result["status"] == "completed"
                    }
                )
            
            # Cache result
            if self.cache and self.settings.enable_caching:
                await self.cache.set(cache_key, result, ttl=self.settings.cache.analysis_ttl)
            
            # Log completion
            logger.log_analysis_complete(query, execution_time, result["status"] == "completed", user_id)
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.log_error(e, {
                "query": query,
                "user_id": user_id,
                "execution_time": execution_time
            })
            
            # Send real-time update for error
            if self.websocket_manager and analysis_id:
                await self.websocket_manager.send_analysis_update(
                    analysis_id, "error", {
                        "status": "failed",
                        "progress": 0,
                        "currentStep": "Analysis failed",
                        "error": str(e)
                    }
                )
            
            # Update analysis record with error
            if self.db_service and analysis_id:
                self.db_service.update_analysis(
                    analysis_id,
                    status="failed",
                    error=str(e)
                )
            
            return {
                "query": query,
                "analysis_id": analysis_id,
                "correlation_id": correlation_id,
                "error": str(e),
                "status": "failed",
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Clean up resources"""
        await self.session.aclose()


# Global orchestrator instance
orchestrator = None

async def get_orchestrator() -> RockyOrchestrator:
    """Get or create orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = RockyOrchestrator()
        await orchestrator.initialize()
    return orchestrator

async def close_orchestrator():
    """Close orchestrator and cleanup resources"""
    global orchestrator
    if orchestrator:
        await orchestrator.close()
        orchestrator = None
