"""
Dataset Ingestion Service for Rocky AI
Handles CSV, Parquet, Excel, and JSON data loading with schema inference
"""
import pandas as pd
import numpy as np
import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetIngestion:
    """Handle dataset loading, validation, and caching"""
    
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.supported_formats = ['.csv', '.parquet', '.xlsx', '.xls', '.json']
        
        # Schema cache
        self.schema_cache = {}
        self.load_schema_cache()
    
    def load_schema_cache(self):
        """Load cached schemas from disk"""
        cache_file = self.cache_dir / "schema_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.schema_cache = json.load(f)
                logger.info(f"Loaded {len(self.schema_cache)} cached schemas")
            except Exception as e:
                logger.warning(f"Failed to load schema cache: {e}")
                self.schema_cache = {}
    
    def save_schema_cache(self):
        """Save schemas to cache"""
        cache_file = self.cache_dir / "schema_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.schema_cache, f, indent=2)
            logger.info("Schema cache saved")
        except Exception as e:
            logger.warning(f"Failed to save schema cache: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file to detect changes"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Failed to generate file hash: {e}")
            return ""
    
    def load_dataset(self, file_path: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load dataset with automatic format detection and schema inference"""
        
        file_path = Path(file_path)
        
        # Validate file exists and is supported
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Generate file hash for caching
        file_hash = self.get_file_hash(str(file_path))
        
        # Check if we have cached schema
        cache_key = f"{file_path.name}_{file_hash}"
        if cache_key in self.schema_cache:
            logger.info(f"Using cached schema for {file_path.name}")
            schema = self.schema_cache[cache_key]
        else:
            # Infer schema
            schema = self._infer_schema(file_path, **kwargs)
            self.schema_cache[cache_key] = schema
            self.save_schema_cache()
        
        # Load data
        df = self._load_data(file_path, schema, **kwargs)
        
        # Validate data
        self._validate_data(df, schema)
        
        return df, schema
    
    def _infer_schema(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Infer dataset schema without loading full data"""
        
        logger.info(f"Inferring schema for {file_path.name}")
        
        schema = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'columns': [],
            'dtypes': {},
            'sample_data': {},
            'missing_values': {},
            'statistics': {},
            'inferred_at': datetime.now().isoformat()
        }
        
        try:
            # Load a small sample to infer schema
            sample_size = kwargs.get('sample_size', 1000)
            
            if file_path.suffix.lower() == '.csv':
                # For CSV, read first few rows to infer dtypes
                df_sample = pd.read_csv(file_path, nrows=sample_size, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                # For Parquet, read metadata
                df_sample = pd.read_parquet(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                # For Excel, read first sheet
                df_sample = pd.read_excel(file_path, nrows=sample_size, **kwargs)
            elif file_path.suffix.lower() == '.json':
                # For JSON, try to infer structure
                df_sample = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_path.suffix}")
            
            # Infer column information
            for col in df_sample.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df_sample[col].dtype),
                    'is_numeric': pd.api.types.is_numeric_dtype(df_sample[col]),
                    'is_categorical': pd.api.types.is_categorical_dtype(df_sample[col]),
                    'is_datetime': pd.api.types.is_datetime64_any_dtype(df_sample[col]),
                    'unique_count': df_sample[col].nunique(),
                    'missing_count': df_sample[col].isnull().sum(),
                    'sample_values': df_sample[col].dropna().head(5).tolist()
                }
                
                # Add statistics for numeric columns
                if col_info['is_numeric']:
                    col_info['statistics'] = {
                        'mean': df_sample[col].mean(),
                        'std': df_sample[col].std(),
                        'min': df_sample[col].min(),
                        'max': df_sample[col].max(),
                        'median': df_sample[col].median()
                    }
                
                schema['columns'].append(col_info)
                schema['dtypes'][col] = col_info['dtype']
                schema['sample_data'][col] = col_info['sample_values']
                schema['missing_values'][col] = col_info['missing_count']
                
                if col_info['is_numeric']:
                    schema['statistics'][col] = col_info['statistics']
            
            # Overall dataset info
            schema['total_rows'] = len(df_sample)
            schema['total_columns'] = len(df_sample.columns)
            schema['memory_usage'] = df_sample.memory_usage(deep=True).sum()
            
        except Exception as e:
            logger.error(f"Failed to infer schema: {e}")
            # Return basic schema
            schema['error'] = str(e)
        
        return schema
    
    def _load_data(self, file_path: Path, schema: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """Load full dataset using inferred schema"""
        
        logger.info(f"Loading full dataset: {file_path.name}")
        
        try:
            # Use inferred dtypes for better performance
            dtype_dict = schema.get('dtypes', {})
            
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, dtype=dtype_dict, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {file_path.suffix}")
            
            # Apply data type conversions based on schema
            for col_info in schema.get('columns', []):
                col_name = col_info['name']
                if col_name in df.columns:
                    if col_info['is_datetime']:
                        df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    elif col_info['is_categorical']:
                        df[col_name] = df[col_name].astype('category')
            
            logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame, schema: Dict[str, Any]):
        """Validate loaded data against schema"""
        
        logger.info("Validating dataset")
        
        # Check column count
        expected_cols = len(schema.get('columns', []))
        actual_cols = len(df.columns)
        
        if expected_cols != actual_cols:
            logger.warning(f"Column count mismatch: expected {expected_cols}, got {actual_cols}")
        
        # Check for missing values
        missing_info = schema.get('missing_values', {})
        for col, expected_missing in missing_info.items():
            if col in df.columns:
                actual_missing = df[col].isnull().sum()
                if actual_missing != expected_missing:
                    logger.warning(f"Missing values changed for {col}: expected {expected_missing}, got {actual_missing}")
        
        # Check data types
        for col_info in schema.get('columns', []):
            col_name = col_info['name']
            if col_name in df.columns:
                expected_dtype = col_info['dtype']
                actual_dtype = str(df[col_name].dtype)
                
                if expected_dtype != actual_dtype:
                    logger.warning(f"Data type changed for {col_name}: expected {expected_dtype}, got {actual_dtype}")
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_rows': df.nunique().to_dict()
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = info['numeric_columns']
        if numeric_cols:
            info['statistics'] = df[numeric_cols].describe().to_dict()
        
        return info
    
    def create_sample_data(self, 
                          n_rows: int = 100,
                          columns: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """Create sample dataset for testing"""
        
        if columns is None:
            columns = [
                {'name': 'id', 'type': 'int', 'min': 1, 'max': n_rows},
                {'name': 'group', 'type': 'categorical', 'values': ['A', 'B', 'C']},
                {'name': 'value', 'type': 'float', 'mean': 100, 'std': 15},
                {'name': 'score', 'type': 'float', 'mean': 75, 'std': 10},
                {'name': 'date', 'type': 'datetime', 'start': '2024-01-01', 'end': '2024-12-31'}
            ]
        
        data = {}
        
        for col in columns:
            col_name = col['name']
            col_type = col['type']
            
            if col_type == 'int':
                data[col_name] = np.random.randint(
                    col.get('min', 0), 
                    col.get('max', 100), 
                    n_rows
                )
            elif col_type == 'float':
                data[col_name] = np.random.normal(
                    col.get('mean', 0), 
                    col.get('std', 1), 
                    n_rows
                )
            elif col_type == 'categorical':
                values = col.get('values', ['A', 'B'])
                data[col_name] = np.random.choice(values, n_rows)
            elif col_type == 'datetime':
                start = pd.to_datetime(col.get('start', '2024-01-01'))
                end = pd.to_datetime(col.get('end', '2024-12-31'))
                data[col_name] = pd.date_range(start, end, periods=n_rows)
            else:
                data[col_name] = [f"sample_{i}" for i in range(n_rows)]
        
        return pd.DataFrame(data)
    
    def export_dataset(self, 
                      df: pd.DataFrame, 
                      file_path: str, 
                      format: str = 'csv',
                      **kwargs) -> str:
        """Export dataset to various formats"""
        
        file_path = Path(file_path)
        
        try:
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            elif format.lower() in ['xlsx', 'excel']:
                df.to_excel(file_path, index=False, **kwargs)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Dataset exported to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize ingestion service
    ingestion = DatasetIngestion()
    
    # Create sample data
    sample_df = ingestion.create_sample_data(n_rows=1000)
    print("Sample dataset created:")
    print(sample_df.head())
    print(f"Shape: {sample_df.shape}")
    
    # Export sample data
    sample_file = ingestion.export_dataset(sample_df, "sample_data.csv")
    print(f"Sample data exported to: {sample_file}")
    
    # Load and analyze the dataset
    df, schema = ingestion.load_dataset(sample_file)
    info = ingestion.get_dataset_info(df)
    
    print("\nDataset info:")
    print(f"Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print(f"Memory usage: {info['memory_usage']} bytes")
    print(f"Missing values: {info['missing_values']}")
