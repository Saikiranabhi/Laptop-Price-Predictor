import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate laptop dataset"""
    
    def __init__(self, filepath: str = "data/raw/laptop_data.xlsx"):
        self.filepath = filepath
    
    def load(self) -> pd.DataFrame:
        """Load Excel file"""
        try:
            df = pd.read_excel(self.filepath)
            logger.info(f"✅ Loaded {len(df)} records from {self.filepath}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate required columns exist"""
        required_cols = ['Company', 'TypeName', 'Inches', 'ScreenResolution', 
                        'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.error(f"❌ Missing columns: {missing}")
            return False
        
        logger.info("✅ Schema validation passed")
        return True