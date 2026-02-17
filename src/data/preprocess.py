import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    """Feature engineering for laptop data"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def clean_ram(self, ram_str: str) -> int:
        """Extract RAM: '8GB' -> 8"""
        return int(re.search(r'(\d+)', str(ram_str)).group(1))
    
    def clean_weight(self, weight_str: str) -> float:
        """Extract weight: '1.37kg' -> 1.37"""
        return float(re.search(r'([\d.]+)', str(weight_str)).group(1))
    
    def extract_cpu_brand(self, cpu_str: str) -> str:
        """Extract CPU brand"""
        cpu_str = str(cpu_str)
        if 'Intel Core i7' in cpu_str:
            return 'Intel Core i7'
        elif 'Intel Core i5' in cpu_str:
            return 'Intel Core i5'
        elif 'Intel Core i3' in cpu_str:
            return 'Intel Core i3'
        elif 'AMD' in cpu_str:
            return 'AMD'
        elif 'Intel' in cpu_str:
            return 'Intel Other'
        else:
            return 'Other'
    
    def extract_gpu_brand(self, gpu_str: str) -> str:
        """Extract GPU brand"""
        gpu_str = str(gpu_str)
        if 'Intel' in gpu_str:
            return 'Intel'
        elif 'AMD' in gpu_str:
            return 'AMD'
        elif 'Nvidia' in gpu_str or 'nvidia' in gpu_str:
            return 'Nvidia'
        else:
            return 'Other'
    
    def parse_memory(self, memory_str: str) -> tuple:
        """Parse memory: '128GB SSD' -> (0, 128)"""
        memory_str = str(memory_str)
        hdd = 0
        ssd = 0
        
        # SSD/Flash
        ssd_match = re.search(r'(\d+)GB\s*(SSD|Flash)', memory_str, re.IGNORECASE)
        if ssd_match:
            ssd = int(ssd_match.group(1))
        
        # HDD
        hdd_match = re.search(r'(\d+)GB\s*HDD', memory_str, re.IGNORECASE)
        if hdd_match:
            hdd = int(hdd_match.group(1))
        
        # TB to GB conversion
        tb_match = re.search(r'(\d+)TB', memory_str, re.IGNORECASE)
        if tb_match:
            if 'SSD' in memory_str or 'Flash' in memory_str:
                ssd = int(tb_match.group(1)) * 1024
            else:
                hdd = int(tb_match.group(1)) * 1024
        
        return hdd, ssd
    
    def extract_resolution(self, res_str: str) -> tuple:
        """Extract resolution: '2560x1600' -> (2560, 1600)"""
        res_str = str(res_str)
        match = re.search(r'(\d{3,4})x(\d{3,4})', res_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1920, 1080  # Default
    
    def calculate_ppi(self, x_res: int, y_res: int, inches: float) -> float:
        """Calculate pixels per inch"""
        if inches == 0:
            return 0
        return ((x_res**2 + y_res**2)**0.5) / inches
    
    def detect_touchscreen(self, res_str: str) -> int:
        """Detect touchscreen"""
        return 1 if 'Touchscreen' in str(res_str) else 0
    
    def detect_ips(self, res_str: str) -> int:
        """Detect IPS display"""
        return 1 if 'IPS' in str(res_str) else 0
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        df = df.copy()
        
        # Clean numeric columns
        df['Ram'] = df['Ram'].apply(self.clean_ram)
        df['Weight'] = df['Weight'].apply(self.clean_weight)
        
        # Extract brands
        df['Cpu_brand'] = df['Cpu'].apply(self.extract_cpu_brand)
        df['Gpu_brand'] = df['Gpu'].apply(self.extract_gpu_brand)
        
        # Parse Memory
        df[['HDD', 'SSD']] = df['Memory'].apply(
            lambda x: pd.Series(self.parse_memory(x))
        )
        
        # Extract resolution features
        df[['X_res', 'Y_res']] = df['ScreenResolution'].apply(
            lambda x: pd.Series(self.extract_resolution(x))
        )
        df['ppi'] = df.apply(
            lambda row: self.calculate_ppi(row['X_res'], row['Y_res'], row['Inches']), 
            axis=1
        )
        df['Touchscreen'] = df['ScreenResolution'].apply(self.detect_touchscreen)
        df['IPS'] = df['ScreenResolution'].apply(self.detect_ips)
        
        # Drop original columns
        df.drop(columns=['Cpu', 'Memory', 'ScreenResolution', 'Gpu', 'X_res', 'Y_res'], 
                inplace=True)
        
        # Encode categorical variables
        categorical_cols = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'OpSys']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders"""
        df = df.copy()
        
        # Apply same transformations
        df['Ram'] = df['Ram'].apply(self.clean_ram)
        df['Weight'] = df['Weight'].apply(self.clean_weight)
        df['Cpu_brand'] = df['Cpu'].apply(self.extract_cpu_brand)
        df['Gpu_brand'] = df['Gpu'].apply(self.extract_gpu_brand)
        
        df[['HDD', 'SSD']] = df['Memory'].apply(
            lambda x: pd.Series(self.parse_memory(x))
        )
        
        df[['X_res', 'Y_res']] = df['ScreenResolution'].apply(
            lambda x: pd.Series(self.extract_resolution(x))
        )
        df['ppi'] = df.apply(
            lambda row: self.calculate_ppi(row['X_res'], row['Y_res'], row['Inches']), 
            axis=1
        )
        df['Touchscreen'] = df['ScreenResolution'].apply(self.detect_touchscreen)
        df['IPS'] = df['ScreenResolution'].apply(self.detect_ips)
        
        df.drop(columns=['Cpu', 'Memory', 'ScreenResolution', 'Gpu', 'X_res', 'Y_res'], 
                inplace=True)
        
        # Use fitted encoders
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col].astype(str))
        
        return df