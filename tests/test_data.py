# Auto-generated file
"""
tests/test_data.py
Unit tests for data loading and preprocessing pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocess import FeatureEngineer


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal sample matching real laptop_data.xlsx schema"""
    return pd.DataFrame([
        {
            'Company': 'Apple',
            'TypeName': 'Ultrabook',
            'Inches': 13.3,
            'ScreenResolution': 'IPS Panel Retina Display 2560x1600',
            'Cpu': 'Intel Core i5 2.3GHz',
            'Ram': '8GB',
            'Memory': '128GB SSD',
            'Gpu': 'Intel Iris Plus Graphics 640',
            'OpSys': 'macOS',
            'Weight': '1.37kg',
            'Price': 71378.68
        },
        {
            'Company': 'Dell',
            'TypeName': 'Notebook',
            'Inches': 15.6,
            'ScreenResolution': '1920x1080',
            'Cpu': 'Intel Core i7 2.8GHz',
            'Ram': '16GB',
            'Memory': '512GB SSD',
            'Gpu': 'Nvidia GeForce GTX 1050',
            'OpSys': 'Windows 10',
            'Weight': '2.04kg',
            'Price': 85000.00
        },
        {
            'Company': 'HP',
            'TypeName': 'Gaming',
            'Inches': 15.6,
            'ScreenResolution': 'IPS Panel Touchscreen 1920x1080',
            'Cpu': 'AMD A10-Series 2.5GHz',
            'Ram': '8GB',
            'Memory': '1TB HDD',
            'Gpu': 'AMD Radeon RX 580',
            'OpSys': 'Windows 10',
            'Weight': '2.50kg',
            'Price': 55000.00
        },
    ])


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


# ─── Schema Tests ─────────────────────────────────────────────────────────────

class TestSchemaValidation:

    def test_required_columns_present(self, sample_df):
        required = ['Company', 'TypeName', 'Inches', 'ScreenResolution',
                    'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Price']
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_price_is_positive(self, sample_df):
        assert (sample_df['Price'] > 0).all(), "Price must be positive"

    def test_inches_in_valid_range(self, sample_df):
        assert sample_df['Inches'].between(10, 20).all(), "Screen size out of range"

    def test_no_null_prices(self, sample_df):
        assert sample_df['Price'].notna().all(), "Price has null values"


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    def test_ram_extraction(self, feature_engineer):
        assert feature_engineer.clean_ram('8GB') == 8
        assert feature_engineer.clean_ram('16GB') == 16
        assert feature_engineer.clean_ram('32GB') == 32

    def test_weight_extraction(self, feature_engineer):
        assert feature_engineer.clean_weight('1.37kg') == pytest.approx(1.37)
        assert feature_engineer.clean_weight('2.5kg') == pytest.approx(2.5)

    def test_cpu_brand_extraction(self, feature_engineer):
        assert feature_engineer.extract_cpu_brand('Intel Core i5 2.3GHz') == 'Intel Core i5'
        assert feature_engineer.extract_cpu_brand('Intel Core i7 3.0GHz') == 'Intel Core i7'
        assert feature_engineer.extract_cpu_brand('Intel Core i3 2.0GHz') == 'Intel Core i3'
        assert feature_engineer.extract_cpu_brand('AMD A10 2.5GHz') == 'AMD'
        assert feature_engineer.extract_cpu_brand('Unknown CPU') == 'Other'

    def test_gpu_brand_extraction(self, feature_engineer):
        assert feature_engineer.extract_gpu_brand('Nvidia GTX 1050') == 'Nvidia'
        assert feature_engineer.extract_gpu_brand('Intel Iris 640') == 'Intel'
        assert feature_engineer.extract_gpu_brand('AMD Radeon RX 580') == 'AMD'

    def test_ssd_parsing(self, feature_engineer):
        hdd, ssd = feature_engineer.parse_memory('128GB SSD')
        assert ssd == 128
        assert hdd == 0

    def test_hdd_parsing(self, feature_engineer):
        hdd, ssd = feature_engineer.parse_memory('1TB HDD')
        assert hdd == 1024
        assert ssd == 0

    def test_flash_storage_parsing(self, feature_engineer):
        hdd, ssd = feature_engineer.parse_memory('128GB Flash Storage')
        assert ssd == 128

    def test_ppi_calculation(self, feature_engineer):
        ppi = feature_engineer.calculate_ppi(1920, 1080, 15.6)
        assert ppi == pytest.approx(141.2, rel=0.01)

    def test_ppi_zero_screen_size(self, feature_engineer):
        ppi = feature_engineer.calculate_ppi(1920, 1080, 0)
        assert ppi == 0

    def test_touchscreen_detection(self, feature_engineer):
        assert feature_engineer.detect_touchscreen('IPS Touchscreen 1920x1080') == 1
        assert feature_engineer.detect_touchscreen('1920x1080') == 0

    def test_ips_detection(self, feature_engineer):
        assert feature_engineer.detect_ips('IPS Panel 2560x1600') == 1
        assert feature_engineer.detect_ips('1366x768') == 0

    def test_resolution_extraction(self, feature_engineer):
        x, y = feature_engineer.extract_resolution('IPS Panel Retina Display 2560x1600')
        assert x == 2560
        assert y == 1600

    def test_resolution_default_fallback(self, feature_engineer):
        x, y = feature_engineer.extract_resolution('No resolution string here')
        assert x == 1920 and y == 1080


# ─── Pipeline Tests ───────────────────────────────────────────────────────────

class TestPipeline:

    def test_fit_transform_output_shape(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        assert len(result) == len(sample_df)

    def test_original_columns_dropped(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        dropped = ['Cpu', 'Memory', 'ScreenResolution', 'Gpu']
        for col in dropped:
            assert col not in result.columns, f"Column {col} should be dropped"

    def test_new_columns_added(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        for col in ['ppi', 'Touchscreen', 'IPS', 'HDD', 'SSD', 'Cpu_brand', 'Gpu_brand']:
            assert col in result.columns, f"Expected column {col} not found"

    def test_no_nulls_after_transform(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        assert result.isnull().sum().sum() == 0, "Null values found after preprocessing"

    def test_ram_is_numeric(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        assert pd.api.types.is_numeric_dtype(result['Ram'])

    def test_weight_is_numeric(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        assert pd.api.types.is_numeric_dtype(result['Weight'])

    def test_categorical_columns_encoded(self, feature_engineer, sample_df):
        result = feature_engineer.fit_transform(sample_df)
        for col in ['Company', 'TypeName', 'OpSys']:
            assert pd.api.types.is_numeric_dtype(result[col]), \
                f"{col} should be encoded as numeric"