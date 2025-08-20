"""
Data models and schemas for advertising ROI optimization engine.
File: data/models.py
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum

class Platform(Enum):
    AMAZON = "amazon"
    WALMART = "walmart"

class CampaignType(Enum):
    SPONSORED_PRODUCTS = "sponsored_products"
    SPONSORED_BRANDS = "sponsored_brands"
    SPONSORED_DISPLAY = "sponsored_display"
    VIDEO = "video"

@dataclass
class CampaignData:
    """Core campaign performance data structure"""
    campaign_id: str
    platform: Platform
    campaign_type: CampaignType
    date: datetime
    impressions: int
    clicks: int
    spend: float
    sales: float
    orders: int
    acos: float
    roas: float
    cpc: float
    ctr: float
    conversion_rate: float
    
@dataclass
class KeywordData:
    """Keyword-level performance data"""
    keyword_id: str
    campaign_id: str
    keyword: str
    match_type: str
    impressions: int
    clicks: int
    spend: float
    sales: float
    orders: int
    date: datetime
    bid: float
    avg_position: float

@dataclass
class ProductData:
    """Product-level sales and financial data"""
    asin: str
    sku: str
    platform: Platform
    date: datetime
    organic_sales: float
    paid_sales: float
    total_sales: float
    units_sold: int
    cost_per_unit: float
    selling_price: float
    margin_per_unit: float
    inventory_value: float
    days_of_supply: int

@dataclass
class FinancialData:
    """Financial integration data from ERP systems"""
    asin: str
    date: datetime
    cost_of_goods_sold: float
    platform_fees: float
    fulfillment_fees: float
    storage_fees: float
    payment_terms_days: int
    currency_rate: float
    carrying_cost_rate: float

@dataclass
class AttributionData:
    """Multi-touch attribution data"""
    customer_id: str
    touchpoint_id: str
    platform: Platform
    touchpoint_type: str  # click, impression, view
    timestamp: datetime
    campaign_id: str
    keyword_id: Optional[str]
    conversion_timestamp: Optional[datetime]
    conversion_value: float
    attribution_weight: float

class DataProcessor:
    """Main data processing class for cleaning and validation"""
    
    def __init__(self):
        self.required_columns = {
            'campaign': ['campaign_id', 'platform', 'date', 'spend', 'sales'],
            'keyword': ['keyword_id', 'campaign_id', 'keyword', 'spend', 'sales'],
            'product': ['asin', 'platform', 'date', 'total_sales'],
            'financial': ['asin', 'date', 'cost_of_goods_sold'],
            'attribution': ['customer_id', 'touchpoint_id', 'platform', 'timestamp']
        }
    
    def validate_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate and clean input data"""
        if data_type not in self.required_columns:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Check required columns
        missing_cols = set(self.required_columns[data_type]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Data type conversions
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def aggregate_campaign_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate campaign data at daily level"""
        agg_dict = {
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'sales': 'sum',
            'orders': 'sum'
        }
        
        result = df.groupby(['campaign_id', 'platform', 'date']).agg(agg_dict).reset_index()
        
        # Calculate derived metrics
        result['cpc'] = result['spend'] / result['clicks'].replace(0, 1)
        result['ctr'] = result['clicks'] / result['impressions'].replace(0, 1)
        result['conversion_rate'] = result['orders'] / result['clicks'].replace(0, 1)
        result['acos'] = result['spend'] / result['sales'].replace(0, 1)
        result['roas'] = result['sales'] / result['spend'].replace(0, 1)
        
        return result
    
    def calculate_rolling_metrics(self, df: pd.DataFrame, window_days: List[int] = [7, 14, 30, 90]) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        df = df.sort_values(['campaign_id', 'date'])
        
        for window in window_days:
            for metric in ['spend', 'sales', 'roas', 'acos']:
                col_name = f'{metric}_rolling_{window}d'
                if metric in ['roas', 'acos']:
                    # For ratios, calculate from rolling spend and sales
                    rolling_spend = df.groupby('campaign_id')['spend'].rolling(window=window, min_periods=1).sum().values
                    rolling_sales = df.groupby('campaign_id')['sales'].rolling(window=window, min_periods=1).sum().values
                    
                    if metric == 'roas':
                        df[col_name] = rolling_sales / np.where(rolling_spend == 0, 1, rolling_spend)
                    else:  # acos
                        df[col_name] = rolling_spend / np.where(rolling_sales == 0, 1, rolling_sales)
                else:
                    df[col_name] = df.groupby('campaign_id')[metric].rolling(window=window, min_periods=1).sum().values
        
        return df

class DataWarehouse:
    """Data warehouse simulation for storing and retrieving processed data"""
    
    def __init__(self):
        self.campaigns = pd.DataFrame()
        self.keywords = pd.DataFrame()
        self.products = pd.DataFrame()
        self.financial = pd.DataFrame()
        self.attribution = pd.DataFrame()
        self.processor = DataProcessor()
    
    def load_campaign_data(self, df: pd.DataFrame) -> None:
        """Load and process campaign data"""
        df = self.processor.validate_data(df, 'campaign')
        df = self.processor.aggregate_campaign_data(df)
        df = self.processor.calculate_rolling_metrics(df)
        self.campaigns = pd.concat([self.campaigns, df], ignore_index=True).drop_duplicates()
    
    def load_keyword_data(self, df: pd.DataFrame) -> None:
        """Load keyword performance data"""
        df = self.processor.validate_data(df, 'keyword')
        self.keywords = pd.concat([self.keywords, df], ignore_index=True).drop_duplicates()
    
    def load_product_data(self, df: pd.DataFrame) -> None:
        """Load product sales data"""
        df = self.processor.validate_data(df, 'product')
        self.products = pd.concat([self.products, df], ignore_index=True).drop_duplicates()
    
    def load_financial_data(self, df: pd.DataFrame) -> None:
        """Load financial data from ERP"""
        df = self.processor.validate_data(df, 'financial')
        self.financial = pd.concat([self.financial, df], ignore_index=True).drop_duplicates()
    
    def load_attribution_data(self, df: pd.DataFrame) -> None:
        """Load attribution tracking data"""
        df = self.processor.validate_data(df, 'attribution')
        self.attribution = pd.concat([self.attribution, df], ignore_index=True).drop_duplicates()
    
    def get_campaign_performance(self, start_date: datetime, end_date: datetime, 
                               platform: Optional[Platform] = None) -> pd.DataFrame:
        """Retrieve campaign performance data for date range"""
        mask = (self.campaigns['date'] >= start_date) & (self.campaigns['date'] <= end_date)
        if platform:
            mask &= (self.campaigns['platform'] == platform.value)
        return self.campaigns[mask].copy()
    
    def get_unified_dataset(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create unified dataset joining all data sources"""
        # Get base campaign data
        campaigns = self.get_campaign_performance(start_date, end_date)
        
        # Join with financial data
        if not self.financial.empty:
            financial_agg = self.financial.groupby(['date']).agg({
                'cost_of_goods_sold': 'mean',
                'platform_fees': 'mean',
                'fulfillment_fees': 'mean',
                'carrying_cost_rate': 'mean'
            }).reset_index()
            campaigns = campaigns.merge(financial_agg, on='date', how='left')
        
        return campaigns