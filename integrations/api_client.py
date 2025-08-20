"""
API integrations for external data sources (Amazon, Walmart, etc.).
File: integrations/api_client.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
import time
import json
import logging
from abc import ABC, abstractmethod
import base64
import hashlib
import hmac
from urllib.parse import urlencode
import warnings
warnings.filterwarnings('ignore')

class BaseAPIClient(ABC):
    """Base class for API clients"""
    
    def __init__(self, client_id: str, client_secret: str, rate_limit: int = 60):
        self.client_id = client_id
        self.client_secret = client_secret
        self.rate_limit = rate_limit  # requests per minute
        self.last_request_time = 0
        self.request_count = 0
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _rate_limit_check(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Wait if we've hit the rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the API"""
        pass
    
    @abstractmethod
    def get_campaign_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get campaign performance data"""
        pass
    
    @abstractmethod
    def get_keyword_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get keyword performance data"""
        pass

class AmazonAdvertisingClient(BaseAPIClient):
    """Amazon Advertising API client"""
    
    def __init__(self, client_id: str, client_secret: str, refresh_token: str, 
                 marketplace_id: str = "ATVPDKIKX0DER"):  # US marketplace
        super().__init__(client_id, client_secret)
        self.refresh_token = refresh_token
        self.marketplace_id = marketplace_id
        self.access_token = None
        self.profile_id = None
        self.base_url = "https://advertising-api.amazon.com"
        
    def authenticate(self) -> bool:
        """Authenticate with Amazon Advertising API"""
        try:
            # Get access token
            token_url = "https://api.amazon.com/auth/o2/token"
            token_data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            response = requests.post(token_url, data=token_data)
            response.raise_for_status()
            
            token_info = response.json()
            self.access_token = token_info['access_token']
            
            # Get profile information
            self._get_profiles()
            
            self.logger.info("Amazon Advertising API authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Amazon authentication failed: {str(e)}")
            return False
    
    def _get_profiles(self):
        """Get advertising profiles"""
        if not self.access_token:
            raise ValueError("Not authenticated")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Amazon-Advertising-API-ClientId': self.client_id
        }
        
        response = self.session.get(f"{self.base_url}/v2/profiles", headers=headers)
        response.raise_for_status()
        
        profiles = response.json()
        # Use first available profile
        if profiles:
            self.profile_id = profiles[0]['profileId']
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Dict:
        """Make authenticated request to Amazon API"""
        self._rate_limit_check()
        
        if not self.access_token or not self.profile_id:
            raise ValueError("Not authenticated")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Amazon-Advertising-API-ClientId': self.client_id,
            'Amazon-Advertising-API-Scope': str(self.profile_id)
        }
        
        url = f"{self.base_url}{endpoint}"
        
        if method == 'GET':
            response = self.session.get(url, headers=headers, params=data)
        elif method == 'POST':
            response = self.session.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    def get_campaign_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get Amazon campaign performance data"""
        
        # First get list of campaigns
        campaigns_response = self._make_request('/v2/sp/campaigns')
        campaign_ids = [c['campaignId'] for c in campaigns_response if c['state'] == 'enabled']
        
        # Get performance reports
        report_data = []
        
        for campaign_id in campaign_ids:
            try:
                # Request campaign report
                report_request = {
                    'reportDate': end_date.strftime('%Y%m%d'),
                    'metrics': [
                        'impressions', 'clicks', 'cost', 'sales', 'orders',
                        'ctr', 'cpc', 'acos', 'roas'
                    ],
                    'campaignType': 'sponsoredProducts'
                }
                
                # Note: In real implementation, you'd use the actual reporting API
                # This is a simplified version for demonstration
                campaign_performance = {
                    'campaign_id': campaign_id,
                    'platform': 'amazon',
                    'campaign_type': 'sponsored_products',
                    'date': end_date,
                    'impressions': np.random.randint(1000, 50000),
                    'clicks': np.random.randint(50, 2000),
                    'spend': np.random.uniform(100, 5000),
                    'sales': np.random.uniform(500, 15000),
                    'orders': np.random.randint(5, 200)
                }
                
                # Calculate derived metrics
                campaign_performance['cpc'] = campaign_performance['spend'] / campaign_performance['clicks']
                campaign_performance['ctr'] = campaign_performance['clicks'] / campaign_performance['impressions'] * 100
                campaign_performance['acos'] = campaign_performance['spend'] / campaign_performance['sales']
                campaign_performance['roas'] = campaign_performance['sales'] / campaign_performance['spend']
                campaign_performance['conversion_rate'] = campaign_performance['orders'] / campaign_performance['clicks'] * 100
                
                report_data.append(campaign_performance)
                
            except Exception as e:
                self.logger.warning(f"Failed to get data for campaign {campaign_id}: {str(e)}")
        
        return pd.DataFrame(report_data)
    
    def get_keyword_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get Amazon keyword performance data"""
        
        # Get keywords for all campaigns
        keywords_response = self._make_request('/v2/sp/keywords')
        
        keyword_data = []
        for keyword in keywords_response:
            if keyword['state'] == 'enabled':
                keyword_performance = {
                    'keyword_id': keyword['keywordId'],
                    'campaign_id': keyword['campaignId'],
                    'keyword': keyword['keywordText'],
                    'match_type': keyword['matchType'],
                    'date': end_date,
                    'impressions': np.random.randint(100, 10000),
                    'clicks': np.random.randint(5, 500),
                    'spend': np.random.uniform(10, 1000),
                    'sales': np.random.uniform(50, 3000),
                    'orders': np.random.randint(1, 50),
                    'bid': keyword.get('bid', np.random.uniform(0.5, 5.0)),
                    'avg_position': np.random.uniform(1, 20)
                }
                keyword_data.append(keyword_performance)
        
        return pd.DataFrame(keyword_data)
    
    def get_product_sales_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get product sales attribution data"""
        
        # In real implementation, this would integrate with Vendor Central or Seller Central
        # For demo purposes, we'll simulate the data structure
        
        products_data = []
        asins = [f"B{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), 8))}" for _ in range(20)]
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for date in date_range:
            for asin in asins:
                product_sales = {
                    'asin': asin,
                    'date': date,
                    'platform': 'amazon',
                    'organic_sales': np.random.uniform(100, 5000),
                    'paid_sales': np.random.uniform(50, 2000),
                    'total_sales': 0,  # Will be calculated
                    'units_sold': np.random.randint(5, 200),
                    'cost_per_unit': np.random.uniform(10, 50),
                    'selling_price': np.random.uniform(20, 100)
                }
                product_sales['total_sales'] = product_sales['organic_sales'] + product_sales['paid_sales']
                product_sales['margin_per_unit'] = product_sales['selling_price'] - product_sales['cost_per_unit']
                
                products_data.append(product_sales)
        
        return pd.DataFrame(products_data)

class WalmartDSPClient(BaseAPIClient):
    """Walmart DSP API client"""
    
    def __init__(self, client_id: str, client_secret: str, access_token: str = None):
        super().__init__(client_id, client_secret)
        self.access_token = access_token
        self.base_url = "https://api.walmart.com/v3/dsp"
        
    def authenticate(self) -> bool:
        """Authenticate with Walmart DSP API"""
        try:
            # Walmart uses OAuth 2.0 client credentials flow
            auth_url = "https://marketplace.walmartapis.com/v3/token"
            
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = auth_string.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                'Authorization': f'Basic {auth_b64}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {'grant_type': 'client_credentials'}
            
            response = requests.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_info = response.json()
            self.access_token = token_info['access_token']
            
            self.logger.info("Walmart DSP API authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Walmart authentication failed: {str(e)}")
            return False
    
    def _make_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Dict:
        """Make authenticated request to Walmart API"""
        self._rate_limit_check()
        
        if not self.access_token:
            raise ValueError("Not authenticated")
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'WM_SVC.NAME': 'Walmart-DSP-API'
        }
        
        url = f"{self.base_url}{endpoint}"
        
        if method == 'GET':
            response = self.session.get(url, headers=headers, params=data)
        elif method == 'POST':
            response = self.session.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    def get_campaign_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get Walmart campaign performance data"""
        
        # Simulate Walmart campaign data structure
        campaign_data = []
        
        # In real implementation, would call Walmart's reporting API
        num_campaigns = np.random.randint(10, 20)
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for campaign_idx in range(num_campaigns):
            campaign_id = f"WM-CAMP-{campaign_idx + 1:06d}"
            
            for date in date_range:
                campaign_performance = {
                    'campaign_id': campaign_id,
                    'platform': 'walmart',
                    'campaign_type': 'sponsored_products',
                    'date': date,
                    'impressions': np.random.randint(500, 30000),
                    'clicks': np.random.randint(25, 1500),
                    'spend': np.random.uniform(50, 3000),
                    'sales': np.random.uniform(200, 9000),
                    'orders': np.random.randint(3, 150)
                }
                
                # Calculate derived metrics
                campaign_performance['cpc'] = campaign_performance['spend'] / campaign_performance['clicks']
                campaign_performance['ctr'] = campaign_performance['clicks'] / campaign_performance['impressions'] * 100
                campaign_performance['acos'] = campaign_performance['spend'] / campaign_performance['sales']
                campaign_performance['roas'] = campaign_performance['sales'] / campaign_performance['spend']
                campaign_performance['conversion_rate'] = campaign_performance['orders'] / campaign_performance['clicks'] * 100
                
                campaign_data.append(campaign_performance)
        
        return pd.DataFrame(campaign_data)
    
    def get_keyword_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get Walmart keyword performance data"""
        
        # Simulate keyword data for Walmart
        keyword_data = []
        keywords = [
            "wireless headphones walmart", "bluetooth speaker", "phone case",
            "charging cable", "tablet stand", "screen protector"
        ]
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for keyword_idx, keyword in enumerate(keywords):
            for date in date_range:
                keyword_performance = {
                    'keyword_id': f"WM-KW-{keyword_idx + 1:06d}",
                    'campaign_id': f"WM-CAMP-{np.random.randint(1, 11):06d}",
                    'keyword': keyword,
                    'match_type': np.random.choice(['exact', 'phrase', 'broad']),
                    'date': date,
                    'impressions': np.random.randint(50, 5000),
                    'clicks': np.random.randint(2, 250),
                    'spend': np.random.uniform(5, 500),
                    'sales': np.random.uniform(25, 1500),
                    'orders': np.random.randint(1, 25),
                    'bid': np.random.uniform(0.3, 3.0),
                    'avg_position': np.random.uniform(1, 15)
                }
                keyword_data.append(keyword_performance)
        
        return pd.DataFrame(keyword_data)

class DataIntegrationManager:
    """Manages data integration from multiple API sources"""
    
    def __init__(self, api_configs: Dict[str, Dict[str, str]]):
        self.api_clients = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize API clients
        if 'amazon' in api_configs:
            config = api_configs['amazon']
            self.api_clients['amazon'] = AmazonAdvertisingClient(
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                refresh_token=config['refresh_token']
            )
        
        if 'walmart' in api_configs:
            config = api_configs['walmart']
            self.api_clients['walmart'] = WalmartDSPClient(
                client_id=config['client_id'],
                client_secret=config['client_secret']
            )
    
    def authenticate_all(self) -> Dict[str, bool]:
        """Authenticate with all configured APIs"""
        auth_results = {}
        
        for platform, client in self.api_clients.items():
            try:
                auth_results[platform] = client.authenticate()
                if auth_results[platform]:
                    self.logger.info(f"Successfully authenticated with {platform}")
                else:
                    self.logger.error(f"Failed to authenticate with {platform}")
            except Exception as e:
                self.logger.error(f"Authentication error for {platform}: {str(e)}")
                auth_results[platform] = False
        
        return auth_results
    
    def fetch_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch data from all authenticated sources"""
        
        all_data = {
            'campaigns': pd.DataFrame(),
            'keywords': pd.DataFrame(),
            'products': pd.DataFrame()
        }
        
        for platform, client in self.api_clients.items():
            try:
                self.logger.info(f"Fetching data from {platform}...")
                
                # Get campaign data
                campaign_data = client.get_campaign_data(start_date, end_date)
                all_data['campaigns'] = pd.concat([all_data['campaigns'], campaign_data], ignore_index=True)
                
                # Get keyword data
                keyword_data = client.get_keyword_data(start_date, end_date)
                all_data['keywords'] = pd.concat([all_data['keywords'], keyword_data], ignore_index=True)
                
                # Get product data (if available)
                if hasattr(client, 'get_product_sales_data'):
                    product_data = client.get_product_sales_data(start_date, end_date)
                    all_data['products'] = pd.concat([all_data['products'], product_data], ignore_index=True)
                
                self.logger.info(f"Successfully fetched data from {platform}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data from {platform}: {str(e)}")
        
        return all_data
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Validate quality of fetched data"""
        
        quality_report = {}
        
        for data_type, df in data.items():
            if df.empty:
                quality_report[data_type] = {
                    'status': 'empty',
                    'record_count': 0,
                    'issues': ['No data available']
                }
                continue
            
            issues = []
            
            # Check for missing values
            missing_rates = df.isnull().sum() / len(df)
            high_missing = missing_rates[missing_rates > 0.3].index.tolist()
            if high_missing:
                issues.append(f"High missing values in: {', '.join(high_missing)}")
            
            # Check for duplicate records
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                issues.append(f"{duplicates} duplicate records found")
            
            # Check for negative values in metrics that shouldn't be negative
            if data_type in ['campaigns', 'keywords']:
                metric_cols = ['impressions', 'clicks', 'spend', 'sales', 'orders']
                for col in metric_cols:
                    if col in df.columns and (df[col] < 0).any():
                        issues.append(f"Negative values found in {col}")
            
            # Check date ranges
            if 'date' in df.columns:
                date_range = df['date'].max() - df['date'].min()
                if date_range.days == 0:
                    issues.append("Single date data only")
            
            quality_report[data_type] = {
                'status': 'good' if not issues else 'issues_found',
                'record_count': len(df),
                'missing_data_rate': missing_rates.mean(),
                'duplicate_count': duplicates,
                'issues': issues
            }
        
        return quality_report
    
    def sync_data_with_warehouse(self, data: Dict[str, pd.DataFrame], 
                                warehouse: 'DataWarehouse') -> bool:
        """Sync fetched data with data warehouse"""
        
        try:
            if not data['campaigns'].empty:
                warehouse.load_campaign_data(data['campaigns'])
                self.logger.info(f"Loaded {len(data['campaigns'])} campaign records")
            
            if not data['keywords'].empty:
                warehouse.load_keyword_data(data['keywords'])
                self.logger.info(f"Loaded {len(data['keywords'])} keyword records")
            
            if not data['products'].empty:
                warehouse.load_product_data(data['products'])
                self.logger.info(f"Loaded {len(data['products'])} product records")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to sync data with warehouse: {str(e)}")
            return False

# Example usage and testing functions
def test_api_clients():
    """Test API client functionality with mock data"""
    
    print("Testing API clients...")
    
    # Test Amazon client
    amazon_client = AmazonAdvertisingClient(
        client_id="test_client_id",
        client_secret="test_client_secret", 
        refresh_token="test_refresh_token"
    )
    
    # Test Walmart client
    walmart_client = WalmartDSPClient(
        client_id="test_client_id",
        client_secret="test_client_secret"
    )
    
    # Test date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print("✅ API clients initialized successfully")
    
    # Note: In real usage, you would authenticate first
    # For testing, we'll use the mock data generation methods
    
    return True

def create_sample_api_config():
    """Create sample API configuration"""
    
    config = {
        'amazon': {
            'client_id': 'your_amazon_client_id',
            'client_secret': 'your_amazon_client_secret',
            'refresh_token': 'your_amazon_refresh_token'
        },
        'walmart': {
            'client_id': 'your_walmart_client_id',
            'client_secret': 'your_walmart_client_secret'
        }
    }
    
    return config

if __name__ == "__main__":
    # Run tests
    test_api_clients()
    
    # Create sample config
    sample_config = create_sample_api_config()
    print("Sample API configuration created")
    print(json.dumps(sample_config, indent=2))
