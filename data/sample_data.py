"""
Sample data generation for testing the advertising ROI optimization engine.
File: data/sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import random
import string

class SampleDataGenerator:
    """Generate realistic sample data for testing the advertising engine"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Sample product catalog
        self.sample_asins = [f"B{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}" for _ in range(50)]
        self.sample_skus = [f"SKU-{str(i).zfill(5)}" for i in range(1, 51)]
        
        # Sample keywords for different categories
        self.keywords = [
            "wireless headphones", "bluetooth speaker", "phone case", "charging cable",
            "tablet stand", "screen protector", "car charger", "power bank",
            "gaming mouse", "keyboard", "laptop bag", "external drive",
            "smart watch", "fitness tracker", "phone mount", "usb cable"
        ]
        
        # Campaign templates
        self.campaign_templates = [
            "Brand-{product}-Exact", "Brand-{product}-Broad", "Generic-{category}-Auto",
            "Competitor-{brand}", "Video-{product}", "Display-{category}"
        ]
    
    def generate_campaign_data(self, days: int = 90, campaigns_count: int = 25) -> pd.DataFrame:
        """Generate sample campaign performance data"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        # Generate campaign IDs
        campaign_ids = [f"CAMP-{str(i).zfill(6)}" for i in range(1, campaigns_count + 1)]
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for campaign_id in campaign_ids:
                platform = random.choice(['amazon', 'walmart'])
                campaign_type = random.choice(['sponsored_products', 'sponsored_brands', 'sponsored_display'])
                
                # Base performance with seasonality and trends
                seasonality_factor = 1 + 0.3 * np.sin(2 * np.pi * day / 365)  # Annual seasonality
                weekend_factor = 0.8 if current_date.weekday() >= 5 else 1.0  # Weekend effect
                
                # Generate correlated metrics
                base_impressions = np.random.lognormal(8, 1) * seasonality_factor * weekend_factor
                ctr = np.random.beta(2, 50) * 100  # 2-8% CTR typically
                clicks = int(base_impressions * ctr / 100)
                
                cpc = np.random.gamma(2, 0.5) + 0.2  # $0.2-$3 CPC range
                spend = clicks * cpc
                
                conversion_rate = np.random.beta(2, 20) * 100  # 2-15% conversion rate
                orders = int(clicks * conversion_rate / 100)
                
                avg_order_value = np.random.gamma(3, 15) + 10  # $10-$100 AOV
                sales = orders * avg_order_value
                
                acos = spend / sales if sales > 0 else 1.0
                roas = sales / spend if spend > 0 else 0.0
                
                data.append({
                    'campaign_id': campaign_id,
                    'platform': platform,
                    'campaign_type': campaign_type,
                    'date': current_date,
                    'impressions': int(base_impressions),
                    'clicks': clicks,
                    'spend': round(spend, 2),
                    'sales': round(sales, 2),
                    'orders': orders,
                    'acos': round(acos, 3),
                    'roas': round(roas, 2),
                    'cpc': round(cpc, 2),
                    'ctr': round(ctr, 3),
                    'conversion_rate': round(conversion_rate, 3)
                })
        
        return pd.DataFrame(data)
    
    def generate_keyword_data(self, campaigns_df: pd.DataFrame) -> pd.DataFrame:
        """Generate keyword-level performance data"""
        data = []
        
        for _, campaign in campaigns_df.iterrows():
            # Each campaign has 3-8 keywords
            num_keywords = random.randint(3, 8)
            
            for i in range(num_keywords):
                keyword = random.choice(self.keywords)
                match_type = random.choice(['exact', 'phrase', 'broad'])
                
                # Keyword performance is fraction of campaign performance
                fraction = np.random.dirichlet([1] * num_keywords)[i]
                
                data.append({
                    'keyword_id': f"KW-{campaign['campaign_id']}-{i+1}",
                    'campaign_id': campaign['campaign_id'],
                    'keyword': keyword,
                    'match_type': match_type,
                    'impressions': int(campaign['impressions'] * fraction),
                    'clicks': int(campaign['clicks'] * fraction),
                    'spend': round(campaign['spend'] * fraction, 2),
                    'sales': round(campaign['sales'] * fraction, 2),
                    'orders': int(campaign['orders'] * fraction),
                    'date': campaign['date'],
                    'bid': round(np.random.gamma(2, 0.5) + 0.1, 2),
                    'avg_position': round(np.random.uniform(1, 20), 1)
                })
        
        return pd.DataFrame(data)
    
    def generate_product_data(self, days: int = 90) -> pd.DataFrame:
        """Generate product-level sales data"""
        data = []
        start_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for i, asin in enumerate(self.sample_asins):
                platform = random.choice(['amazon', 'walmart'])
                
                # Product performance varies by product
                base_sales = np.random.gamma(2, 50) * (1 + 0.2 * np.sin(2 * np.pi * day / 365))
                
                # Split between organic and paid
                paid_ratio = np.random.beta(3, 7)  # 10-50% typically paid
                paid_sales = base_sales * paid_ratio
                organic_sales = base_sales * (1 - paid_ratio)
                
                units_sold = int(base_sales / np.random.uniform(15, 80))  # Variable price point
                selling_price = base_sales / max(units_sold, 1)
                cost_per_unit = selling_price * np.random.uniform(0.3, 0.7)  # 30-70% margin
                
                data.append({
                    'asin': asin,
                    'sku': self.sample_skus[i],
                    'platform': platform,
                    'date': current_date,
                    'organic_sales': round(organic_sales, 2),
                    'paid_sales': round(paid_sales, 2),
                    'total_sales': round(base_sales, 2),
                    'units_sold': units_sold,
                    'cost_per_unit': round(cost_per_unit, 2),
                    'selling_price': round(selling_price, 2),
                    'margin_per_unit': round(selling_price - cost_per_unit, 2),
                    'inventory_value': round(cost_per_unit * np.random.uniform(50, 200), 2),
                    'days_of_supply': random.randint(15, 90)
                })
        
        return pd.DataFrame(data)
    
    def generate_financial_data(self, product_df: pd.DataFrame) -> pd.DataFrame:
        """Generate financial data aligned with products"""
        data = []
        
        unique_dates = product_df['date'].unique()
        unique_asins = product_df['asin'].unique()
        
        for date in unique_dates:
            for asin in unique_asins:
                product_row = product_df[
                    (product_df['date'] == date) & (product_df['asin'] == asin)
                ].iloc[0] if len(product_df[
                    (product_df['date'] == date) & (product_df['asin'] == asin)
                ]) > 0 else None
                
                if product_row is not None:
                    total_sales = product_row['total_sales']
                    cost_per_unit = product_row['cost_per_unit']
                    units_sold = product_row['units_sold']
                    
                    data.append({
                        'asin': asin,
                        'date': date,
                        'cost_of_goods_sold': round(cost_per_unit * units_sold, 2),
                        'platform_fees': round(total_sales * np.random.uniform(0.08, 0.15), 2),  # 8-15% fees
                        'fulfillment_fees': round(units_sold * np.random.uniform(2, 8), 2),  # $2-8 per unit
                        'storage_fees': round(np.random.uniform(0.5, 3), 2),  # Daily storage
                        'payment_terms_days': random.choice([14, 30, 45]),  # Payment terms
                        'currency_rate': round(np.random.normal(1.0, 0.02), 4),  # FX rate
                        'carrying_cost_rate': round(np.random.uniform(0.12, 0.18), 4)  # 12-18% annual
                    })
        
        return pd.DataFrame(data)
    
    def generate_attribution_data(self, campaigns_df: pd.DataFrame, 
                                keywords_df: pd.DataFrame) -> pd.DataFrame:
        """Generate multi-touch attribution data"""
        data = []
        
        # Generate customer journeys
        num_customers = 1000
        
        for customer_id in range(1, num_customers + 1):
            # Each customer has 1-5 touchpoints before conversion
            num_touchpoints = random.randint(1, 5)
            conversion_value = np.random.gamma(3, 20) + 10  # $10-100 order value
            
            # Random conversion date from campaign data
            conversion_date = random.choice(campaigns_df['date'].tolist())
            
            for touchpoint_idx in range(num_touchpoints):
                # Touchpoint occurs 0-14 days before conversion
                touchpoint_date = conversion_date - timedelta(days=random.randint(0, 14))
                
                # Select random campaign and keyword
                campaign_row = campaigns_df[campaigns_df['date'] >= touchpoint_date].sample(1).iloc[0]
                keyword_row = keywords_df[keywords_df['campaign_id'] == campaign_row['campaign_id']].sample(1)
                
                if len(keyword_row) > 0:
                    keyword_row = keyword_row.iloc[0]
                    
                    # Attribution weight decreases with time
                    days_before = (conversion_date - touchpoint_date).days
                    attribution_weight = 1.0 / (1 + 0.1 * days_before)
                    
                    data.append({
                        'customer_id': f"CUST-{str(customer_id).zfill(6)}",
                        'touchpoint_id': f"TP-{customer_id}-{touchpoint_idx+1}",
                        'platform': campaign_row['platform'],
                        'touchpoint_type': random.choice(['click', 'impression', 'view']),
                        'timestamp': touchpoint_date,
                        'campaign_id': campaign_row['campaign_id'],
                        'keyword_id': keyword_row['keyword_id'],
                        'conversion_timestamp': conversion_date if touchpoint_idx == num_touchpoints - 1 else None,
                        'conversion_value': conversion_value if touchpoint_idx == num_touchpoints - 1 else 0.0,
                        'attribution_weight': round(attribution_weight, 3)
                    })
        
        return pd.DataFrame(data)
    
    def generate_competitive_data(self, keywords_df: pd.DataFrame) -> pd.DataFrame:
        """Generate competitive intelligence data"""
        data = []
        
        unique_keywords = keywords_df['keyword'].unique()
        unique_dates = keywords_df['date'].unique()
        
        for date in unique_dates:
            for keyword in unique_keywords:
                data.append({
                    'keyword': keyword,
                    'date': date,
                    'competitor_count': random.randint(5, 25),
                    'avg_competitor_bid': round(np.random.gamma(2, 0.8), 2),
                    'our_rank': random.randint(1, 10),
                    'market_share': round(np.random.beta(2, 8), 3),  # 5-30% market share
                    'competitive_intensity': round(np.random.uniform(0.3, 1.0), 2),
                    'seasonal_demand_index': round(1 + 0.4 * np.sin(2 * np.pi * 
                        (date - unique_dates[0]).days / 365), 2)
                })
        
        return pd.DataFrame(data)
    
    def generate_all_sample_data(self, days: int = 90) -> dict:
        """Generate complete sample dataset"""
        print(f"Generating {days} days of sample data...")
        
        # Generate base campaign data
        campaigns = self.generate_campaign_data(days)
        print(f"Generated {len(campaigns)} campaign records")
        
        # Generate related data
        keywords = self.generate_keyword_data(campaigns)
        print(f"Generated {len(keywords)} keyword records")
        
        products = self.generate_product_data(days)
        print(f"Generated {len(products)} product records")
        
        financial = self.generate_financial_data(products)
        print(f"Generated {len(financial)} financial records")
        
        attribution = self.generate_attribution_data(campaigns, keywords)
        print(f"Generated {len(attribution)} attribution records")
        
        competitive = self.generate_competitive_data(keywords)
        print(f"Generated {len(competitive)} competitive records")
        
        return {
            'campaigns': campaigns,
            'keywords': keywords,
            'products': products,
            'financial': financial,
            'attribution': attribution,
            'competitive': competitive
        }

def save_sample_data_to_csv(data_dict: dict, output_dir: str = "sample_data/"):
    """Save generated data to CSV files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for data_type, df in data_dict.items():
        filename = f"{output_dir}{data_type}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {data_type} data to {filename}")

if __name__ == "__main__":
    # Generate and save sample data
    generator = SampleDataGenerator()
    sample_data = generator.generate_all_sample_data(days=90)
    save_sample_data_to_csv(sample_data)
    
    # Display summary statistics
    print("\n=== Sample Data Summary ===")
    for data_type, df in sample_data.items():
        print(f"\n{data_type.upper()}:")
        print(f"  Records: {len(df):,}")
        if 'spend' in df.columns:
            print(f"  Total Spend: ${df['spend'].sum():,.2f}")
        if 'sales' in df.columns:
            print(f"  Total Sales: ${df['sales'].sum():,.2f}")
        if 'date' in df.columns:
            print(f"  Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Columns: {list(df.columns)}")
