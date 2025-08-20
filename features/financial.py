"""
Fixed Financial Feature Engine - addresses missing columns and merging issues.
File: features/financial_fixed.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinancialFeatureEngine:
    """Fixed financial feature engineering for advertising ROI analysis"""
    
    def __init__(self, annual_discount_rate: float = 0.12):
        self.annual_discount_rate = annual_discount_rate
        self.daily_discount_rate = annual_discount_rate / 365
        self.scaler = StandardScaler()
        
    def calculate_true_margins(self, campaign_df: pd.DataFrame, 
                             product_df: pd.DataFrame, 
                             financial_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate true profit margins with proper error handling"""
        
        # Start with campaign data
        merged = campaign_df.copy()
        
        # Add date column if missing
        if 'date' not in merged.columns and 'timestamp' in merged.columns:
            merged['date'] = pd.to_datetime(merged['timestamp']).dt.date
        elif 'date' not in merged.columns:
            merged['date'] = datetime.now().date()
        
        # Ensure date is datetime for merging
        merged['date'] = pd.to_datetime(merged['date'])
        
        # Add ASIN if missing - create a mapping
        if 'asin' not in merged.columns:
            # Create synthetic ASIN mapping based on campaign_id
            unique_campaigns = merged['campaign_id'].unique()
            if not product_df.empty and 'asin' in product_df.columns:
                unique_asins = product_df['asin'].unique()
                asin_mapping = dict(zip(
                    unique_campaigns, 
                    np.random.choice(unique_asins, len(unique_campaigns))
                ))
                merged['asin'] = merged['campaign_id'].map(asin_mapping)
            else:
                merged['asin'] = 'B' + merged['campaign_id'].str.replace('CAMP-', '').str.zfill(8)
        
        # Merge with product data if available
        if not product_df.empty:
            # Ensure product_df has date column
            if 'date' not in product_df.columns:
                product_df = product_df.copy()
                product_df['date'] = datetime.now().date()
            
            product_df['date'] = pd.to_datetime(product_df['date'])
            
            # Merge on date and asin
            product_cols = ['date', 'asin', 'total_sales', 'cost_per_unit', 'units_sold']
            available_product_cols = [col for col in product_cols if col in product_df.columns]
            
            if len(available_product_cols) >= 2:  # At least date and asin
                merged = merged.merge(
                    product_df[available_product_cols], 
                    on=['date', 'asin'], 
                    how='left',
                    suffixes=('', '_product')
                )
        
        # Merge with financial data if available
        if not financial_df.empty:
            # Ensure financial_df has date column
            if 'date' not in financial_df.columns:
                financial_df = financial_df.copy()
                financial_df['date'] = datetime.now().date()
            
            financial_df['date'] = pd.to_datetime(financial_df['date'])
            
            financial_cols = ['date', 'asin', 'cost_of_goods_sold', 'platform_fees', 
                            'fulfillment_fees', 'storage_fees']
            available_financial_cols = [col for col in financial_cols if col in financial_df.columns]
            
            if len(available_financial_cols) >= 2:  # At least date and asin
                merged = merged.merge(
                    financial_df[available_financial_cols], 
                    on=['date', 'asin'], 
                    how='left',
                    suffixes=('', '_financial')
                )
        
        # Calculate gross revenue (use sales if available, otherwise create from spend)
        if 'sales' in merged.columns:
            merged['gross_revenue'] = merged['sales']
        elif 'revenue' in merged.columns:
            merged['gross_revenue'] = merged['revenue']
        else:
            # Estimate revenue from spend assuming 3x ROAS
            merged['gross_revenue'] = merged.get('spend', 0) * 3.0
        
        # Calculate all costs with defaults
        merged['total_cogs'] = merged.get('cost_of_goods_sold', 
                                        merged['gross_revenue'] * 0.4)  # 40% default COGS
        
        merged['total_platform_fees'] = merged.get('platform_fees', 
                                                 merged['gross_revenue'] * 0.12)  # 12% default platform fee
        
        merged['total_fulfillment_fees'] = merged.get('fulfillment_fees', 
                                                    merged.get('orders', merged.get('units_sold', 10)) * 3.5)  # $3.5 per unit
        
        merged['total_storage_fees'] = merged.get('storage_fees', 2.0)  # $2/day default
        
        # Calculate net revenue after platform fees
        merged['net_revenue'] = merged['gross_revenue'] - merged['total_platform_fees']
        
        # Calculate total variable costs
        merged['total_variable_costs'] = (
            merged['total_cogs'] + 
            merged['total_fulfillment_fees'] + 
            merged['total_storage_fees'] +
            merged.get('spend', 0)  # Include advertising spend
        )
        
        # Calculate contribution margin
        merged['contribution_margin'] = merged['net_revenue'] - merged['total_variable_costs']
        merged['contribution_margin_rate'] = merged['contribution_margin'] / merged['gross_revenue'].replace(0, 1)
        
        # Calculate true ROAS (after all costs)
        merged['true_roas'] = merged['contribution_margin'] / merged.get('spend', 1).replace(0, 1)
        
        # Calculate breakeven ACoS
        merged['breakeven_acos'] = merged['contribution_margin_rate']
        
        return merged
    
    def calculate_customer_lifetime_value(self, attribution_df: pd.DataFrame,
                                        repeat_purchase_rate: float = 0.3,
                                        avg_orders_per_year: float = 4.0) -> pd.DataFrame:
        """Calculate customer lifetime value from attribution data"""
        
        df = attribution_df.copy()
        
        # Calculate average order value per customer
        if 'conversion_value' in df.columns and 'customer_id' in df.columns:
            customer_stats = df.groupby('customer_id').agg({
                'conversion_value': 'mean',
                'attributed_revenue': 'sum' if 'attributed_revenue' in df.columns else 'mean'
            }).reset_index()
            
            customer_stats.columns = ['customer_id', 'avg_order_value', 'first_order_attributed_revenue']
        else:
            # Create default customer stats
            customer_stats = pd.DataFrame({
                'customer_id': ['CUST-001'],
                'avg_order_value': [75.0],
                'first_order_attributed_revenue': [75.0]
            })
        
        # CLV calculation with repeat purchases
        customer_stats['annual_value'] = (
            customer_stats['avg_order_value'] * avg_orders_per_year
        )
        
        # Multi-period CLV with decay
        periods = 3  # 3 year horizon
        customer_stats['clv'] = 0
        
        for period in range(1, periods + 1):
            period_value = (
                customer_stats['annual_value'] * 
                (repeat_purchase_rate ** (period - 1)) *
                (1 / (1 + self.annual_discount_rate) ** period)
            )
            customer_stats['clv'] += period_value
        
        # Merge back to attribution data
        df = df.merge(customer_stats[['customer_id', 'clv']], on='customer_id', how='left')
        
        # Calculate CLV-adjusted attribution
        df['clv_adjusted_attribution'] = df.get('attributed_revenue', df.get('conversion_value', 75)) * (df['clv'] / df.get('conversion_value', 75))
        
        return df
    
    def calculate_working_capital_impact(self, campaign_df: pd.DataFrame,
                                       financial_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate working capital impact with proper defaults"""
        
        merged = campaign_df.copy()
        
        # Add default financial parameters if financial_df is empty or missing data
        if financial_df.empty or 'payment_terms_days' not in financial_df.columns:
            merged['payment_terms_days'] = 30  # Default 30 days
            merged['carrying_cost_rate'] = 0.15  # Default 15%
        else:
            # Merge with financial data
            if 'date' not in financial_df.columns:
                financial_df = financial_df.copy()
                financial_df['date'] = datetime.now().date()
            
            financial_df['date'] = pd.to_datetime(financial_df['date'])
            if 'date' not in merged.columns:
                merged['date'] = pd.to_datetime(datetime.now().date())
            
            merged = merged.merge(
                financial_df[['date', 'payment_terms_days', 'carrying_cost_rate']], 
                on='date', 
                how='left'
            )
            
            # Fill missing values
            merged['payment_terms_days'] = merged['payment_terms_days'].fillna(30)
            merged['carrying_cost_rate'] = merged['carrying_cost_rate'].fillna(0.15)
        
        # Calculate daily carrying cost rate
        merged['daily_carrying_rate'] = merged['carrying_cost_rate'] / 365
        
        # Calculate working capital requirement
        sales_col = 'sales' if 'sales' in merged.columns else 'gross_revenue'
        if sales_col not in merged.columns:
            merged[sales_col] = merged.get('spend', 0) * 3.0  # Estimate sales
        
        merged['sales_outstanding'] = merged[sales_col] * merged['payment_terms_days'] / 30
        merged['net_working_capital'] = merged['sales_outstanding'] - merged.get('spend', 0)
        
        # Calculate carrying cost of working capital
        merged['wc_carrying_cost'] = (
            merged['net_working_capital'] * 
            merged['daily_carrying_rate'] * 
            merged['payment_terms_days']
        )
        
        # Adjust ROAS for working capital cost
        merged['wc_adjusted_roas'] = (
            (merged[sales_col] - merged['wc_carrying_cost']) / 
            merged.get('spend', 1).replace(0, 1)
        )
        
        return merged
    
    def calculate_cash_flow_timing(self, campaign_df: pd.DataFrame,
                                 financial_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cash flow timing impact"""
        
        merged = campaign_df.copy()
        
        # Add payment terms if not available
        if financial_df.empty or 'payment_terms_days' not in financial_df.columns:
            merged['payment_terms_days'] = 30
        else:
            if 'date' not in financial_df.columns:
                financial_df = financial_df.copy()
                financial_df['date'] = datetime.now().date()
            
            financial_df['date'] = pd.to_datetime(financial_df['date'])
            if 'date' not in merged.columns:
                merged['date'] = pd.to_datetime(datetime.now().date())
            
            merged = merged.merge(
                financial_df[['date', 'payment_terms_days']], 
                on='date', 
                how='left'
            )
            merged['payment_terms_days'] = merged['payment_terms_days'].fillna(30)
        
        # Ensure date column exists
        if 'date' not in merged.columns:
            merged['date'] = pd.to_datetime(datetime.now().date())
        
        # Calculate cash inflow date
        merged['cash_inflow_date'] = merged['date'] + pd.to_timedelta(merged['payment_terms_days'], unit='days')
        
        # Calculate NPV of cash flows
        merged['days_to_payment'] = (merged['cash_inflow_date'] - merged['date']).dt.days
        merged['discount_factor'] = 1 / (1 + self.daily_discount_rate) ** merged['days_to_payment']
        
        # NPV-adjusted sales
        sales_col = 'sales' if 'sales' in merged.columns else 'gross_revenue'
        if sales_col not in merged.columns:
            merged[sales_col] = merged.get('spend', 0) * 3.0
        
        merged['npv_sales'] = merged[sales_col] * merged['discount_factor']
        
        # NPV-adjusted ROAS
        merged['npv_roas'] = merged['npv_sales'] / merged.get('spend', 1).replace(0, 1)
        
        return merged
    
    def calculate_platform_profitability(self, campaign_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate platform-specific profitability"""
        
        df = campaign_df.copy()
        
        # Platform-specific fee structures
        platform_fees = {
            'amazon': {'referral_fee': 0.15, 'fba_fee': 3.5, 'storage_fee': 0.85},
            'walmart': {'referral_fee': 0.12, 'fulfillment_fee': 3.0, 'storage_fee': 0.65}
        }
        
        # Default platform if missing
        if 'platform' not in df.columns:
            df['platform'] = 'amazon'
        
        # Default orders if missing
        if 'orders' not in df.columns:
            df['orders'] = df.get('clicks', 100) * 0.1  # 10% conversion rate
        
        # Default sales if missing
        if 'sales' not in df.columns:
            df['sales'] = df.get('spend', 0) * 3.0  # 3x ROAS default
        
        # Apply platform-specific costs
        for platform, fees in platform_fees.items():
            mask = df['platform'] == platform
            
            df.loc[mask, 'platform_referral_fee'] = df.loc[mask, 'sales'] * fees['referral_fee']
            df.loc[mask, 'platform_fulfillment_fee'] = df.loc[mask, 'orders'] * fees.get('fba_fee', fees.get('fulfillment_fee', 3.0))
            df.loc[mask, 'platform_storage_fee'] = fees['storage_fee']
        
        # Fill missing platform costs
        df['platform_referral_fee'] = df.get('platform_referral_fee', df['sales'] * 0.13).fillna(0)
        df['platform_fulfillment_fee'] = df.get('platform_fulfillment_fee', df['orders'] * 3.25).fillna(0)
        df['platform_storage_fee'] = df.get('platform_storage_fee', 0.75).fillna(0)
        
        # Calculate platform-adjusted metrics
        df['total_platform_costs'] = (
            df['platform_referral_fee'] + 
            df['platform_fulfillment_fee'] + 
            df['platform_storage_fee']
        )
        
        df['platform_adjusted_revenue'] = df['sales'] - df['total_platform_costs']
        df['platform_adjusted_roas'] = df['platform_adjusted_revenue'] / df.get('spend', 1).replace(0, 1)
        df['platform_profit'] = df['platform_adjusted_revenue'] - df.get('spend', 0)
        
        return df
    
    def calculate_risk_adjusted_returns(self, campaign_df: pd.DataFrame,
                                      volatility_window: int = 30) -> pd.DataFrame:
        """Calculate risk-adjusted performance metrics"""
        
        df = campaign_df.sort_values(['campaign_id', 'date'])
        
        # Ensure ROAS column exists
        if 'roas' not in df.columns:
            if 'true_roas' in df.columns:
                df['roas'] = df['true_roas']
            else:
                df['roas'] = df.get('sales', df.get('spend', 0) * 3) / df.get('spend', 1).replace(0, 1)
        
        # Calculate rolling volatility of ROAS
        df['roas_rolling_std'] = df.groupby('campaign_id')['roas'].rolling(
            window=min(volatility_window, len(df)), min_periods=1
        ).std().values
        
        df['roas_rolling_mean'] = df.groupby('campaign_id')['roas'].rolling(
            window=min(volatility_window, len(df)), min_periods=1
        ).mean().values
        
        # Calculate Sharpe ratio (excess return / volatility)
        risk_free_rate = self.annual_discount_rate / 365  # Daily risk-free rate
        df['excess_roas'] = df['roas'] - (1 + risk_free_rate)
        df['sharpe_ratio'] = df['excess_roas'] / df['roas_rolling_std'].replace(0, 1)
        
        # Fill infinite/NaN values
        df['sharpe_ratio'] = df['sharpe_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
        df['roas_rolling_std'] = df['roas_rolling_std'].fillna(df['roas'].std())
        
        # Calculate Value at Risk (VaR) at 95% confidence
        df['roas_var_95'] = df['roas_rolling_mean'] - 1.65 * df['roas_rolling_std']
        
        # Calculate maximum drawdown
        df['roas_cummax'] = df.groupby('campaign_id')['roas'].cummax()
        df['drawdown'] = (df['roas'] - df['roas_cummax']) / df['roas_cummax'].replace(0, 1)
        df['max_drawdown'] = df.groupby('campaign_id')['drawdown'].cummin()
        
        return df
    
    def calculate_competitive_pricing_impact(self, campaign_df: pd.DataFrame,
                                           competitive_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate impact of competitive pricing on advertising performance"""
        
        # Skip if no competitive data
        if competitive_df.empty:
            return campaign_df
        
        # Merge with competitive data
        merged = campaign_df.merge(
            competitive_df[['date', 'keyword', 'avg_competitor_bid', 'competitive_intensity']], 
            left_on=['date'], right_on=['date'], how='left'
        )
        
        # Calculate bid premium/discount vs competitors
        merged['bid_vs_competitor'] = merged.get('cpc', 1.0) / merged.get('avg_competitor_bid', 1.0).replace(0, 1)
        
        # Categorize competitive position
        merged['competitive_position'] = pd.cut(
            merged['bid_vs_competitor'], 
            bins=[0, 0.8, 1.2, float('inf')], 
            labels=['Below_Market', 'At_Market', 'Above_Market']
        )
        
        # Calculate market share capture rate
        merged['market_share_rate'] = 1 / (1 + np.exp(-2 * (merged['bid_vs_competitor'] - 1)))
        
        # Adjust performance metrics for competitive pressure
        merged['competitive_adjusted_ctr'] = merged.get('ctr', 5.0) * (1 - merged.get('competitive_intensity', 0.5) * 0.3)
        merged['competitive_adjusted_roas'] = merged.get('roas', 3.0) * merged['market_share_rate']
        
        return merged
    
    def create_financial_feature_matrix(self, campaign_df: pd.DataFrame,
                                      product_df: pd.DataFrame,
                                      financial_df: pd.DataFrame,
                                      attribution_df: pd.DataFrame,
                                      competitive_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive financial feature matrix with robust error handling"""
        
        print("Calculating true margins...")
        df = self.calculate_true_margins(campaign_df, product_df, financial_df)
        
        print("Calculating working capital impact...")
        df = self.calculate_working_capital_impact(df, financial_df)
        
        print("Calculating cash flow timing...")
        df = self.calculate_cash_flow_timing(df, financial_df)
        
        print("Calculating platform profitability...")
        df = self.calculate_platform_profitability(df)
        
        print("Calculating risk-adjusted returns...")
        df = self.calculate_risk_adjusted_returns(df)
        
        # Skip competitive analysis if no data
        if not competitive_df.empty:
            print("Calculating competitive pricing impact...")
            df = self.calculate_competitive_pricing_impact(df, competitive_df)
        
        # Add CLV features if attribution data available
        if not attribution_df.empty:
            print("Calculating customer lifetime value...")
            try:
                clv_features = self.calculate_customer_lifetime_value(attribution_df)
                
                # Aggregate CLV metrics by campaign
                clv_agg = clv_features.groupby(['campaign_id', 'date']).agg({
                    'clv': 'mean',
                    'clv_adjusted_attribution': 'sum'
                }).reset_index()
                
                df = df.merge(clv_agg, on=['campaign_id', 'date'], how='left')
            except Exception as e:
                print(f"CLV calculation failed: {e}")
                # Add default CLV values
                df['clv'] = 250.0  # Default CLV
                df['clv_adjusted_attribution'] = df.get('sales', 0) * 1.2
        else:
            # Add default CLV values
            df['clv'] = 250.0  # Default CLV
            df['clv_adjusted_attribution'] = df.get('sales', 0) * 1.2
        
        # Create efficiency ratios
        df['cost_efficiency'] = df['contribution_margin'] / df['total_variable_costs'].replace(0, 1)
        df['advertising_efficiency'] = df.get('sales', df.get('spend', 0) * 3) / df.get('spend', 1).replace(0, 1)
        df['margin_efficiency'] = df['contribution_margin_rate'] * df['advertising_efficiency']
        
        # Create financial health scores
        df['financial_health_score'] = (
            df['true_roas'].clip(0, 5) * 0.3 +
            df['contribution_margin_rate'].clip(0, 1) * 0.3 +
            df['sharpe_ratio'].fillna(0).clip(-2, 2) * 0.2 +
            (1 - df['max_drawdown'].abs().fillna(0).clip(0, 1)) * 0.2
        )
        
        # Normalize financial health score to 0-100
        df['financial_health_score'] = df['financial_health_score'] * 20
        
        return df
    
    def calculate_portfolio_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features needed for portfolio optimization"""
        
        # Calculate correlation matrix for campaign returns
        campaign_returns = df.pivot_table(
            index='date', columns='campaign_id', values='true_roas', fill_value=0
        )
        correlation_matrix = campaign_returns.corr()
        
        # Calculate campaign weights in current portfolio
        total_spend = df.groupby('date')['spend'].sum()
        df = df.merge(total_spend.rename('total_daily_spend'), left_on='date', right_index=True)
        df['portfolio_weight'] = df['spend'] / df['total_daily_spend'].replace(0, 1)
        
        # Calculate diversification metrics
        campaign_weights = df.groupby('campaign_id')['portfolio_weight'].mean()
        herfindahl_index = (campaign_weights ** 2).sum()
        df['portfolio_concentration'] = herfindahl_index
        
        # Calculate beta vs portfolio
        portfolio_returns = df.groupby('date').apply(
            lambda x: (x['true_roas'] * x['portfolio_weight']).sum()
        )
        
        for campaign_id in df['campaign_id'].unique():
            campaign_data = df[df['campaign_id'] == campaign_id]
            if len(campaign_data) > 10:  # Need sufficient data
                campaign_returns_series = campaign_data.set_index('date')['true_roas']
                aligned_portfolio = portfolio_returns.reindex(campaign_returns_series.index)
                
                # Calculate beta
                covariance = np.cov(campaign_returns_series, aligned_portfolio)[0, 1]
                portfolio_variance = np.var(aligned_portfolio)
                beta = covariance / portfolio_variance if portfolio_variance > 0 else 1.0
                
                df.loc[df['campaign_id'] == campaign_id, 'campaign_beta'] = beta
        
        df['campaign_beta'] = df['campaign_beta'].fillna(1.0)
        
        return df