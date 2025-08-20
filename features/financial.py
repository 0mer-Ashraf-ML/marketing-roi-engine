"""
Financial feature engineering for advertising ROI optimization.
File: features/financial.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinancialFeatureEngine:
    """Advanced financial feature engineering for advertising ROI analysis"""
    
    def __init__(self, annual_discount_rate: float = 0.12):
        self.annual_discount_rate = annual_discount_rate
        self.daily_discount_rate = annual_discount_rate / 365
        self.scaler = StandardScaler()
        
    def calculate_true_margins(self, campaign_df: pd.DataFrame, 
                             product_df: pd.DataFrame, 
                             financial_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate true profit margins after all costs and fees"""
        
        # Merge campaign data with product sales
        merged = campaign_df.merge(
            product_df[['date', 'asin', 'total_sales', 'cost_per_unit', 'units_sold']], 
            on='date', how='left'
        )
        
        # Merge with financial data
        merged = merged.merge(
            financial_df[['date', 'asin', 'cost_of_goods_sold', 'platform_fees', 
                         'fulfillment_fees', 'storage_fees']], 
            on=['date', 'asin'], how='left'
        )
        
        # Calculate gross revenue
        merged['gross_revenue'] = merged['sales']
        
        # Calculate all costs
        merged['total_cogs'] = merged['cost_of_goods_sold'].fillna(
            merged['cost_per_unit'] * merged['units_sold']
        )
        merged['total_platform_fees'] = merged['platform_fees'].fillna(
            merged['gross_revenue'] * 0.12  # Default 12% platform fee
        )
        merged['total_fulfillment_fees'] = merged['fulfillment_fees'].fillna(
            merged['units_sold'] * 3.5  # Default $3.5 per unit
        )
        merged['total_storage_fees'] = merged['storage_fees'].fillna(2.0)  # Default $2/day
        
        # Calculate net revenue after platform fees
        merged['net_revenue'] = merged['gross_revenue'] - merged['total_platform_fees']
        
        # Calculate total variable costs
        merged['total_variable_costs'] = (
            merged['total_cogs'] + 
            merged['total_fulfillment_fees'] + 
            merged['total_storage_fees'] +
            merged['spend']  # Include advertising spend
        )
        
        # Calculate contribution margin
        merged['contribution_margin'] = merged['net_revenue'] - merged['total_variable_costs']
        merged['contribution_margin_rate'] = merged['contribution_margin'] / merged['gross_revenue'].replace(0, 1)
        
        # Calculate true ROAS (after all costs)
        merged['true_roas'] = merged['contribution_margin'] / merged['spend'].replace(0, 1)
        
        # Calculate breakeven ACoS
        merged['breakeven_acos'] = merged['contribution_margin_rate']
        
        return merged
    
    def calculate_customer_lifetime_value(self, attribution_df: pd.DataFrame,
                                        repeat_purchase_rate: float = 0.3,
                                        avg_orders_per_year: float = 4.0) -> pd.DataFrame:
        """Calculate customer lifetime value from attribution data"""
        
        df = attribution_df.copy()
        
        # Calculate average order value per customer
        customer_stats = df.groupby('customer_id').agg({
            'conversion_value': 'mean',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'avg_order_value', 'first_order_attributed_revenue']
        
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
        df['clv_adjusted_attribution'] = df['attributed_revenue'] * (df['clv'] / df['conversion_value'])
        
        return df
    
    def calculate_working_capital_impact(self, campaign_df: pd.DataFrame,
                                       financial_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate advertising impact on working capital requirements"""
        
        merged = campaign_df.merge(
            financial_df[['date', 'asin', 'payment_terms_days', 'carrying_cost_rate']], 
            on='date', how='left'
        )
        
        # Fill missing values
        merged['payment_terms_days'] = merged['payment_terms_days'].fillna(30)
        merged['carrying_cost_rate'] = merged['carrying_cost_rate'].fillna(0.15)
        
        # Calculate daily carrying cost rate
        merged['daily_carrying_rate'] = merged['carrying_cost_rate'] / 365
        
        # Calculate working capital requirement
        # WC = (Sales * Payment Terms) - (Ad Spend immediate outflow)
        merged['sales_outstanding'] = merged['sales'] * merged['payment_terms_days'] / 30
        merged['net_working_capital'] = merged['sales_outstanding'] - merged['spend']
        
        # Calculate carrying cost of working capital
        merged['wc_carrying_cost'] = (
            merged['net_working_capital'] * 
            merged['daily_carrying_rate'] * 
            merged['payment_terms_days']
        )
        
        # Adjust ROAS for working capital cost
        merged['wc_adjusted_roas'] = (
            (merged['sales'] - merged['wc_carrying_cost']) / merged['spend'].replace(0, 1)
        )
        
        return merged
    
    def calculate_cash_flow_timing(self, campaign_df: pd.DataFrame,
                                 financial_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cash flow timing impact of advertising"""
        
        merged = campaign_df.merge(
            financial_df[['date', 'asin', 'payment_terms_days']], 
            on='date', how='left'
        )
        
        merged['payment_terms_days'] = merged['payment_terms_days'].fillna(30)
        
        # Calculate cash inflow date
        merged['cash_inflow_date'] = merged['date'] + pd.to_timedelta(merged['payment_terms_days'], unit='days')
        
        # Calculate NPV of cash flows
        merged['days_to_payment'] = (merged['cash_inflow_date'] - merged['date']).dt.days
        merged['discount_factor'] = 1 / (1 + self.daily_discount_rate) ** merged['days_to_payment']
        
        # NPV-adjusted sales
        merged['npv_sales'] = merged['sales'] * merged['discount_factor']
        
        # NPV-adjusted ROAS
        merged['npv_roas'] = merged['npv_sales'] / merged['spend'].replace(0, 1)
        
        # Calculate payback period
        merged['cumulative_cash_flow'] = merged.groupby('campaign_id')['npv_sales'].cumsum() - merged.groupby('campaign_id')['spend'].cumsum()
        
        return merged
    
    def calculate_platform_profitability(self, campaign_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate profitability metrics by platform"""
        
        df = campaign_df.copy()
        
        # Platform-specific fee structures
        platform_fees = {
            'amazon': {'referral_fee': 0.15, 'fba_fee': 3.5, 'storage_fee': 0.85},
            'walmart': {'referral_fee': 0.12, 'fulfillment_fee': 3.0, 'storage_fee': 0.65}
        }
        
        # Apply platform-specific costs
        for platform, fees in platform_fees.items():
            mask = df['platform'] == platform
            
            df.loc[mask, 'platform_referral_fee'] = df.loc[mask, 'sales'] * fees['referral_fee']
            df.loc[mask, 'platform_fulfillment_fee'] = df.loc[mask, 'orders'] * fees['fba_fee']
            df.loc[mask, 'platform_storage_fee'] = fees['storage_fee']
        
        # Calculate platform-adjusted metrics
        df['total_platform_costs'] = (
            df['platform_referral_fee'].fillna(0) + 
            df['platform_fulfillment_fee'].fillna(0) + 
            df['platform_storage_fee'].fillna(0)
        )
        
        df['platform_adjusted_revenue'] = df['sales'] - df['total_platform_costs']
        df['platform_adjusted_roas'] = df['platform_adjusted_revenue'] / df['spend'].replace(0, 1)
        df['platform_profit'] = df['platform_adjusted_revenue'] - df['spend']
        
        return df
    
    def calculate_risk_adjusted_returns(self, campaign_df: pd.DataFrame,
                                      volatility_window: int = 30) -> pd.DataFrame:
        """Calculate risk-adjusted performance metrics"""
        
        df = campaign_df.sort_values(['campaign_id', 'date'])
        
        # Calculate rolling volatility of ROAS
        df['roas_rolling_std'] = df.groupby('campaign_id')['roas'].rolling(
            window=volatility_window, min_periods=7
        ).std().values
        
        df['roas_rolling_mean'] = df.groupby('campaign_id')['roas'].rolling(
            window=volatility_window, min_periods=7
        ).mean().values
        
        # Calculate Sharpe ratio (excess return / volatility)
        risk_free_rate = self.annual_discount_rate / 365  # Daily risk-free rate
        df['excess_roas'] = df['roas'] - (1 + risk_free_rate)
        df['sharpe_ratio'] = df['excess_roas'] / df['roas_rolling_std'].replace(0, 1)
        
        # Calculate Value at Risk (VaR) at 95% confidence
        df['roas_var_95'] = df['roas_rolling_mean'] - 1.65 * df['roas_rolling_std']
        
        # Calculate maximum drawdown
        df['roas_cummax'] = df.groupby('campaign_id')['roas'].cummax()
        df['drawdown'] = (df['roas'] - df['roas_cummax']) / df['roas_cummax']
        df['max_drawdown'] = df.groupby('campaign_id')['drawdown'].cummin()
        
        return df
    
    def calculate_competitive_pricing_impact(self, campaign_df: pd.DataFrame,
                                           competitive_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate impact of competitive pricing on advertising performance"""
        
        # Merge with competitive data
        merged = campaign_df.merge(
            competitive_df[['date', 'keyword', 'avg_competitor_bid', 'competitive_intensity']], 
            left_on=['date'], right_on=['date'], how='left'
        )
        
        # Calculate bid premium/discount vs competitors
        merged['bid_vs_competitor'] = merged['cpc'] / merged['avg_competitor_bid'].replace(0, 1)
        
        # Categorize competitive position
        merged['competitive_position'] = pd.cut(
            merged['bid_vs_competitor'], 
            bins=[0, 0.8, 1.2, float('inf')], 
            labels=['Below_Market', 'At_Market', 'Above_Market']
        )
        
        # Calculate market share capture rate
        merged['market_share_rate'] = 1 / (1 + np.exp(-2 * (merged['bid_vs_competitor'] - 1)))
        
        # Adjust performance metrics for competitive pressure
        merged['competitive_adjusted_ctr'] = merged['ctr'] * (1 - merged['competitive_intensity'] * 0.3)
        merged['competitive_adjusted_roas'] = merged['roas'] * merged['market_share_rate']
        
        return merged
    
    def create_financial_feature_matrix(self, campaign_df: pd.DataFrame,
                                      product_df: pd.DataFrame,
                                      financial_df: pd.DataFrame,
                                      attribution_df: pd.DataFrame,
                                      competitive_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive financial feature matrix"""
        
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
        
        print("Calculating competitive pricing impact...")
        if not competitive_df.empty:
            df = self.calculate_competitive_pricing_impact(df, competitive_df)
        
        # Add CLV features if attribution data available
        if not attribution_df.empty:
            print("Calculating customer lifetime value...")
            clv_features = self.calculate_customer_lifetime_value(attribution_df)
            
            # Aggregate CLV metrics by campaign
            clv_agg = clv_features.groupby(['campaign_id', 'date']).agg({
                'clv': 'mean',
                'clv_adjusted_attribution': 'sum'
            }).reset_index()
            
            df = df.merge(clv_agg, on=['campaign_id', 'date'], how='left')
        
        # Create efficiency ratios
        df['cost_efficiency'] = df['contribution_margin'] / df['total_variable_costs'].replace(0, 1)
        df['advertising_efficiency'] = df['sales'] / df['spend'].replace(0, 1)
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
        df['portfolio_weight'] = df['spend'] / df['total_daily_spend']
        
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