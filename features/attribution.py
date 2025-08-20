"""
Fixed Attribution feature engineering for multi-touch attribution.
File: features/attribution.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.special import comb
from sklearn.preprocessing import StandardScaler
import itertools

class AttributionFeatureEngine:
    """Fixed Attribution feature engineering for multi-touch attribution"""
    
    def __init__(self, decay_rate: float = 0.1, window_days: int = 30):
        self.decay_rate = decay_rate
        self.window_days = window_days
        self.scaler = StandardScaler()
        
    def calculate_time_decay_weights(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-decay attribution weights"""
        df = touchpoint_df.copy()
        
        # Handle missing conversion_timestamp
        if 'conversion_timestamp' not in df.columns:
            latest_timestamps = df.groupby('customer_id')['timestamp'].max()
            df = df.merge(
                latest_timestamps.rename('conversion_timestamp'), 
                left_on='customer_id', 
                right_index=True, 
                how='left'
            )
        
        # Calculate days between touchpoint and conversion
        df['days_to_conversion'] = (df['conversion_timestamp'] - df['timestamp']).dt.days
        df['days_to_conversion'] = df['days_to_conversion'].fillna(0).abs()
        
        # Time decay weight
        df['time_decay_weight'] = np.exp(-self.decay_rate * df['days_to_conversion'])
        
        # Normalize weights within each customer journey
        df['normalized_weight'] = df.groupby('customer_id')['time_decay_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        
        return df
    
    def calculate_position_weights(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-based attribution weights"""
        df = touchpoint_df.copy()
        
        df = df.sort_values(['customer_id', 'timestamp'])
        df['touchpoint_position'] = df.groupby('customer_id').cumcount() + 1
        df['total_touchpoints'] = df.groupby('customer_id')['touchpoint_position'].transform('max')
        
        def position_weight(row):
            if row['total_touchpoints'] == 1:
                return 1.0
            elif row['touchpoint_position'] == 1:
                return 0.4
            elif row['touchpoint_position'] == row['total_touchpoints']:
                return 0.4
            else:
                middle_count = row['total_touchpoints'] - 2
                return 0.2 / middle_count if middle_count > 0 else 0.0
        
        df['position_weight'] = df.apply(position_weight, axis=1)
        return df
    
    def calculate_shapley_values(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simplified Shapley value attribution"""
        df = touchpoint_df.copy()
        
        if 'conversion_value' not in df.columns:
            df['conversion_value'] = 50.0
        
        shapley_values = []
        
        for customer_id, journey in df.groupby('customer_id'):
            shapley_value = 1.0 / len(journey)
            for _, row in journey.iterrows():
                shapley_values.append({
                    'customer_id': customer_id,
                    'touchpoint_id': row['touchpoint_id'],
                    'shapley_value': shapley_value
                })
        
        shapley_df = pd.DataFrame(shapley_values)
        df = df.merge(shapley_df, on=['customer_id', 'touchpoint_id'], how='left')
        df['shapley_value'] = df['shapley_value'].fillna(0)
        
        return df
    
    def calculate_markov_chain_attribution(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simplified Markov chain attribution weights"""
        df = touchpoint_df.copy()
        
        touchpoint_type_value = {
            'click': 0.5,
            'view': 0.3,
            'impression': 0.2
        }
        
        df['markov_weight'] = df['touchpoint_type'].map(touchpoint_type_value).fillna(0.1)
        df['markov_weight_normalized'] = df.groupby('customer_id')['markov_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        
        return df
    
    def create_attribution_features(self, touchpoint_df: pd.DataFrame, 
                                  campaign_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive attribution feature set with error handling"""
        
        df = touchpoint_df.copy()
        
        # CRITICAL FIX: Add conversion_value if missing
        if 'conversion_value' not in df.columns:
            df['conversion_value'] = 50.0
        
        # Calculate attribution weights
        df = self.calculate_time_decay_weights(df)
        df = self.calculate_position_weights(df)
        df = self.calculate_shapley_values(df)
        df = self.calculate_markov_chain_attribution(df)
        
        # CRITICAL FIX: Calculate attributed revenue BEFORE other operations
        df['attributed_revenue'] = df['conversion_value'] * df.get('normalized_weight', 1.0)
        
        # Add campaign features if available
        if not campaign_df.empty and 'campaign_id' in campaign_df.columns:
            campaign_features = campaign_df.groupby('campaign_id').agg({
                'spend': 'mean',
                'roas': 'mean',
                'acos': 'mean',
                'ctr': 'mean',
                'conversion_rate': 'mean'
            }).reset_index()
            
            campaign_features.columns = ['campaign_id'] + [f'campaign_{col}' for col in campaign_features.columns[1:]]
            df = df.merge(campaign_features, on='campaign_id', how='left')
            
            # Fill missing values
            for col in [c for c in df.columns if c.startswith('campaign_')]:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        else:
            # Default campaign features
            df['campaign_spend'] = 100.0
            df['campaign_roas'] = 3.0
            df['campaign_acos'] = 0.33
            df['campaign_ctr'] = 5.0
            df['campaign_conversion_rate'] = 10.0
        
        # Create interaction features
        df['spend_x_time_weight'] = df['campaign_spend'] * df.get('time_decay_weight', 1.0)
        df['roas_x_position_weight'] = df['campaign_roas'] * df.get('position_weight', 1.0)
        df['ctr_x_shapley'] = df['campaign_ctr'] * df.get('shapley_value', 1.0)
        
        # Journey-level features
        journey_features = df.groupby('customer_id').agg({
            'touchpoint_position': 'max',
            'time_decay_weight': 'sum',
            'days_to_conversion': 'max',
            'conversion_value': 'first'
        }).reset_index()
        
        journey_features.columns = ['customer_id', 'journey_length', 'total_weighted_touches', 
                                  'journey_duration_days', 'order_value']
        
        df = df.merge(journey_features, on='customer_id', how='left')
        
        # Required ML features
        df['is_first_touch'] = (df['touchpoint_position'] == 1).astype(int)
        df['is_last_touch'] = (df['touchpoint_position'] == df['journey_length']).astype(int)
        df['is_middle_touch'] = ((df['touchpoint_position'] > 1) & 
                               (df['touchpoint_position'] < df['journey_length'])).astype(int)
        
        df['platform_amazon'] = (df['platform'] == 'amazon').astype(int)
        df['platform_walmart'] = (df['platform'] == 'walmart').astype(int)
        df['touchpoint_click'] = (df['touchpoint_type'] == 'click').astype(int)
        df['touchpoint_view'] = (df['touchpoint_type'] == 'view').astype(int)
        df['touchpoint_impression'] = (df['touchpoint_type'] == 'impression').astype(int)
        
        return df
    
    def create_ensemble_attribution(self, df: pd.DataFrame, 
                                  weights: Dict[str, float] = None) -> pd.DataFrame:
        """Create ensemble attribution combining multiple models"""
        
        if weights is None:
            weights = {
                'time_decay': 0.3,
                'position': 0.2,
                'shapley': 0.3,
                'markov': 0.2
            }
        
        # Ensure all required columns exist
        required_cols = ['normalized_weight', 'position_weight', 'shapley_value', 'markov_weight_normalized']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.25
        
        # Ensemble weight
        df['ensemble_weight'] = (
            df['normalized_weight'] * weights['time_decay'] +
            df['position_weight'] * weights['position'] +
            df['shapley_value'] * weights['shapley'] +
            df['markov_weight_normalized'] * weights['markov']
        )
        
        # Normalize ensemble weights
        df['ensemble_weight_normalized'] = df.groupby('customer_id')['ensemble_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        
        # Update attributed revenue
        df['attributed_revenue'] = df['conversion_value'] * df['ensemble_weight_normalized']
        
        return df
    