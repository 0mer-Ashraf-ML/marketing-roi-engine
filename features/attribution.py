"""
Attribution feature engineering for multi-touch attribution modeling.
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
    """Advanced attribution feature engineering for multi-touch attribution"""
    
    def __init__(self, decay_rate: float = 0.1, window_days: int = 30):
        self.decay_rate = decay_rate  # Time decay rate for attribution
        self.window_days = window_days  # Attribution window
        self.scaler = StandardScaler()
        
    def calculate_time_decay_weights(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-decay attribution weights"""
        df = touchpoint_df.copy()
        
        # Calculate days between touchpoint and conversion
        df['days_to_conversion'] = (df['conversion_timestamp'] - df['timestamp']).dt.days
        
        # Time decay weight: w = e^(-decay_rate * days)
        df['time_decay_weight'] = np.exp(-self.decay_rate * df['days_to_conversion'])
        
        # Normalize weights within each customer journey
        df['normalized_weight'] = df.groupby('customer_id')['time_decay_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        
        return df
    
    def calculate_position_weights(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-based attribution weights"""
        df = touchpoint_df.copy()
        
        # Sort touchpoints by timestamp within each customer journey
        df = df.sort_values(['customer_id', 'timestamp'])
        df['touchpoint_position'] = df.groupby('customer_id').cumcount() + 1
        df['total_touchpoints'] = df.groupby('customer_id')['touchpoint_position'].transform('max')
        
        # Position weights: First = 40%, Last = 40%, Middle = 20% split
        def position_weight(row):
            if row['total_touchpoints'] == 1:
                return 1.0
            elif row['touchpoint_position'] == 1:  # First touch
                return 0.4
            elif row['touchpoint_position'] == row['total_touchpoints']:  # Last touch
                return 0.4
            else:  # Middle touches
                middle_count = row['total_touchpoints'] - 2
                return 0.2 / middle_count if middle_count > 0 else 0.0
        
        df['position_weight'] = df.apply(position_weight, axis=1)
        
        return df
    
    def calculate_shapley_values(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Shapley value attribution for each touchpoint"""
        df = touchpoint_df.copy()
        shapley_values = []
        
        # Group by customer journey
        for customer_id, journey in df.groupby('customer_id'):
            if len(journey) == 1:
                shapley_values.append({'customer_id': customer_id, 
                                     'touchpoint_id': journey.iloc[0]['touchpoint_id'],
                                     'shapley_value': 1.0})
                continue
            
            touchpoints = journey['touchpoint_id'].tolist()
            conversion_value = journey['conversion_value'].iloc[0]
            
            # Calculate Shapley values for each touchpoint
            for touchpoint in touchpoints:
                shapley_value = self._calculate_single_shapley_value(
                    touchpoint, touchpoints, journey, conversion_value
                )
                shapley_values.append({
                    'customer_id': customer_id,
                    'touchpoint_id': touchpoint, 
                    'shapley_value': shapley_value
                })
        
        shapley_df = pd.DataFrame(shapley_values)
        df = df.merge(shapley_df, on=['customer_id', 'touchpoint_id'], how='left')
        
        return df
    
    def _calculate_single_shapley_value(self, touchpoint: str, all_touchpoints: List[str], 
                                      journey_df: pd.DataFrame, conversion_value: float) -> float:
        """Calculate Shapley value for a single touchpoint"""
        n = len(all_touchpoints)
        shapley_value = 0.0
        
        # For each possible subset not containing the touchpoint
        other_touchpoints = [tp for tp in all_touchpoints if tp != touchpoint]
        
        for r in range(n):
            for subset in itertools.combinations(other_touchpoints, r):
                subset_list = list(subset)
                subset_with_touchpoint = subset_list + [touchpoint]
                
                # Marginal contribution
                v_with = self._coalition_value(subset_with_touchpoint, journey_df, conversion_value)
                v_without = self._coalition_value(subset_list, journey_df, conversion_value)
                marginal_contribution = v_with - v_without
                
                # Weight by coalition size
                weight = 1.0 / (comb(n-1, len(subset_list)) * n)
                shapley_value += weight * marginal_contribution
        
        return shapley_value
    
    def _coalition_value(self, coalition: List[str], journey_df: pd.DataFrame, 
                        conversion_value: float) -> float:
        """Calculate the value function for a coalition of touchpoints"""
        if not coalition:
            return 0.0
        
        # Simple value function: diminishing returns to scale
        # More sophisticated models could use ML to predict conversion probability
        coalition_size = len(coalition)
        
        # Base conversion probability based on touchpoint types
        base_prob = 0.1  # 10% base conversion
        
        for touchpoint_id in coalition:
            touchpoint_data = journey_df[journey_df['touchpoint_id'] == touchpoint_id].iloc[0]
            touchpoint_type = touchpoint_data['touchpoint_type']
            
            # Different touchpoint types have different impact
            if touchpoint_type == 'click':
                base_prob += 0.3
            elif touchpoint_type == 'view':
                base_prob += 0.1
            else:  # impression
                base_prob += 0.05
        
        # Diminishing returns: additional touchpoints are less valuable
        diminishing_factor = 1.0 - 0.1 * (coalition_size - 1)
        final_prob = min(base_prob * diminishing_factor, 1.0)
        
        return final_prob * conversion_value
    
    def calculate_markov_chain_attribution(self, touchpoint_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Markov chain attribution weights"""
        df = touchpoint_df.copy()
        
        # Build transition matrix
        transitions = self._build_transition_matrix(df)
        
        # Calculate removal effects for each touchpoint type
        removal_effects = {}
        for touchpoint_type in df['touchpoint_type'].unique():
            removal_effects[touchpoint_type] = self._calculate_removal_effect(
                transitions, touchpoint_type
            )
        
        # Assign attribution weights based on removal effects
        df['markov_weight'] = df['touchpoint_type'].map(removal_effects)
        
        # Normalize within customer journeys
        df['markov_weight_normalized'] = df.groupby('customer_id')['markov_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        
        return df
    
    def _build_transition_matrix(self, df: pd.DataFrame) -> Dict:
        """Build transition probability matrix for Markov chain"""
        # Simplified transition matrix - in practice would be more sophisticated
        transitions = {}
        
        for customer_id, journey in df.groupby('customer_id'):
            journey_sorted = journey.sort_values('timestamp')
            touchpoint_sequence = journey_sorted['touchpoint_type'].tolist()
            
            for i in range(len(touchpoint_sequence) - 1):
                current_state = touchpoint_sequence[i]
                next_state = touchpoint_sequence[i + 1]
                
                if current_state not in transitions:
                    transitions[current_state] = {}
                if next_state not in transitions[current_state]:
                    transitions[current_state][next_state] = 0
                
                transitions[current_state][next_state] += 1
        
        # Convert counts to probabilities
        for current_state in transitions:
            total = sum(transitions[current_state].values())
            for next_state in transitions[current_state]:
                transitions[current_state][next_state] /= total
        
        return transitions
    
    def _calculate_removal_effect(self, transitions: Dict, removed_touchpoint: str) -> float:
        """Calculate the effect of removing a touchpoint type from the model"""
        # Simplified removal effect calculation
        # In practice, would simulate full Markov chain with/without touchpoint
        
        total_effect = 0.0
        if removed_touchpoint in transitions:
            for next_state, prob in transitions[removed_touchpoint].items():
                if next_state == 'conversion':
                    total_effect += prob
                else:
                    # Indirect effect through subsequent states
                    total_effect += prob * 0.5
        
        return total_effect
    
    def create_attribution_features(self, touchpoint_df: pd.DataFrame, 
                                  campaign_df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive attribution feature set"""
        
        # Calculate different attribution weights
        df = self.calculate_time_decay_weights(touchpoint_df)
        df = self.calculate_position_weights(df)
        df = self.calculate_shapley_values(df)
        df = self.calculate_markov_chain_attribution(df)
        
        # Add campaign-level features
        campaign_features = campaign_df.groupby('campaign_id').agg({
            'spend': 'mean',
            'roas': 'mean',
            'acos': 'mean',
            'ctr': 'mean',
            'conversion_rate': 'mean'
        }).reset_index()
        campaign_features.columns = ['campaign_id'] + [f'campaign_{col}' for col in campaign_features.columns[1:]]
        
        df = df.merge(campaign_features, on='campaign_id', how='left')
        
        # Create interaction features
        df['spend_x_time_weight'] = df['campaign_spend'] * df['time_decay_weight']
        df['roas_x_position_weight'] = df['campaign_roas'] * df['position_weight']
        df['ctr_x_shapley'] = df['campaign_ctr'] * df['shapley_value']
        
        # Create journey-level features
        journey_features = df.groupby('customer_id').agg({
            'touchpoint_position': 'max',  # Journey length
            'time_decay_weight': 'sum',    # Total weighted touches
            'days_to_conversion': 'max',   # Journey duration
            'conversion_value': 'first'    # Order value
        }).reset_index()
        
        journey_features.columns = ['customer_id', 'journey_length', 'total_weighted_touches', 
                                  'journey_duration_days', 'order_value']
        
        df = df.merge(journey_features, on='customer_id', how='left')
        
        # Touchpoint sequence features
        df['is_first_touch'] = (df['touchpoint_position'] == 1).astype(int)
        df['is_last_touch'] = (df['touchpoint_position'] == df['journey_length']).astype(int)
        df['is_middle_touch'] = ((df['touchpoint_position'] > 1) & 
                               (df['touchpoint_position'] < df['journey_length'])).astype(int)
        
        # Platform and channel features
        df['platform_amazon'] = (df['platform'] == 'amazon').astype(int)
        df['platform_walmart'] = (df['platform'] == 'walmart').astype(int)
        df['touchpoint_click'] = (df['touchpoint_type'] == 'click').astype(int)
        df['touchpoint_view'] = (df['touchpoint_type'] == 'view').astype(int)
        df['touchpoint_impression'] = (df['touchpoint_type'] == 'impression').astype(int)
        
        return df
    
    def calculate_incrementality(self, df: pd.DataFrame, 
                               test_campaigns: List[str], 
                               control_campaigns: List[str]) -> Dict:
        """Calculate incremental attribution using test/control methodology"""
        
        test_data = df[df['campaign_id'].isin(test_campaigns)]
        control_data = df[df['campaign_id'].isin(control_campaigns)]
        
        # Calculate conversion rates
        test_conversion_rate = test_data['conversion_value'].sum() / len(test_data)
        control_conversion_rate = control_data['conversion_value'].sum() / len(control_data)
        
        # Incremental lift
        incremental_lift = (test_conversion_rate - control_conversion_rate) / control_conversion_rate
        
        # Statistical significance test (simplified)
        from scipy import stats
        test_conversions = (test_data['conversion_value'] > 0).astype(int)
        control_conversions = (control_data['conversion_value'] > 0).astype(int)
        
        t_stat, p_value = stats.ttest_ind(test_conversions, control_conversions)
        
        return {
            'test_conversion_rate': test_conversion_rate,
            'control_conversion_rate': control_conversion_rate,
            'incremental_lift': incremental_lift,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
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
        
        # Normalize attribution weights to sum to 1 within each journey
        attribution_cols = ['time_decay_weight', 'position_weight', 'shapley_value', 'markov_weight_normalized']
        
        # Ensemble weight
        df['ensemble_weight'] = (
            df['normalized_weight'] * weights['time_decay'] +
            df['position_weight'] * weights['position'] +
            df['shapley_value'] * weights['shapley'] +
            df['markov_weight_normalized'] * weights['markov']
        )
        
        # Normalize ensemble weights within journeys
        df['ensemble_weight_normalized'] = df.groupby('customer_id')['ensemble_weight'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0
        )
        
        # Calculate attributed revenue
        df['attributed_revenue'] = df['conversion_value'] * df['ensemble_weight_normalized']
        
        return df
