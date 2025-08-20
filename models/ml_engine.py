"""
Machine Learning models for attribution, prediction, and optimization.
File: models/ml_engine.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Optimization libraries
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import cvxpy as cp

# Time series
from sklearn.linear_model import LinearRegression as Prophet_Substitute
import itertools

class AttributionMLModel:
    """Advanced ML model for multi-touch attribution"""
    
    def __init__(self):
        self.models = {
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'linear': Ridge(alpha=1.0)
        }
        self.ensemble_weights = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def prepare_features(self, attribution_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for attribution modeling"""
        
        # Select feature columns
        feature_cols = [
            'time_decay_weight', 'position_weight', 'touchpoint_position', 'journey_length',
            'journey_duration_days', 'campaign_spend', 'campaign_roas', 'campaign_ctr',
            'campaign_conversion_rate', 'is_first_touch', 'is_last_touch', 'is_middle_touch',
            'platform_amazon', 'platform_walmart', 'touchpoint_click', 'touchpoint_view',
            'spend_x_time_weight', 'roas_x_position_weight', 'total_weighted_touches'
        ]
        
        # Handle missing columns
        available_cols = [col for col in feature_cols if col in attribution_df.columns]
        X = attribution_df[available_cols].copy()
        
        # Fill missing values
        X = X.fillna(0)
        
        # Target variable: attributed revenue
        y = attribution_df['attributed_revenue'].values
        
        return X.values, y
    
    def train(self, attribution_df: pd.DataFrame) -> Dict[str, float]:
        """Train ensemble attribution model"""
        
        X, y = self.prepare_features(attribution_df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train individual models
        model_scores = {}
        predictions = {}
        
        for name, model in self.models.items():
            print(f"Training {name} model...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_scores[name] = score
            predictions[name] = y_pred
            print(f"{name} R² score: {score:.4f}")
        
        # Calculate ensemble weights based on performance
        total_score = sum(max(0, score) for score in model_scores.values())
        self.ensemble_weights = {
            name: max(0, score) / total_score 
            for name, score in model_scores.items()
        } if total_score > 0 else {name: 0.25 for name in self.models.keys()}
        
        # Evaluate ensemble
        ensemble_pred = self._ensemble_predict(predictions)
        ensemble_score = r2_score(y_test, ensemble_pred)
        
        print(f"Ensemble R² score: {ensemble_score:.4f}")
        print(f"Ensemble weights: {self.ensemble_weights}")
        
        self.is_trained = True
        
        return {
            'individual_scores': model_scores,
            'ensemble_score': ensemble_score,
            'ensemble_weights': self.ensemble_weights
        }
    
    def _ensemble_predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions using ensemble weights"""
        ensemble_pred = np.zeros(len(list(predictions.values())[0]))
        
        for name, pred in predictions.items():
            ensemble_pred += pred * self.ensemble_weights[name]
        
        return ensemble_pred
    
    def predict(self, attribution_df: pd.DataFrame) -> np.ndarray:
        """Predict attributed revenue"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.prepare_features(attribution_df)
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        return self._ensemble_predict(predictions)

class PerformancePredictionModel:
    """ML model for predicting campaign performance"""
    
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBRegressor(n_estimators=200, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=200, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42)
        }
        self.scalers = {}
        self.performance_models = {}  # Separate models for different metrics
        self.feature_importance = {}
        
    def prepare_performance_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Prepare features for performance prediction"""
        
        # Feature engineering for time series
        df = df.sort_values(['campaign_id', 'date'])
        
        # Lag features
        lag_features = ['spend', 'sales', 'roas', 'acos', 'clicks', 'impressions']
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 3, 7]:
                    df[f'{feature}_lag_{lag}'] = df.groupby('campaign_id')[feature].shift(lag)
        
        # Rolling features
        for feature in lag_features:
            if feature in df.columns:
                for window in [7, 14, 30]:
                    df[f'{feature}_rolling_{window}'] = df.groupby('campaign_id')[feature].rolling(
                        window=window, min_periods=1
                    ).mean().values
        
        # Day of week and month features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Seasonality features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Platform and campaign type encoding
        df['platform_encoded'] = LabelEncoder().fit_transform(df['platform'].astype(str))
        df['campaign_type_encoded'] = LabelEncoder().fit_transform(df['campaign_type'].astype(str))
        
        # Select feature columns
        feature_cols = [
            'spend', 'clicks', 'impressions', 'cpc', 'ctr',
            'platform_encoded', 'campaign_type_encoded',
            'day_of_week', 'month', 'quarter', 'sin_day', 'cos_day'
        ]
        
        # Add lag and rolling features
        feature_cols.extend([col for col in df.columns if '_lag_' in col or '_rolling_' in col])
        
        # Keep only available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].fillna(0)
        
        # Target variables
        targets = {
            'sales': df['sales'].values,
            'roas': df['roas'].values,
            'acos': df['acos'].values,
            'clicks': df['clicks'].values,
            'conversion_rate': df['conversion_rate'].values if 'conversion_rate' in df.columns else df['clicks'].values * 0
        }
        
        return X, targets
    
    def train_performance_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Train models for different performance metrics"""
        
        X, targets = self.prepare_performance_features(df)
        
        # Split data chronologically for time series
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        
        results = {}
        
        for target_name, y in targets.items():
            print(f"\nTraining models for {target_name}...")
            
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Handle zero/infinite values
            mask = np.isfinite(y_train) & (y_train >= 0)
            X_train_clean = X_train[mask]
            y_train_clean = y_train[mask]
            
            if len(y_train_clean) < 10:
                print(f"Insufficient data for {target_name}")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target_name] = scaler
            
            # Train models
            target_models = {}
            target_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    model.fit(X_train_scaled, y_train_clean)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Handle predictions
                    y_pred = np.clip(y_pred, 0, None)  # Ensure non-negative
                    
                    score = r2_score(y_test, y_pred)
                    target_models[model_name] = model
                    target_scores[model_name] = score
                    
                    print(f"  {model_name} R² score: {score:.4f}")
                    
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
                    target_scores[model_name] = -999
            
            self.performance_models[target_name] = target_models
            results[target_name] = target_scores
            
            # Feature importance for best model
            if target_scores:
                best_model_name = max(target_scores, key=target_scores.get)
                best_model = target_models[best_model_name]
                
                if hasattr(best_model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    self.feature_importance[target_name] = importance
                    print(f"  Top features: {importance.head(3)['feature'].tolist()}")
        
        return results
    
    def predict_performance(self, df: pd.DataFrame, 
                          target: str = 'sales', 
                          model_name: str = 'xgboost') -> np.ndarray:
        """Predict specific performance metric"""
        
        X, _ = self.prepare_performance_features(df)
        
        if target not in self.performance_models or model_name not in self.performance_models[target]:
            raise ValueError(f"Model not trained for {target} with {model_name}")
        
        scaler = self.scalers[target]
        model = self.performance_models[target][model_name]
        
        X_scaled = scaler.transform(X.fillna(0))
        predictions = model.predict(X_scaled)
        
        return np.clip(predictions, 0, None)  # Ensure non-negative

class BudgetOptimizationEngine:
    """Portfolio optimization engine for budget allocation"""
    
    def __init__(self, risk_aversion: float = 0.5):
        self.risk_aversion = risk_aversion
        self.expected_returns = {}
        self.covariance_matrix = None
        self.constraints = {}
        
    def calculate_expected_returns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate expected returns for each campaign"""
        
        # Use recent performance to estimate expected returns
        recent_data = df[df['date'] >= df['date'].max() - timedelta(days=30)]
        
        campaign_returns = recent_data.groupby('campaign_id').agg({
            'true_roas': 'mean',
            'sales': 'sum',
            'spend': 'sum',
            'sharpe_ratio': 'mean'
        }).reset_index()
        
        # Calculate risk-adjusted expected returns
        campaign_returns['risk_adjusted_return'] = (
            campaign_returns['true_roas'] * 
            (1 + campaign_returns['sharpe_ratio'].fillna(0) * 0.1)
        )
        
        self.expected_returns = dict(zip(
            campaign_returns['campaign_id'], 
            campaign_returns['risk_adjusted_return']
        ))
        
        return self.expected_returns
    
    def calculate_covariance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix of campaign returns"""
        
        # Pivot to get returns matrix
        returns_matrix = df.pivot_table(
            index='date', columns='campaign_id', values='true_roas', fill_value=0
        )
        
        # Calculate covariance matrix
        self.covariance_matrix = returns_matrix.cov().values
        
        return self.covariance_matrix
    
    def optimize_portfolio(self, total_budget: float, 
                         current_allocations: Dict[str, float] = None,
                         constraints: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Optimize budget allocation using mean-variance optimization"""
        
        if not self.expected_returns or self.covariance_matrix is None:
            raise ValueError("Must calculate expected returns and covariance matrix first")
        
        campaigns = list(self.expected_returns.keys())
        n_campaigns = len(campaigns)
        
        if n_campaigns == 0:
            return {'allocations': {}, 'expected_return': 0, 'risk': 0}
        
        # Define optimization variables
        weights = cp.Variable(n_campaigns)
        
        # Expected portfolio return
        expected_returns_array = np.array([self.expected_returns[c] for c in campaigns])
        portfolio_return = weights.T @ expected_returns_array
        
        # Portfolio risk (variance)
        portfolio_risk = cp.quad_form(weights, self.covariance_matrix)
        
        # Objective: maximize return - risk_aversion * risk
        objective = cp.Maximize(portfolio_return - self.risk_aversion * portfolio_risk)
        
        # Constraints
        constraints_list = [
            weights >= 0,  # No short selling
            cp.sum(weights) == 1,  # Budget constraint (weights sum to 1)
        ]
        
        # Additional constraints if specified
        if constraints:
            for campaign_id, constraint in constraints.items():
                if campaign_id in campaigns:
                    idx = campaigns.index(campaign_id)
                    
                    if 'min_allocation' in constraint:
                        constraints_list.append(weights[idx] >= constraint['min_allocation'])
                    
                    if 'max_allocation' in constraint:
                        constraints_list.append(weights[idx] <= constraint['max_allocation'])
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints_list)
        
        try:
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                
                # Convert to budget allocations
                allocations = {
                    campaigns[i]: float(optimal_weights[i] * total_budget)
                    for i in range(n_campaigns)
                }
                
                # Calculate expected portfolio metrics
                expected_return = float(portfolio_return.value)
                expected_risk = float(portfolio_risk.value)
                
                return {
                    'allocations': allocations,
                    'weights': dict(zip(campaigns, optimal_weights)),
                    'expected_return': expected_return,
                    'expected_risk': expected_risk,
                    'sharpe_ratio': expected_return / np.sqrt(expected_risk) if expected_risk > 0 else 0,
                    'status': 'optimal'
                }
            else:
                print(f"Optimization failed with status: {problem.status}")
                return {'status': 'failed', 'allocations': {}}
                
        except Exception as e:
            print(f"Optimization error: {e}")
            
            # Fallback: equal weight allocation
            equal_weight = total_budget / n_campaigns
            return {
                'allocations': {campaign: equal_weight for campaign in campaigns},
                'status': 'fallback',
                'error': str(e)
            }
    
    def simulate_allocation_scenarios(self, df: pd.DataFrame, 
                                    total_budget: float,
                                    scenarios: List[Dict[str, float]]) -> List[Dict]:
        """Simulate different allocation scenarios"""
        
        results = []
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')
            allocations = scenario.get('allocations', {})
            
            # Simulate performance
            simulated_performance = self._simulate_performance(df, allocations)
            
            results.append({
                'scenario_name': scenario_name,
                'allocations': allocations,
                'simulated_performance': simulated_performance
            })
        
        return results
    
    def _simulate_performance(self, df: pd.DataFrame, 
                            allocations: Dict[str, float]) -> Dict[str, float]:
        """Simulate performance given budget allocations"""
        
        total_expected_sales = 0
        total_expected_spend = sum(allocations.values())
        total_expected_risk = 0
        
        for campaign_id, budget in allocations.items():
            if campaign_id in self.expected_returns:
                expected_roas = self.expected_returns[campaign_id]
                expected_sales = budget * expected_roas
                total_expected_sales += expected_sales
        
        portfolio_roas = total_expected_sales / total_expected_spend if total_expected_spend > 0 else 0
        
        return {
            'expected_sales': total_expected_sales,
            'expected_spend': total_expected_spend,
            'expected_roas': portfolio_roas,
            'expected_profit': total_expected_sales - total_expected_spend
        }

class RealTimeOptimizationEngine:
    """Real-time optimization engine for dynamic bidding and budget allocation"""
    
    def __init__(self):
        self.performance_model = PerformancePredictionModel()
        self.budget_optimizer = BudgetOptimizationEngine()
        self.learning_rate = 0.1
        self.exploration_rate = 0.05
        
    def multi_armed_bandit_optimization(self, campaigns: List[str], 
                                      current_performance: Dict[str, float],
                                      budget_constraints: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Optimize using multi-armed bandit approach"""
        
        # Initialize or update campaign values
        if not hasattr(self, 'campaign_values'):
            self.campaign_values = {c: 1.0 for c in campaigns}
            self.campaign_counts = {c: 1 for c in campaigns}
        
        # Update values based on recent performance
        for campaign, performance in current_performance.items():
            if campaign in self.campaign_values:
                # Running average update
                self.campaign_counts[campaign] += 1
                self.campaign_values[campaign] += (
                    performance - self.campaign_values[campaign]
                ) / self.campaign_counts[campaign]
        
        # Calculate upper confidence bounds
        total_pulls = sum(self.campaign_counts.values())
        ucb_values = {}
        
        for campaign in campaigns:
            if campaign in self.campaign_values:
                confidence_bonus = np.sqrt(
                    2 * np.log(total_pulls) / self.campaign_counts[campaign]
                )
                ucb_values[campaign] = self.campaign_values[campaign] + confidence_bonus
            else:
                ucb_values[campaign] = float('inf')  # Explore unknown campaigns
        
        # Allocate budget based on UCB values and constraints
        allocations = {}
        remaining_budget = sum(max_budget for _, max_budget in budget_constraints.values())
        
        # Sort campaigns by UCB value
        sorted_campaigns = sorted(campaigns, key=lambda c: ucb_values[c], reverse=True)
        
        for campaign in sorted_campaigns:
            min_budget, max_budget = budget_constraints.get(campaign, (0, remaining_budget))
            
            # Allocate based on UCB value and constraints
            if remaining_budget > 0:
                allocation = min(max_budget, remaining_budget * 0.3)  # Conservative allocation
                allocations[campaign] = max(min_budget, allocation)
                remaining_budget -= allocations[campaign]
        
        return allocations
    
    def reinforcement_learning_optimization(self, state_features: np.ndarray,
                                          action_space: List[Dict[str, float]],
                                          reward_history: List[float]) -> Dict[str, float]:
        """Simple Q-learning based optimization"""
        
        # Simplified RL - in practice would use more sophisticated algorithms
        if not hasattr(self, 'q_table'):
            self.q_table = np.random.random((100, len(action_space)))
        
        # Discretize state
        state_index = min(int(np.sum(state_features) * 10) % 100, 99)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.exploration_rate:
            action_index = np.random.randint(len(action_space))
        else:
            action_index = np.argmax(self.q_table[state_index])
        
        # Update Q-value if we have reward history
        if reward_history:
            last_reward = reward_history[-1]
            self.q_table[state_index, action_index] += self.learning_rate * (
                last_reward - self.q_table[state_index, action_index]
            )
        
        return action_space[action_index]
