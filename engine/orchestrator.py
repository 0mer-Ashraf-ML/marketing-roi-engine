"""
Main orchestration engine that coordinates all components of the advertising ROI system.
File: engine/orchestrator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import all our custom modules
from data.models import DataWarehouse, DataProcessor
from features.attribution import AttributionFeatureEngine
from features.financial import FinancialFeatureEngine
from models.ml_engine import AttributionMLModel, PerformancePredictionModel, BudgetOptimizationEngine, RealTimeOptimizationEngine
from financial.roi_calculator import FinancialROICalculator

class AdvertisingROIOrchestrator:
    """Main orchestration engine for the advertising ROI optimization system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.setup_logging()
        
        # Initialize core components
        self.data_warehouse = DataWarehouse()
        self.attribution_engine = AttributionFeatureEngine(
            decay_rate=self.config['attribution']['decay_rate'],
            window_days=self.config['attribution']['window_days']
        )
        self.financial_engine = FinancialFeatureEngine(
            annual_discount_rate=self.config['financial']['discount_rate']
        )
        self.roi_calculator = FinancialROICalculator(
            discount_rate=self.config['financial']['discount_rate'],
            tax_rate=self.config['financial']['tax_rate']
        )
        
        # ML Models
        self.attribution_model = AttributionMLModel()
        self.performance_model = PerformancePredictionModel()
        self.budget_optimizer = BudgetOptimizationEngine(
            risk_aversion=self.config['optimization']['risk_aversion']
        )
        self.realtime_optimizer = RealTimeOptimizationEngine()
        
        # State tracking
        self.last_update = None
        self.model_performance = {}
        self.optimization_history = []
        
        self.logger.info("Advertising ROI Orchestrator initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the system"""
        return {
            'attribution': {
                'decay_rate': 0.1,
                'window_days': 30,
                'models': ['time_decay', 'position', 'shapley', 'markov']
            },
            'financial': {
                'discount_rate': 0.12,
                'tax_rate': 0.25,
                'working_capital_rate': 0.15
            },
            'optimization': {
                'risk_aversion': 0.5,
                'rebalance_frequency': 'daily',
                'min_allocation': 0.05,
                'max_allocation': 0.40
            },
            'models': {
                'retrain_frequency': 'weekly',
                'performance_threshold': 0.7,
                'ensemble_weights': {
                    'xgboost': 0.4,
                    'gradient_boost': 0.3,
                    'random_forest': 0.3
                }
            },
            'alerts': {
                'performance_decline_threshold': 0.15,
                'budget_overrun_threshold': 0.10,
                'roas_decline_threshold': 0.20
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_sources: Dict[str, pd.DataFrame]) -> None:
        """Load all data sources into the system"""
        
        self.logger.info("Loading data into system...")
        
        # Load campaign data
        if 'campaigns' in data_sources:
            self.data_warehouse.load_campaign_data(data_sources['campaigns'])
            self.logger.info(f"Loaded {len(data_sources['campaigns'])} campaign records")
        
        # Load keyword data
        if 'keywords' in data_sources:
            self.data_warehouse.load_keyword_data(data_sources['keywords'])
            self.logger.info(f"Loaded {len(data_sources['keywords'])} keyword records")
        
        # Load product data
        if 'products' in data_sources:
            self.data_warehouse.load_product_data(data_sources['products'])
            self.logger.info(f"Loaded {len(data_sources['products'])} product records")
        
        # Load financial data
        if 'financial' in data_sources:
            self.data_warehouse.load_financial_data(data_sources['financial'])
            self.logger.info(f"Loaded {len(data_sources['financial'])} financial records")
        
        # Load attribution data
        if 'attribution' in data_sources:
            self.data_warehouse.load_attribution_data(data_sources['attribution'])
            self.logger.info(f"Loaded {len(data_sources['attribution'])} attribution records")
        
        self.last_update = datetime.now()
        self.logger.info("Data loading completed successfully")
    
    def run_attribution_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run comprehensive attribution analysis"""
        
        self.logger.info(f"Running attribution analysis from {start_date} to {end_date}")
        
        try:
            # Get attribution data
            attribution_data = self.data_warehouse.attribution[
                (self.data_warehouse.attribution['timestamp'] >= start_date) &
                (self.data_warehouse.attribution['timestamp'] <= end_date)
            ].copy()
            
            if attribution_data.empty:
                self.logger.warning("No attribution data found for the specified period")
                return {'status': 'no_data'}
            
            # Get campaign data for feature enrichment
            campaign_data = self.data_warehouse.get_campaign_performance(start_date, end_date)
            
            # Create attribution features
            attribution_features = self.attribution_engine.create_attribution_features(
                attribution_data, campaign_data
            )
            
            # Train attribution model if not already trained or needs retraining
            if not self.attribution_model.is_trained or self._should_retrain_models():
                self.logger.info("Training attribution model...")
                model_performance = self.attribution_model.train(attribution_features)
                self.model_performance['attribution'] = model_performance
            
            # Generate predictions
            predicted_attribution = self.attribution_model.predict(attribution_features)
            attribution_features['predicted_attribution'] = predicted_attribution
            
            # Create ensemble attribution
            final_attribution = self.attribution_engine.create_ensemble_attribution(
                attribution_features
            )
            
            # Calculate attribution insights
            insights = self._calculate_attribution_insights(final_attribution)
            
            return {
                'status': 'success',
                'attribution_data': final_attribution,
                'insights': insights,
                'model_performance': self.model_performance.get('attribution', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error in attribution analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def run_financial_analysis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run comprehensive financial analysis"""
        
        self.logger.info(f"Running financial analysis from {start_date} to {end_date}")
        
        try:
            # Get unified dataset
            unified_data = self.data_warehouse.get_unified_dataset(start_date, end_date)
            
            if unified_data.empty:
                self.logger.warning("No financial data found for the specified period")
                return {'status': 'no_data'}
            
            # Create financial features
            financial_features = self.financial_engine.create_financial_feature_matrix(
                campaign_df=unified_data,
                product_df=self.data_warehouse.products,
                financial_df=self.data_warehouse.financial,
                attribution_df=self.data_warehouse.attribution,
                competitive_df=pd.DataFrame()  # Empty for now
            )
            
            # Calculate comprehensive ROI metrics
            roi_results = []
            
            for _, row in financial_features.iterrows():
                campaign_data = {
                    'revenue': row['sales'],
                    'ad_spend': row['spend'],
                    'platform_fees': row.get('platform_fees', row['sales'] * 0.12),
                    'cost_of_goods': row.get('cost_of_goods_sold', row['sales'] * 0.4),
                    'avg_order_value': row['sales'] / max(row['orders'], 1)
                }
                
                roi_metrics = self.roi_calculator.calculate_comprehensive_roi_metrics(
                    campaign_data, self.config['financial']
                )
                roi_metrics['campaign_id'] = row['campaign_id']
                roi_metrics['date'] = row['date']
                roi_results.append(roi_metrics)
            
            roi_df = pd.DataFrame(roi_results)
            
            # Generate financial insights
            financial_insights = self._calculate_financial_insights(roi_df, financial_features)
            
            return {
                'status': 'success',
                'financial_features': financial_features,
                'roi_metrics': roi_df,
                'insights': financial_insights
            }
            
        except Exception as e:
            self.logger.error(f"Error in financial analysis: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def run_performance_prediction(self, start_date: datetime, end_date: datetime, 
                                 forecast_days: int = 30) -> Dict[str, Any]:
        """Run performance prediction and forecasting"""
        
        self.logger.info(f"Running performance prediction for {forecast_days} days")
        
        try:
            # Get historical data for training
            unified_data = self.data_warehouse.get_unified_dataset(start_date, end_date)
            
            if len(unified_data) < 50:  # Need minimum data for training
                self.logger.warning("Insufficient data for performance prediction")
                return {'status': 'insufficient_data'}
            
            # Train performance models if needed
            if not hasattr(self.performance_model, 'performance_models') or self._should_retrain_models():
                self.logger.info("Training performance prediction models...")
                model_results = self.performance_model.train_performance_models(unified_data)
                self.model_performance['prediction'] = model_results
            
            # Create forecast scenarios
            forecast_results = {}
            
            # Get latest data point for each campaign
            latest_data = unified_data.groupby('campaign_id').last().reset_index()
            
            for target in ['sales', 'roas', 'acos']:
                try:
                    # Generate forecasts for each campaign
                    forecasts = []
                    
                    for _, campaign_row in latest_data.iterrows():
                        # Create future dates
                        future_dates = [
                            campaign_row['date'] + timedelta(days=i) 
                            for i in range(1, forecast_days + 1)
                        ]
                        
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'campaign_id': [campaign_row['campaign_id']] * forecast_days,
                            'date': future_dates,
                            'spend': [campaign_row['spend']] * forecast_days,  # Assume constant spend
                            'platform': [campaign_row['platform']] * forecast_days,
                            'campaign_type': [campaign_row.get('campaign_type', 'sponsored_products')] * forecast_days
                        })
                        
                        # Add other required features with forward fill
                        for col in unified_data.columns:
                            if col not in forecast_df.columns:
                                forecast_df[col] = campaign_row.get(col, 0)
                        
                        # Generate predictions
                        predictions = self.performance_model.predict_performance(
                            forecast_df, target=target
                        )
                        
                        forecast_df[f'predicted_{target}'] = predictions
                        forecasts.append(forecast_df)
                    
                    if forecasts:
                        forecast_results[target] = pd.concat(forecasts, ignore_index=True)
                    
                except Exception as e:
                    self.logger.warning(f"Could not generate forecast for {target}: {str(e)}")
            
            # Calculate forecast insights
            prediction_insights = self._calculate_prediction_insights(forecast_results)
            
            return {
                'status': 'success',
                'forecasts': forecast_results,
                'insights': prediction_insights,
                'model_performance': self.model_performance.get('prediction', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error in performance prediction: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def run_budget_optimization(self, total_budget: float, 
                              optimization_period: int = 30,
                              constraints: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Run budget optimization analysis"""
        
        self.logger.info(f"Running budget optimization for ${total_budget:,.2f}")
        
        try:
            # Get recent performance data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            unified_data = self.data_warehouse.get_unified_dataset(start_date, end_date)
            
            if unified_data.empty:
                return {'status': 'no_data'}
            
            # Calculate expected returns
            expected_returns = self.budget_optimizer.calculate_expected_returns(unified_data)
            
            # Calculate covariance matrix
            covariance_matrix = self.budget_optimizer.calculate_covariance_matrix(unified_data)
            
            # Run optimization
            optimization_result = self.budget_optimizer.optimize_portfolio(
                total_budget=total_budget,
                constraints=constraints
            )
            
            if optimization_result['status'] == 'optimal':
                # Calculate expected performance
                expected_performance = self._calculate_expected_performance(
                    optimization_result['allocations'], expected_returns
                )
                
                # Generate allocation insights
                allocation_insights = self._calculate_allocation_insights(
                    optimization_result, unified_data
                )
                
                # Store optimization history
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'total_budget': total_budget,
                    'allocations': optimization_result['allocations'],
                    'expected_return': optimization_result['expected_return'],
                    'expected_risk': optimization_result['expected_risk']
                })
                
                return {
                    'status': 'success',
                    'optimization': optimization_result,
                    'expected_performance': expected_performance,
                    'insights': allocation_insights,
                    'current_allocations': self._get_current_allocations(unified_data)
                }
            else:
                return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error in budget optimization: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis results"""
        
        recommendations = []
        
        try:
            # Attribution recommendations
            if 'attribution' in analysis_results and analysis_results['attribution']['status'] == 'success':
                attr_insights = analysis_results['attribution']['insights']
                
                if attr_insights.get('underperforming_touchpoints'):
                    recommendations.append({
                        'category': 'Attribution Optimization',
                        'priority': 'High',
                        'title': 'Optimize Underperforming Touchpoints',
                        'description': f"Found {len(attr_insights['underperforming_touchpoints'])} touchpoints with low attribution values",
                        'action_items': [
                            'Review keyword targeting for low-attribution touchpoints',
                            'Consider budget reallocation from underperforming touchpoints',
                            'Test different ad creative for improved engagement'
                        ],
                        'expected_impact': 'Could improve overall attribution efficiency by 15-25%'
                    })
            
            # Financial recommendations
            if 'financial' in analysis_results and analysis_results['financial']['status'] == 'success':
                roi_insights = analysis_results['financial']['insights']
                
                if roi_insights.get('low_roi_campaigns'):
                    recommendations.append({
                        'category': 'Financial Optimization',
                        'priority': 'High',
                        'title': 'Address Low ROI Campaigns',
                        'description': f"Identified {len(roi_insights['low_roi_campaigns'])} campaigns with below-target ROI",
                        'action_items': [
                            'Pause or reduce budget for campaigns with ROI < 1.5x',
                            'Analyze cost structure for improvement opportunities',
                            'Focus on higher-margin products'
                        ],
                        'expected_impact': 'Could improve overall portfolio ROI by 20-30%'
                    })
            
            # Budget optimization recommendations
            if 'optimization' in analysis_results and analysis_results['optimization']['status'] == 'success':
                opt_insights = analysis_results['optimization']['insights']
                
                if opt_insights.get('reallocation_opportunities'):
                    recommendations.append({
                        'category': 'Budget Allocation',
                        'priority': 'Medium',
                        'title': 'Implement Optimal Budget Allocation',
                        'description': 'Mathematical optimization suggests significant budget reallocation opportunities',
                        'action_items': [
                            'Increase budget for high-performing campaigns',
                            'Reduce allocation to underperforming campaigns',
                            'Monitor performance after reallocation'
                        ],
                        'expected_impact': f"Could improve portfolio return by {opt_insights.get('improvement_potential', 0):.1%}"
                    })
            
            # Performance prediction recommendations
            if 'prediction' in analysis_results and analysis_results['prediction']['status'] == 'success':
                pred_insights = analysis_results['prediction']['insights']
                
                if pred_insights.get('declining_performance_campaigns'):
                    recommendations.append({
                        'category': 'Performance Management',
                        'priority': 'Medium',
                        'title': 'Address Predicted Performance Decline',
                        'description': 'ML models predict performance decline for several campaigns',
                        'action_items': [
                            'Preemptively adjust bidding strategies',
                            'Refresh ad creative and targeting',
                            'Consider seasonal adjustments'
                        ],
                        'expected_impact': 'Could prevent 10-20% performance decline'
                    })
            
            # General recommendations
            recommendations.extend([
                {
                    'category': 'System Health',
                    'priority': 'Low',
                    'title': 'Regular Model Retraining',
                    'description': 'Ensure ML models stay current with latest data',
                    'action_items': [
                        'Schedule weekly model retraining',
                        'Monitor model performance metrics',
                        'Update feature engineering as needed'
                    ],
                    'expected_impact': 'Maintains prediction accuracy over time'
                },
                {
                    'category': 'Data Quality',
                    'priority': 'Medium',
                    'title': 'Enhance Attribution Tracking',
                    'description': 'Improve data collection for better attribution accuracy',
                    'action_items': [
                        'Implement cross-device tracking',
                        'Enhance conversion tracking setup',
                        'Regular data quality audits'
                    ],
                    'expected_impact': 'Could improve attribution accuracy by 10-15%'
                }
            ])
            
            # Sort by priority
            priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
            recommendations.sort(key=lambda x: priority_order[x['priority']], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def run_full_analysis(self, start_date: datetime, end_date: datetime,
                         total_budget: float = None) -> Dict[str, Any]:
        """Run complete end-to-end analysis"""
        
        self.logger.info("Starting full analysis pipeline...")
        
        results = {
            'analysis_date': datetime.now(),
            'period': {'start': start_date, 'end': end_date}
        }
        
        # 1. Attribution Analysis
        results['attribution'] = self.run_attribution_analysis(start_date, end_date)
        
        # 2. Financial Analysis
        results['financial'] = self.run_financial_analysis(start_date, end_date)
        
        # 3. Performance Prediction
        results['prediction'] = self.run_performance_prediction(start_date, end_date)
        
        # 4. Budget Optimization (if budget specified)
        if total_budget:
            results['optimization'] = self.run_budget_optimization(total_budget)
        
        # 5. Generate Recommendations
        results['recommendations'] = self.generate_recommendations(results)
        
        # 6. Create Executive Summary
        results['executive_summary'] = self._create_executive_summary(results)
        
        self.logger.info("Full analysis pipeline completed successfully")
        
        return results
    
    def _should_retrain_models(self) -> bool:
        """Determine if models should be retrained"""
        if self.last_update is None:
            return True
        
        days_since_update = (datetime.now() - self.last_update).days
        
        # Retrain weekly by default
        retrain_frequency = self.config['models']['retrain_frequency']
        
        if retrain_frequency == 'daily':
            return days_since_update >= 1
        elif retrain_frequency == 'weekly':
            return days_since_update >= 7
        elif retrain_frequency == 'monthly':
            return days_since_update >= 30
        
        return False
    
    def _calculate_attribution_insights(self, attribution_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate insights from attribution analysis"""
        
        insights = {}
        
        # Top contributing touchpoints
        top_touchpoints = attribution_df.groupby('touchpoint_type')['attributed_revenue'].sum().sort_values(ascending=False)
        insights['top_touchpoints'] = top_touchpoints.to_dict()
        
        # Platform performance
        platform_performance = attribution_df.groupby('platform')['attributed_revenue'].sum()
        insights['platform_performance'] = platform_performance.to_dict()
        
        # Underperforming touchpoints
        underperforming = attribution_df[attribution_df['attributed_revenue'] < attribution_df['attributed_revenue'].quantile(0.25)]
        insights['underperforming_touchpoints'] = underperforming['touchpoint_id'].tolist()
        
        # Average customer journey length
        journey_stats = attribution_df.groupby('customer_id')['touchpoint_position'].max()
        insights['avg_journey_length'] = journey_stats.mean()
        insights['journey_length_distribution'] = journey_stats.value_counts().to_dict()
        
        return insights
    
    def _calculate_financial_insights(self, roi_df: pd.DataFrame, 
                                    financial_features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate insights from financial analysis"""
        
        insights = {}
        
        # ROI performance by campaign
        campaign_roi = roi_df.groupby('campaign_id')['composite_roi_score'].mean().sort_values(ascending=False)
        insights['top_roi_campaigns'] = campaign_roi.head(10).to_dict()
        insights['low_roi_campaigns'] = campaign_roi[campaign_roi < 1.5].index.tolist()
        
        # Platform profitability
        platform_performance = financial_features.groupby('platform').agg({
            'true_roas': 'mean',
            'contribution_margin_rate': 'mean',
            'financial_health_score': 'mean'
        }).round(3)
        insights['platform_profitability'] = platform_performance.to_dict()
        
        # Cost efficiency analysis
        insights['avg_contribution_margin'] = financial_features['contribution_margin_rate'].mean()
        insights['working_capital_impact'] = financial_features['wc_carrying_cost'].sum()
        
        # Risk metrics
        insights['portfolio_risk'] = {
            'avg_volatility': financial_features['roas_rolling_std'].mean(),
            'max_drawdown': financial_features['max_drawdown'].min(),
            'sharpe_ratio': financial_features['sharpe_ratio'].mean()
        }
        
        return insights
    
    def _calculate_prediction_insights(self, forecast_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate insights from prediction analysis"""
        
        insights = {}
        
        for target, forecast_df in forecast_results.items():
            target_insights = {}
            
            # Predicted performance trends
            campaign_trends = forecast_df.groupby('campaign_id')[f'predicted_{target}'].agg(['mean', 'std', 'min', 'max'])
            
            # Identify declining campaigns
            declining_campaigns = []
            for campaign_id in forecast_df['campaign_id'].unique():
                campaign_data = forecast_df[forecast_df['campaign_id'] == campaign_id].sort_values('date')
                if len(campaign_data) > 5:
                    trend = np.polyfit(range(len(campaign_data)), campaign_data[f'predicted_{target}'], 1)[0]
                    if trend < -0.01:  # Declining trend
                        declining_campaigns.append(campaign_id)
            
            target_insights['declining_performance_campaigns'] = declining_campaigns
            target_insights['performance_distribution'] = campaign_trends.to_dict()
            
            insights[target] = target_insights
        
        return insights
    
    def _calculate_allocation_insights(self, optimization_result: Dict[str, Any], 
                                    current_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate insights from budget allocation optimization"""
        
        insights = {}
        
        # Current vs optimal allocation comparison
        current_allocations = self._get_current_allocations(current_data)
        optimal_allocations = optimization_result['allocations']
        
        reallocation_opportunities = {}
        total_reallocation = 0
        
        for campaign_id in optimal_allocations.keys():
            current = current_allocations.get(campaign_id, 0)
            optimal = optimal_allocations[campaign_id]
            difference = optimal - current
            
            if abs(difference) > current * 0.1:  # More than 10% difference
                reallocation_opportunities[campaign_id] = {
                    'current': current,
                    'optimal': optimal,
                    'change': difference,
                    'change_percentage': (difference / current * 100) if current > 0 else 0
                }
                total_reallocation += abs(difference)
        
        insights['reallocation_opportunities'] = reallocation_opportunities
        insights['total_reallocation_amount'] = total_reallocation
        insights['improvement_potential'] = optimization_result.get('expected_return', 0) - 1.0
        
        return insights
    
    def _get_current_allocations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get current budget allocations by campaign"""
        
        recent_data = data[data['date'] >= data['date'].max() - timedelta(days=7)]
        current_allocations = recent_data.groupby('campaign_id')['spend'].sum().to_dict()
        
        return current_allocations
    
    def _calculate_expected_performance(self, allocations: Dict[str, float], 
                                      expected_returns: Dict[str, float]) -> Dict[str, float]:
        """Calculate expected performance given allocations"""
        
        total_spend = sum(allocations.values())
        expected_revenue = sum(
            allocation * expected_returns.get(campaign_id, 1.0)
            for campaign_id, allocation in allocations.items()
        )
        
        return {
            'total_spend': total_spend,
            'expected_revenue': expected_revenue,
            'expected_roas': expected_revenue / total_spend if total_spend > 0 else 0,
            'expected_profit': expected_revenue - total_spend
        }
    
    def _create_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of all analysis results"""
        
        summary = {
            'overall_status': 'healthy',
            'key_metrics': {},
            'critical_issues': [],
            'top_opportunities': [],
            'next_actions': []
        }
        
        # Aggregate key metrics
        if results.get('financial', {}).get('status') == 'success':
            roi_data = results['financial']['roi_metrics']
            summary['key_metrics']['avg_true_roas'] = roi_data['true_roas'].mean()
            summary['key_metrics']['avg_contribution_roas'] = roi_data['contribution_roas'].mean()
            summary['key_metrics']['total_revenue'] = roi_data['sales'].sum() if 'sales' in roi_data.columns else 0
        
        if results.get('optimization', {}).get('status') == 'success':
            opt_data = results['optimization']['optimization']
            summary['key_metrics']['portfolio_expected_return'] = opt_data.get('expected_return', 0)
            summary['key_metrics']['portfolio_risk'] = opt_data.get('expected_risk', 0)
        
        # Identify critical issues
        for rec in results.get('recommendations', []):
            if rec['priority'] == 'High':
                summary['critical_issues'].append(rec['title'])
            elif rec['priority'] == 'Medium':
                summary['top_opportunities'].append(rec['title'])
        
        # Next actions from high-priority recommendations
        high_priority_recs = [r for r in results.get('recommendations', []) if r['priority'] == 'High']
        for rec in high_priority_recs[:3]:  # Top 3 actions
            if rec.get('action_items'):
                summary['next_actions'].extend(rec['action_items'][:2])  # Top 2 actions per recommendation
        
        # Overall health assessment
        if len(summary['critical_issues']) > 3:
            summary['overall_status'] = 'needs_attention'
        elif len(summary['critical_issues']) > 1:
            summary['overall_status'] = 'moderate_risk'
        
        return summary