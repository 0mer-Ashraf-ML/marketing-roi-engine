"""
Comprehensive financial ROI calculator for advertising optimization.
File: financial/roi_calculator.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class FinancialROICalculator:
    """Advanced ROI calculator with multiple financial modeling approaches"""
    
    def __init__(self, 
                 discount_rate: float = 0.12,
                 tax_rate: float = 0.25,
                 working_capital_rate: float = 0.15):
        self.discount_rate = discount_rate
        self.tax_rate = tax_rate
        self.working_capital_rate = working_capital_rate
        self.daily_discount_rate = discount_rate / 365
        
    def calculate_basic_roas(self, revenue: float, ad_spend: float) -> float:
        """Calculate basic Return on Ad Spend (ROAS)"""
        return revenue / ad_spend if ad_spend > 0 else 0.0
    
    def calculate_true_roas(self, 
                          revenue: float, 
                          ad_spend: float,
                          platform_fees: float = 0.0,
                          fulfillment_fees: float = 0.0,
                          cost_of_goods: float = 0.0) -> float:
        """Calculate true ROAS after all costs"""
        
        net_revenue = revenue - platform_fees - fulfillment_fees - cost_of_goods
        return net_revenue / ad_spend if ad_spend > 0 else 0.0
    
    def calculate_contribution_margin_roas(self,
                                         revenue: float,
                                         ad_spend: float,
                                         variable_costs: Dict[str, float]) -> Dict[str, float]:
        """Calculate ROAS based on contribution margin analysis"""
        
        total_variable_costs = sum(variable_costs.values()) + ad_spend
        contribution_margin = revenue - total_variable_costs
        contribution_margin_rate = contribution_margin / revenue if revenue > 0 else 0.0
        
        return {
            'contribution_margin': contribution_margin,
            'contribution_margin_rate': contribution_margin_rate,
            'contribution_roas': contribution_margin / ad_spend if ad_spend > 0 else 0.0,
            'breakeven_roas': 1 / contribution_margin_rate if contribution_margin_rate > 0 else float('inf')
        }
    
    def calculate_npv_roas(self,
                          cash_flows: List[Tuple[datetime, float]],
                          ad_spend: float,
                          valuation_date: datetime) -> Dict[str, float]:
        """Calculate NPV-adjusted ROAS considering cash flow timing"""
        
        npv_revenue = 0.0
        
        for cash_flow_date, amount in cash_flows:
            days_to_payment = (cash_flow_date - valuation_date).days
            discount_factor = 1 / (1 + self.daily_discount_rate) ** days_to_payment
            npv_revenue += amount * discount_factor
        
        npv_roas = npv_revenue / ad_spend if ad_spend > 0 else 0.0
        
        return {
            'npv_revenue': npv_revenue,
            'npv_roas': npv_roas,
            'timing_impact': (npv_revenue - sum(amount for _, amount in cash_flows)) / sum(amount for _, amount in cash_flows) if cash_flows else 0.0
        }
    
    def calculate_customer_lifetime_value_roas(self,
                                             first_order_value: float,
                                             repeat_purchase_rate: float,
                                             average_order_frequency: float,
                                             gross_margin_rate: float,
                                             retention_periods: int,
                                             ad_spend: float) -> Dict[str, float]:
        """Calculate ROAS including customer lifetime value"""
        
        # Calculate CLV using cohort-based model
        clv = 0.0
        
        for period in range(1, retention_periods + 1):
            # Revenue in each period
            period_revenue = first_order_value * average_order_frequency * (repeat_purchase_rate ** (period - 1))
            
            # Apply margin
            period_contribution = period_revenue * gross_margin_rate
            
            # Discount to present value
            discount_factor = 1 / (1 + self.discount_rate) ** period
            
            clv += period_contribution * discount_factor
        
        clv_roas = clv / ad_spend if ad_spend > 0 else 0.0
        
        return {
            'customer_lifetime_value': clv,
            'clv_roas': clv_roas,
            'clv_multiple': clv / first_order_value if first_order_value > 0 else 0.0,
            'payback_period_months': ad_spend / (first_order_value * gross_margin_rate * average_order_frequency / 12) if first_order_value > 0 and gross_margin_rate > 0 else float('inf')
        }
    
    def calculate_incremental_roas(self,
                                 treatment_revenue: float,
                                 treatment_spend: float,
                                 control_revenue: float,
                                 control_spend: float,
                                 confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate incremental ROAS using test/control methodology"""
        
        # Calculate incremental metrics
        incremental_revenue = treatment_revenue - control_revenue
        incremental_spend = treatment_spend - control_spend
        
        incremental_roas = incremental_revenue / incremental_spend if incremental_spend > 0 else 0.0
        
        # Calculate statistical significance (simplified)
        from scipy import stats
        
        # Mock statistical test - in practice would use proper A/B test analysis
        treatment_roas = treatment_revenue / treatment_spend if treatment_spend > 0 else 0.0
        control_roas = control_revenue / control_spend if control_spend > 0 else 0.0
        
        # Simplified confidence interval
        pooled_variance = 0.1  # Would calculate from actual data
        standard_error = np.sqrt(pooled_variance * (1/100 + 1/100))  # Assuming 100 observations each
        
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * standard_error
        
        return {
            'incremental_revenue': incremental_revenue,
            'incremental_roas': incremental_roas,
            'treatment_roas': treatment_roas,
            'control_roas': control_roas,
            'confidence_interval_lower': incremental_roas - margin_of_error,
            'confidence_interval_upper': incremental_roas + margin_of_error,
            'statistical_significance': abs(incremental_roas) > margin_of_error
        }
    
    def calculate_portfolio_roas(self,
                               campaigns: List[Dict[str, float]],
                               correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate portfolio-level ROAS with risk adjustments"""
        
        if not campaigns:
            return {'portfolio_roas': 0.0, 'risk_adjusted_roas': 0.0}
        
        # Calculate weighted portfolio metrics
        total_spend = sum(c['spend'] for c in campaigns)
        total_revenue = sum(c['revenue'] for c in campaigns)
        
        portfolio_roas = total_revenue / total_spend if total_spend > 0 else 0.0
        
        # Calculate portfolio variance
        if correlation_matrix is not None and len(campaigns) == correlation_matrix.shape[0]:
            weights = np.array([c['spend'] / total_spend for c in campaigns])
            returns = np.array([c['revenue'] / c['spend'] if c['spend'] > 0 else 0.0 for c in campaigns])
            
            # Portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(correlation_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Risk-adjusted return (Sharpe ratio concept)
            risk_free_rate = self.discount_rate
            sharpe_ratio = (portfolio_roas - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0.0
            
            # Risk-adjusted ROAS
            risk_adjusted_roas = portfolio_roas * (1 + sharpe_ratio * 0.1)  # Adjust based on risk
        else:
            portfolio_std = 0.0
            sharpe_ratio = 0.0
            risk_adjusted_roas = portfolio_roas
        
        return {
            'portfolio_roas': portfolio_roas,
            'portfolio_std': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'risk_adjusted_roas': risk_adjusted_roas,
            'diversification_ratio': len(campaigns) / (1 + portfolio_std)
        }
    
    def calculate_working_capital_adjusted_roas(self,
                                              revenue: float,
                                              ad_spend: float,
                                              payment_terms_days: int,
                                              inventory_days: int,
                                              payable_days: int) -> Dict[str, float]:
        """Calculate ROAS adjusted for working capital requirements"""
        
        # Calculate working capital components
        accounts_receivable = revenue * (payment_terms_days / 365)
        inventory_investment = ad_spend * 0.3 * (inventory_days / 365)  # Assume 30% of ad spend drives inventory
        accounts_payable = ad_spend * (payable_days / 365)
        
        net_working_capital = accounts_receivable + inventory_investment - accounts_payable
        
        # Working capital carrying cost
        wc_carrying_cost = net_working_capital * self.working_capital_rate
        
        # Adjusted revenue
        adjusted_revenue = revenue - wc_carrying_cost
        wc_adjusted_roas = adjusted_revenue / ad_spend if ad_spend > 0 else 0.0
        
        return {
            'accounts_receivable': accounts_receivable,
            'inventory_investment': inventory_investment,
            'accounts_payable': accounts_payable,
            'net_working_capital': net_working_capital,
            'wc_carrying_cost': wc_carrying_cost,
            'wc_adjusted_roas': wc_adjusted_roas,
            'wc_impact_percentage': (wc_carrying_cost / revenue * 100) if revenue > 0 else 0.0
        }
    
    def calculate_tax_adjusted_roas(self,
                                  revenue: float,
                                  ad_spend: float,
                                  other_deductible_costs: float = 0.0) -> Dict[str, float]:
        """Calculate ROAS with tax implications"""
        
        # Taxable income
        taxable_income = revenue - ad_spend - other_deductible_costs
        
        # Tax liability
        tax_liability = max(0, taxable_income * self.tax_rate)
        
        # After-tax profit
        after_tax_profit = taxable_income - tax_liability
        
        # Tax-adjusted ROAS
        tax_adjusted_roas = (revenue - tax_liability) / ad_spend if ad_spend > 0 else 0.0
        
        # Tax shield value from advertising
        tax_shield = ad_spend * self.tax_rate
        
        return {
            'taxable_income': taxable_income,
            'tax_liability': tax_liability,
            'after_tax_profit': after_tax_profit,
            'tax_adjusted_roas': tax_adjusted_roas,
            'tax_shield_value': tax_shield,
            'effective_ad_cost': ad_spend - tax_shield
        }
    
    def calculate_comprehensive_roi_metrics(self,
                                          campaign_data: Dict[str, Union[float, List, Dict]],
                                          financial_params: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate comprehensive ROI metrics combining all methodologies"""
        
        if financial_params is None:
            financial_params = {}
        
        results = {}
        
        # Basic metrics
        revenue = campaign_data.get('revenue', 0.0)
        ad_spend = campaign_data.get('ad_spend', 0.0)
        
        results['basic_roas'] = self.calculate_basic_roas(revenue, ad_spend)
        
        # True ROAS with costs
        platform_fees = campaign_data.get('platform_fees', revenue * 0.12)
        fulfillment_fees = campaign_data.get('fulfillment_fees', revenue * 0.08)
        cost_of_goods = campaign_data.get('cost_of_goods', revenue * 0.4)
        
        results['true_roas'] = self.calculate_true_roas(
            revenue, ad_spend, platform_fees, fulfillment_fees, cost_of_goods
        )
        
        # Contribution margin analysis
        variable_costs = {
            'platform_fees': platform_fees,
            'fulfillment_fees': fulfillment_fees,
            'cost_of_goods': cost_of_goods
        }
        
        contrib_metrics = self.calculate_contribution_margin_roas(revenue, ad_spend, variable_costs)
        results.update(contrib_metrics)
        
        # NPV analysis if cash flow timing provided
        if 'cash_flows' in campaign_data:
            npv_metrics = self.calculate_npv_roas(
                campaign_data['cash_flows'],
                ad_spend,
                campaign_data.get('valuation_date', datetime.now())
            )
            results.update(npv_metrics)
        
        # CLV analysis
        clv_metrics = self.calculate_customer_lifetime_value_roas(
            first_order_value=campaign_data.get('avg_order_value', revenue),
            repeat_purchase_rate=financial_params.get('repeat_purchase_rate', 0.3),
            average_order_frequency=financial_params.get('avg_order_frequency', 4.0),
            gross_margin_rate=financial_params.get('gross_margin_rate', 0.35),
            retention_periods=financial_params.get('retention_periods', 3),
            ad_spend=ad_spend
        )
        results.update(clv_metrics)
        
        # Working capital analysis
        wc_metrics = self.calculate_working_capital_adjusted_roas(
            revenue, ad_spend,
            payment_terms_days=financial_params.get('payment_terms_days', 30),
            inventory_days=financial_params.get('inventory_days', 45),
            payable_days=financial_params.get('payable_days', 30)
        )
        results.update(wc_metrics)
        
        # Tax analysis
        tax_metrics = self.calculate_tax_adjusted_roas(
            revenue, ad_spend,
            other_deductible_costs=campaign_data.get('other_costs', 0.0)
        )
        results.update(tax_metrics)
        
        # Calculate composite ROI score
        weights = {
            'true_roas': 0.3,
            'contribution_roas': 0.25,
            'clv_roas': 0.2,
            'wc_adjusted_roas': 0.15,
            'tax_adjusted_roas': 0.1
        }
        
        composite_score = sum(
            results.get(metric, 0) * weight 
            for metric, weight in weights.items()
        )
        
        results['composite_roi_score'] = composite_score
        
        # Risk-adjusted metrics
        volatility = campaign_data.get('roas_volatility', 0.1)
        results['risk_adjusted_roi'] = composite_score / (1 + volatility)
        
        return results
    
    def benchmark_roi_performance(self,
                                campaign_metrics: Dict[str, float],
                                industry_benchmarks: Dict[str, float] = None) -> Dict[str, Any]:
        """Benchmark ROI performance against industry standards"""
        
        if industry_benchmarks is None:
            # Default e-commerce benchmarks
            industry_benchmarks = {
                'basic_roas': 4.0,
                'true_roas': 2.5,
                'contribution_roas': 1.8,
                'clv_roas': 3.0,
                'composite_roi_score': 2.5
            }
        
        benchmark_results = {}
        
        for metric, campaign_value in campaign_metrics.items():
            if metric in industry_benchmarks:
                benchmark_value = industry_benchmarks[metric]
                
                benchmark_results[f'{metric}_vs_benchmark'] = {
                    'campaign_value': campaign_value,
                    'benchmark_value': benchmark_value,
                    'performance_ratio': campaign_value / benchmark_value if benchmark_value > 0 else 0,
                    'performance_difference': campaign_value - benchmark_value,
                    'percentile_estimate': min(100, max(0, campaign_value / benchmark_value * 50))
                }
        
        # Overall performance rating
        avg_performance_ratio = np.mean([
            result['performance_ratio'] 
            for result in benchmark_results.values()
        ])
        
        if avg_performance_ratio >= 1.2:
            performance_rating = 'Excellent'
        elif avg_performance_ratio >= 1.0:
            performance_rating = 'Good'
        elif avg_performance_ratio >= 0.8:
            performance_rating = 'Average'
        else:
            performance_rating = 'Below Average'
        
        return {
            'detailed_benchmarks': benchmark_results,
            'overall_performance_ratio': avg_performance_ratio,
            'performance_rating': performance_rating
        }
    
    def generate_roi_optimization_recommendations(self,
                                                current_metrics: Dict[str, float],
                                                target_metrics: Dict[str, float] = None) -> List[Dict[str, str]]:
        """Generate actionable recommendations for ROI optimization"""
        
        recommendations = []
        
        if target_metrics is None:
            target_metrics = {
                'true_roas': 3.0,
                'contribution_roas': 2.0,
                'clv_roas': 4.0
            }
        
        # Analyze gaps and generate recommendations
        for metric, target in target_metrics.items():
            current = current_metrics.get(metric, 0)
            gap = target - current
            
            if gap > 0:
                if metric == 'true_roas':
                    if current < 1.5:
                        recommendations.append({
                            'priority': 'High',
                            'category': 'Cost Optimization',
                            'recommendation': 'Focus on reducing platform fees and fulfillment costs. Consider FBA optimization or alternative fulfillment methods.',
                            'expected_impact': f'Could improve true ROAS by {gap:.2f}x'
                        })
                    else:
                        recommendations.append({
                            'priority': 'Medium',
                            'category': 'Revenue Optimization',
                            'recommendation': 'Optimize product pricing and focus on higher-margin products in ad campaigns.',
                            'expected_impact': f'Could improve true ROAS by {gap:.2f}x'
                        })
                
                elif metric == 'contribution_roas':
                    recommendations.append({
                        'priority': 'High',
                        'category': 'Margin Improvement',
                        'recommendation': 'Analyze variable cost structure. Focus advertising on products with higher contribution margins.',
                        'expected_impact': f'Could improve contribution ROAS by {gap:.2f}x'
                    })
                
                elif metric == 'clv_roas':
                    recommendations.append({
                        'priority': 'Medium',
                        'category': 'Customer Retention',
                        'recommendation': 'Implement customer retention programs and focus on acquiring high-LTV customers.',
                        'expected_impact': f'Could improve CLV ROAS by {gap:.2f}x'
                    })
        
        # General optimization recommendations
        if current_metrics.get('composite_roi_score', 0) < 2.0:
            recommendations.append({
                'priority': 'High',
                'category': 'Strategy',
                'recommendation': 'Consider reallocating budget to higher-performing campaigns and platforms.',
                'expected_impact': 'Could improve overall ROI by 20-40%'
            })
        
        return sorted(recommendations, key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}[x['priority']], reverse=True)
