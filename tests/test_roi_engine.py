"""
Comprehensive unit tests for the advertising ROI optimization engine.
File: tests/test_roi_engine.py
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules to test
from data.sample_data import SampleDataGenerator
from data.models import DataWarehouse, DataProcessor
from features.attribution import AttributionFeatureEngine
from features.financial import FinancialFeatureEngine
from models.ml_engine import AttributionMLModel, PerformancePredictionModel, BudgetOptimizationEngine
from financial.roi_calculator import FinancialROICalculator
from engine.orchestrator import AdvertisingROIOrchestrator
from reporting.analytics import ReportingEngine

class TestSampleDataGenerator(unittest.TestCase):
    """Test sample data generation functionality"""
    
    def setUp(self):
        self.generator = SampleDataGenerator(seed=42)
    
    def test_generate_campaign_data(self):
        """Test campaign data generation"""
        campaigns = self.generator.generate_campaign_data(days=30, campaigns_count=10)
        
        self.assertIsInstance(campaigns, pd.DataFrame)
        self.assertGreater(len(campaigns), 0)
        self.assertTrue(all(col in campaigns.columns for col in 
                          ['campaign_id', 'platform', 'date', 'spend', 'sales']))
        
        # Test data quality
        self.assertTrue((campaigns['spend'] >= 0).all())
        self.assertTrue((campaigns['sales'] >= 0).all())
        self.assertTrue((campaigns['clicks'] >= 0).all())
    
    def test_generate_keyword_data(self):
        """Test keyword data generation"""
        campaigns = self.generator.generate_campaign_data(days=7, campaigns_count=5)
        keywords = self.generator.generate_keyword_data(campaigns)
        
        self.assertIsInstance(keywords, pd.DataFrame)
        self.assertGreater(len(keywords), 0)
        self.assertTrue(all(col in keywords.columns for col in 
                          ['keyword_id', 'campaign_id', 'keyword', 'spend', 'sales']))
    
    def test_generate_financial_data(self):
        """Test financial data generation"""
        products = self.generator.generate_product_data(days=7)
        financial = self.generator.generate_financial_data(products)
        
        self.assertIsInstance(financial, pd.DataFrame)
        self.assertGreater(len(financial), 0)
        self.assertTrue(all(col in financial.columns for col in 
                          ['asin', 'date', 'cost_of_goods_sold', 'platform_fees']))

class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality"""
    
    def setUp(self):
        self.processor = DataProcessor()
        self.sample_data = {
            'campaign_id': ['CAMP-001', 'CAMP-002', 'CAMP-001'],
            'platform': ['amazon', 'walmart', 'amazon'],
            'date': ['2023-01-01', '2023-01-01', '2023-01-02'],
            'spend': [100.0, 200.0, 150.0],
            'sales': [400.0, 600.0, 500.0],
            'impressions': [1000, 2000, 1500],
            'clicks': [50, 100, 75],
            'orders': [4, 6, 5]
        }
        self.df = pd.DataFrame(self.sample_data)
    
    def test_validate_data(self):
        """Test data validation"""
        validated_df = self.processor.validate_data(self.df, 'campaign')
        
        self.assertIsInstance(validated_df, pd.DataFrame)
        self.assertEqual(len(validated_df), 3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(validated_df['date']))
    
    def test_aggregate_campaign_data(self):
        """Test campaign data aggregation"""
        validated_df = self.processor.validate_data(self.df, 'campaign')
        aggregated_df = self.processor.aggregate_campaign_data(validated_df)
        
        self.assertIsInstance(aggregated_df, pd.DataFrame)
        self.assertTrue('cpc' in aggregated_df.columns)
        self.assertTrue('ctr' in aggregated_df.columns)
        self.assertTrue('roas' in aggregated_df.columns)
    
    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation"""
        validated_df = self.processor.validate_data(self.df, 'campaign')
        aggregated_df = self.processor.aggregate_campaign_data(validated_df)
        rolling_df = self.processor.calculate_rolling_metrics(aggregated_df, [7])
        
        self.assertTrue('spend_rolling_7d' in rolling_df.columns)
        self.assertTrue('roas_rolling_7d' in rolling_df.columns)

class TestDataWarehouse(unittest.TestCase):
    """Test data warehouse functionality"""
    
    def setUp(self):
        self.warehouse = DataWarehouse()
        self.generator = SampleDataGenerator(seed=42)
        
        # Generate sample data
        self.sample_campaigns = self.generator.generate_campaign_data(days=30)
        self.sample_keywords = self.generator.generate_keyword_data(self.sample_campaigns)
        self.sample_products = self.generator.generate_product_data(days=30)
    
    def test_load_campaign_data(self):
        """Test loading campaign data"""
        initial_count = len(self.warehouse.campaigns)
        self.warehouse.load_campaign_data(self.sample_campaigns)
        
        self.assertGreater(len(self.warehouse.campaigns), initial_count)
    
    def test_get_campaign_performance(self):
        """Test retrieving campaign performance"""
        self.warehouse.load_campaign_data(self.sample_campaigns)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        performance_data = self.warehouse.get_campaign_performance(start_date, end_date)
        
        self.assertIsInstance(performance_data, pd.DataFrame)
    
    def test_get_unified_dataset(self):
        """Test unified dataset creation"""
        self.warehouse.load_campaign_data(self.sample_campaigns)
        self.warehouse.load_product_data(self.sample_products)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        unified_data = self.warehouse.get_unified_dataset(start_date, end_date)
        
        self.assertIsInstance(unified_data, pd.DataFrame)

class TestAttributionFeatureEngine(unittest.TestCase):
    """Test attribution feature engineering"""
    
    def setUp(self):
        self.attribution_engine = AttributionFeatureEngine()
        self.generator = SampleDataGenerator(seed=42)
        
        # Generate sample data
        campaigns = self.generator.generate_campaign_data(days=30)
        keywords = self.generator.generate_keyword_data(campaigns)
        self.attribution_data = self.generator.generate_attribution_data(campaigns, keywords)
    
    def test_calculate_time_decay_weights(self):
        """Test time decay attribution calculation"""
        result = self.attribution_engine.calculate_time_decay_weights(self.attribution_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('time_decay_weight' in result.columns)
        self.assertTrue('normalized_weight' in result.columns)
        
        # Check weights are between 0 and 1
        self.assertTrue((result['time_decay_weight'] >= 0).all())
        self.assertTrue((result['time_decay_weight'] <= 1).all())
    
    def test_calculate_position_weights(self):
        """Test position-based attribution calculation"""
        result = self.attribution_engine.calculate_position_weights(self.attribution_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('position_weight' in result.columns)
        self.assertTrue('touchpoint_position' in result.columns)
    
    def test_create_attribution_features(self):
        """Test comprehensive attribution feature creation"""
        campaigns = self.generator.generate_campaign_data(days=30)
        
        features = self.attribution_engine.create_attribution_features(
            self.attribution_data, campaigns
        )
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), len(self.attribution_data.columns))

class TestFinancialFeatureEngine(unittest.TestCase):
    """Test financial feature engineering"""
    
    def setUp(self):
        self.financial_engine = FinancialFeatureEngine()
        self.generator = SampleDataGenerator(seed=42)
        
        # Generate sample data
        campaigns = self.generator.generate_campaign_data(days=30)
        self.products = self.generator.generate_product_data(days=30)
        self.financial_data = self.generator.generate_financial_data(self.products)
        self.campaigns = campaigns
    
    def test_calculate_true_margins(self):
        """Test true margin calculation"""
        result = self.financial_engine.calculate_true_margins(
            self.campaigns, self.products, self.financial_data
        )
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('contribution_margin' in result.columns)
        self.assertTrue('true_roas' in result.columns)
    
    def test_calculate_working_capital_impact(self):
        """Test working capital impact calculation"""
        result = self.financial_engine.calculate_working_capital_adjusted_roas(
            revenue=1000, ad_spend=200, payment_terms_days=30,
            inventory_days=45, payable_days=30
        )
        
        self.assertIsInstance(result, dict)
        self.assertTrue('wc_adjusted_roas' in result)
        self.assertTrue('net_working_capital' in result)

class TestROICalculator(unittest.TestCase):
    """Test ROI calculator functionality"""
    
    def setUp(self):
        self.roi_calculator = FinancialROICalculator()
    
    def test_calculate_basic_roas(self):
        """Test basic ROAS calculation"""
        roas = self.roi_calculator.calculate_basic_roas(revenue=1000, ad_spend=250)
        self.assertEqual(roas, 4.0)
        
        # Test division by zero
        roas_zero = self.roi_calculator.calculate_basic_roas(revenue=1000, ad_spend=0)
        self.assertEqual(roas_zero, 0.0)
    
    def test_calculate_true_roas(self):
        """Test true ROAS calculation"""
        true_roas = self.roi_calculator.calculate_true_roas(
            revenue=1000, ad_spend=250, platform_fees=100, 
            fulfillment_fees=50, cost_of_goods=300
        )
        
        self.assertIsInstance(true_roas, float)
        self.assertGreater(true_roas, 0)
    
    def test_calculate_clv_roas(self):
        """Test CLV ROAS calculation"""
        clv_metrics = self.roi_calculator.calculate_customer_lifetime_value_roas(
            first_order_value=100, repeat_purchase_rate=0.3,
            average_order_frequency=4.0, gross_margin_rate=0.35,
            retention_periods=3, ad_spend=25
        )
        
        self.assertIsInstance(clv_metrics, dict)
        self.assertTrue('clv_roas' in clv_metrics)
        self.assertTrue('customer_lifetime_value' in clv_metrics)
    
    def test_comprehensive_roi_metrics(self):
        """Test comprehensive ROI calculation"""
        campaign_data = {
            'revenue': 1000,
            'ad_spend': 250,
            'avg_order_value': 50
        }
        
        metrics = self.roi_calculator.calculate_comprehensive_roi_metrics(campaign_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertTrue('basic_roas' in metrics)
        self.assertTrue('true_roas' in metrics)
        self.assertTrue('composite_roi_score' in metrics)

class TestMLModels(unittest.TestCase):
    """Test machine learning models"""
    
    def setUp(self):
        self.generator = SampleDataGenerator(seed=42)
        self.attribution_model = AttributionMLModel()
        self.performance_model = PerformancePredictionModel()
        self.budget_optimizer = BudgetOptimizationEngine()
    
    def test_attribution_model_training(self):
        """Test attribution model training"""
        # Generate attribution data with features
        campaigns = self.generator.generate_campaign_data(days=30)
        keywords = self.generator.generate_keyword_data(campaigns)
        attribution_data = self.generator.generate_attribution_data(campaigns, keywords)
        
        # Add required feature columns for testing
        attribution_data['time_decay_weight'] = np.random.random(len(attribution_data))
        attribution_data['position_weight'] = np.random.random(len(attribution_data))
        attribution_data['touchpoint_position'] = np.random.randint(1, 5, len(attribution_data))
        attribution_data['journey_length'] = np.random.randint(1, 8, len(attribution_data))
        attribution_data['journey_duration_days'] = np.random.randint(1, 30, len(attribution_data))
        attribution_data['campaign_spend'] = np.random.uniform(10, 1000, len(attribution_data))
        attribution_data['campaign_roas'] = np.random.uniform(1, 5, len(attribution_data))
        attribution_data['campaign_ctr'] = np.random.uniform(1, 10, len(attribution_data))
        attribution_data['campaign_conversion_rate'] = np.random.uniform(1, 15, len(attribution_data))
        attribution_data['is_first_touch'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['is_last_touch'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['is_middle_touch'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['platform_amazon'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['platform_walmart'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['touchpoint_click'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['touchpoint_view'] = np.random.randint(0, 2, len(attribution_data))
        attribution_data['spend_x_time_weight'] = attribution_data['campaign_spend'] * attribution_data['time_decay_weight']
        attribution_data['roas_x_position_weight'] = attribution_data['campaign_roas'] * attribution_data['position_weight']
        attribution_data['total_weighted_touches'] = np.random.uniform(1, 10, len(attribution_data))
        
        # Train model
        result = self.attribution_model.train(attribution_data)
        
        self.assertIsInstance(result, dict)
        self.assertTrue('individual_scores' in result)
        self.assertTrue('ensemble_score' in result)
        self.assertTrue(self.attribution_model.is_trained)
    
    def test_performance_model_training(self):
        """Test performance prediction model training"""
        campaigns = self.generator.generate_campaign_data(days=60)
        
        # Train models
        results = self.performance_model.train_performance_models(campaigns)
        
        self.assertIsInstance(results, dict)
        self.assertTrue(len(results) > 0)
    
    def test_budget_optimization(self):
        """Test budget optimization"""
        campaigns = self.generator.generate_campaign_data(days=30)
        
        # Calculate expected returns
        expected_returns = self.budget_optimizer.calculate_expected_returns(campaigns)
        self.assertIsInstance(expected_returns, dict)
        self.assertGreater(len(expected_returns), 0)
        
        # Calculate covariance matrix
        covariance_matrix = self.budget_optimizer.calculate_covariance_matrix(campaigns)
        self.assertIsInstance(covariance_matrix, np.ndarray)
        
        # Run optimization
        result = self.budget_optimizer.optimize_portfolio(total_budget=10000)
        self.assertIsInstance(result, dict)

class TestOrchestrator(unittest.TestCase):
    """Test orchestration engine"""
    
    def setUp(self):
        self.orchestrator = AdvertisingROIOrchestrator()
        self.generator = SampleDataGenerator(seed=42)
        
        # Generate sample data
        self.sample_data = self.generator.generate_all_sample_data(days=30)
    
    def test_load_data(self):
        """Test data loading"""
        self.orchestrator.load_data(self.sample_data)
        
        self.assertIsNotNone(self.orchestrator.last_update)
        self.assertGreater(len(self.orchestrator.data_warehouse.campaigns), 0)
    
    def test_run_attribution_analysis(self):
        """Test attribution analysis"""
        self.orchestrator.load_data(self.sample_data)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = self.orchestrator.run_attribution_analysis(start_date, end_date)
        
        self.assertIsInstance(result, dict)
        self.assertTrue('status' in result)
    
    def test_run_financial_analysis(self):
        """Test financial analysis"""
        self.orchestrator.load_data(self.sample_data)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = self.orchestrator.run_financial_analysis(start_date, end_date)
        
        self.assertIsInstance(result, dict)
        self.assertTrue('status' in result)
    
    def test_full_analysis(self):
        """Test full analysis pipeline"""
        self.orchestrator.load_data(self.sample_data)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        result = self.orchestrator.run_full_analysis(
            start_date, end_date, total_budget=50000
        )
        
        self.assertIsInstance(result, dict)
        self.assertTrue('attribution' in result)
        self.assertTrue('financial' in result)
        self.assertTrue('recommendations' in result)

class TestReportingEngine(unittest.TestCase):
    """Test reporting engine"""
    
    def setUp(self):
        self.reporting_engine = ReportingEngine()
        
        # Create mock analysis results
        self.mock_results = {
            'period': {'start': datetime.now() - timedelta(days=30), 'end': datetime.now()},
            'executive_summary': {
                'overall_status': 'healthy',
                'key_metrics': {'avg_true_roas': 3.5},
                'critical_issues': [],
                'next_actions': ['Optimize campaign targeting']
            },
            'recommendations': [
                {
                    'priority': 'High',
                    'category': 'Attribution Optimization',
                    'title': 'Improve Attribution Tracking',
                    'expected_impact': 'Could improve ROI by 15%'
                }
            ]
        }
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        report = self.reporting_engine.generate_comprehensive_report(self.mock_results)
        
        self.assertIsInstance(report, dict)
        self.assertTrue('report_metadata' in report)
        self.assertTrue('executive_summary' in report)
        self.assertTrue('recommendations' in report)
    
    def test_calculate_kpis(self):
        """Test KPI calculation"""
        kpis = self.reporting_engine._calculate_kpis(self.mock_results)
        
        self.assertIsInstance(kpis, dict)
        self.assertTrue('financial_kpis' in kpis)
        self.assertTrue('efficiency_kpis' in kpis)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.orchestrator = AdvertisingROIOrchestrator()
        self.generator = SampleDataGenerator(seed=42)
        self.reporting_engine = ReportingEngine()
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        
        # Step 1: Generate data
        sample_data = self.generator.generate_all_sample_data(days=30)
        self.assertIsInstance(sample_data, dict)
        self.assertGreater(len(sample_data['campaigns']), 0)
        
        # Step 2: Load data
        self.orchestrator.load_data(sample_data)
        self.assertIsNotNone(self.orchestrator.last_update)
        
        # Step 3: Run analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        results = self.orchestrator.run_full_analysis(
            start_date, end_date, total_budget=25000
        )
        
        self.assertIsInstance(results, dict)
        self.assertTrue('analysis_date' in results)
        
        # Step 4: Generate report
        report = self.reporting_engine.generate_comprehensive_report(results)
        self.assertIsInstance(report, dict)
        self.assertTrue('report_metadata' in report)
        
        # Verify key components work together
        self.assertTrue(len(results.get('recommendations', [])) >= 0)

def run_performance_tests():
    """Run performance tests for large datasets"""
    
    print("Running performance tests...")
    
    generator = SampleDataGenerator(seed=42)
    
    # Test with larger dataset
    start_time = datetime.now()
    large_data = generator.generate_all_sample_data(days=180)  # 6 months
    generation_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Large dataset generation: {generation_time:.2f} seconds")
    print(f"Records generated: {sum(len(df) for df in large_data.values()):,}")
    
    # Test orchestrator with large dataset
    orchestrator = AdvertisingROIOrchestrator()
    
    start_time = datetime.now()
    orchestrator.load_data(large_data)
    load_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Data loading time: {load_time:.2f} seconds")
    
    # Test analysis performance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    start_time = datetime.now()
    results = orchestrator.run_full_analysis(start_date, end_date, total_budget=100000)
    analysis_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Full analysis time: {analysis_time:.2f} seconds")
    
    return {
        'generation_time': generation_time,
        'load_time': load_time,
        'analysis_time': analysis_time,
        'total_records': sum(len(df) for df in large_data.values())
    }

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSampleDataGenerator,
        TestDataProcessor, 
        TestDataWarehouse,
        TestAttributionFeatureEngine,
        TestFinancialFeatureEngine,
        TestROICalculator,
        TestMLModels,
        TestOrchestrator,
        TestReportingEngine,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    print("🧪 Running ROI Engine Unit Tests...")
    print("=" * 60)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Run performance tests
    print("\n" + "=" * 60)
    print("🚀 Running Performance Tests...")
    performance_results = run_performance_tests()
    
    print("\nPerformance Summary:")
    for metric, value in performance_results.items():
        if 'time' in metric:
            print(f"  {metric}: {value:.2f} seconds")
        else:
            print(f"  {metric}: {value:,}")
    
    print("\n✅ All tests completed!")
