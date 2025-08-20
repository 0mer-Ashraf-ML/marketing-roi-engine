"""
Main execution script demonstrating the complete advertising ROI optimization workflow.
File: main.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
from pathlib import Path
import json
from typing import Any, Dict, List

warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our modules
from data.sample_data import SampleDataGenerator
from data.models import DataWarehouse
from engine.orchestrator import AdvertisingROIOrchestrator
from reporting.analytics import ReportingEngine, export_report_to_formats

def main():
    """Main execution function demonstrating the complete workflow"""
    
    print("=" * 80)
    print("🚀 ADVERTISING ROI OPTIMIZATION & ATTRIBUTION ENGINE")
    print("=" * 80)
    print()
    
    # Step 1: Generate Sample Data
    print("📊 Step 1: Generating Sample Data...")
    print("-" * 40)
    
    data_generator = SampleDataGenerator(seed=42)
    sample_data = data_generator.generate_all_sample_data(days=90)
    
    print("✅ Sample data generated successfully!")
    print(f"   • Campaigns: {len(sample_data['campaigns']):,} records")
    print(f"   • Keywords: {len(sample_data['keywords']):,} records") 
    print(f"   • Products: {len(sample_data['products']):,} records")
    print(f"   • Attribution: {len(sample_data['attribution']):,} records")
    print()
    
    # Step 2: Initialize System
    print("⚙️  Step 2: Initializing ROI Optimization System...")
    print("-" * 40)
    
    # Custom configuration
    config = {
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
            'rebalance_frequency': 'weekly',
            'min_allocation': 0.05,
            'max_allocation': 0.40
        }
    }
    
    orchestrator = AdvertisingROIOrchestrator(config=config)
    print("✅ System initialized successfully!")
    print()
    
    # Step 3: Load Data
    print("📥 Step 3: Loading Data into System...")
    print("-" * 40)
    
    orchestrator.load_data(sample_data)
    print("✅ Data loaded successfully!")
    print()
    
    # Step 4: Run Complete Analysis
    print("🔍 Step 4: Running Complete ROI Analysis...")
    print("-" * 40)
    
    # Define analysis period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    total_budget = 100000.0  # $100K budget for optimization
    
    print(f"   • Analysis Period: {start_date.date()} to {end_date.date()}")
    print(f"   • Optimization Budget: ${total_budget:,.2f}")
    print()
    
    # Run full analysis
    analysis_results = orchestrator.run_full_analysis(
        start_date=start_date,
        end_date=end_date,
        total_budget=total_budget
    )
    
    print("✅ Analysis completed successfully!")
    print()
    
    # Step 5: Display Key Results
    print("📈 Step 5: Key Analysis Results")
    print("-" * 40)
    
    display_analysis_summary(analysis_results)
    
    # Step 6: Generate Comprehensive Report
    print("📋 Step 6: Generating Comprehensive Report...")
    print("-" * 40)
    
    reporting_engine = ReportingEngine()
    comprehensive_report = reporting_engine.generate_comprehensive_report(analysis_results)
    
    print("✅ Report generated successfully!")
    print()
    
    # Step 7: Export Results
    print("💾 Step 7: Exporting Results...")
    print("-" * 40)
    
    # Create output directories
    Path("output").mkdir(exist_ok=True)
    Path("output/reports").mkdir(exist_ok=True)
    Path("output/data").mkdir(exist_ok=True)
    
    # Export comprehensive report
    report_files = export_report_to_formats(comprehensive_report, "output/reports/")
    
    # Export sample data for reference
    for data_type, df in sample_data.items():
        filename = f"output/data/{data_type}.csv"
        df.to_csv(filename, index=False)
    
    # Export key analysis results
    if analysis_results['financial']['status'] == 'success':
        analysis_results['financial']['roi_metrics'].to_csv(
            "output/data/roi_analysis.csv", index=False
        )
    
    print("✅ Results exported successfully!")
    for format_type, filepath in report_files.items():
        print(f"   • {format_type.upper()}: {filepath}")
    print()
    
    # Step 8: Display Recommendations
    print("💡 Step 8: Strategic Recommendations")
    print("-" * 40)
    
    display_recommendations(analysis_results.get('recommendations', []))
    
    print()
    print("=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"Check the 'output' directory for detailed reports and data exports.")
    print("Thank you for using the Advertising ROI Optimization Engine!")
    print()

def display_analysis_summary(analysis_results: Dict[str, any]) -> None:
    """Display key analysis results in a formatted way"""
    
    # Executive Summary
    exec_summary = analysis_results.get('executive_summary', {})
    print(f"🎯 Overall Health Status: {exec_summary.get('overall_status', 'Unknown').upper()}")
    print()
    
    # Financial Performance
    if 'financial' in analysis_results and analysis_results['financial']['status'] == 'success':
        roi_data = analysis_results['financial']['roi_metrics']
        
        print("💰 FINANCIAL PERFORMANCE:")
        print(f"   • Total Ad Spend: ${roi_data.get('ad_spend', pd.Series([0])).sum():,.2f}")
        print(f"   • Total Revenue: ${roi_data.get('revenue', pd.Series([0])).sum():,.2f}")
        print(f"   • Average True ROAS: {roi_data.get('true_roas', pd.Series([0])).mean():.2f}x")
        print(f"   • Average Contribution ROAS: {roi_data.get('contribution_roas', pd.Series([0])).mean():.2f}x")
        print(f"   • Composite ROI Score: {roi_data.get('composite_roi_score', pd.Series([0])).mean():.2f}")
        print()
    
    # Attribution Insights
    if 'attribution' in analysis_results and analysis_results['attribution']['status'] == 'success':
        attr_insights = analysis_results['attribution']['insights']
        
        print("🎯 ATTRIBUTION INSIGHTS:")
        print(f"   • Average Customer Journey Length: {attr_insights.get('avg_journey_length', 0):.1f} touchpoints")
        
        top_touchpoints = attr_insights.get('top_touchpoints', {})
        if top_touchpoints:
            best_touchpoint = max(top_touchpoints.items(), key=lambda x: x[1])
            print(f"   • Top Performing Touchpoint: {best_touchpoint[0]} (${best_touchpoint[1]:,.2f})")
        
        platform_perf = attr_insights.get('platform_performance', {})
        if platform_perf:
            print(f"   • Platform Performance: {dict(platform_perf)}")
        print()
    
    # Optimization Results
    if 'optimization' in analysis_results and analysis_results['optimization']['status'] == 'success':
        opt_data = analysis_results['optimization']['optimization']
        
        print("⚖️ OPTIMIZATION RESULTS:")
        print(f"   • Portfolio Expected Return: {opt_data.get('expected_return', 0):.2f}x")
        print(f"   • Portfolio Risk Level: {opt_data.get('expected_risk', 0):.4f}")
        print(f"   • Sharpe Ratio: {opt_data.get('sharpe_ratio', 0):.2f}")
        
        allocations = opt_data.get('allocations', {})
        if allocations:
            top_allocation = max(allocations.items(), key=lambda x: x[1])
            print(f"   • Largest Recommended Allocation: {top_allocation[0]} (${top_allocation[1]:,.2f})")
        print()
    
    # Prediction Insights
    if 'prediction' in analysis_results and analysis_results['prediction']['status'] == 'success':
        pred_insights = analysis_results['prediction']['insights']
        
        print("🔮 PREDICTIVE INSIGHTS:")
        
        declining_campaigns = []
        for target, insights in pred_insights.items():
            if isinstance(insights, dict):
                declining = insights.get('declining_performance_campaigns', [])
                declining_campaigns.extend(declining)
        
        print(f"   • Campaigns Predicted to Decline: {len(set(declining_campaigns))}")
        
        if 'prediction' in analysis_results:
            model_perf = analysis_results['prediction'].get('model_performance', {})
            if model_perf:
                avg_score = np.mean([
                    scores.get('ensemble_score', 0) 
                    for scores in model_perf.values() 
                    if isinstance(scores, dict)
                ])
                print(f"   • Average Model Accuracy: {avg_score:.1%}")
        print()

def display_recommendations(recommendations: List[Dict[str, any]]) -> None:
    """Display recommendations in a formatted way"""
    
    if not recommendations:
        print("No specific recommendations generated.")
        return
    
    # Group by priority
    high_priority = [r for r in recommendations if r.get('priority') == 'High']
    medium_priority = [r for r in recommendations if r.get('priority') == 'Medium']
    low_priority = [r for r in recommendations if r.get('priority') == 'Low']
    
    def display_recommendation_group(recs: List[Dict], priority_level: str, emoji: str):
        if recs:
            print(f"{emoji} {priority_level.upper()} PRIORITY ({len(recs)} items):")
            for i, rec in enumerate(recs[:3], 1):  # Show top 3
                print(f"   {i}. {rec.get('title', 'Unknown')}")
                print(f"      Category: {rec.get('category', 'General')}")
                print(f"      Impact: {rec.get('expected_impact', 'Unknown')}")
                
                action_items = rec.get('action_items', [])
                if action_items:
                    print(f"      Actions: {', '.join(action_items[:2])}")
                print()
    
    display_recommendation_group(high_priority, "High", "🚨")
    display_recommendation_group(medium_priority, "Medium", "⚠️")
    display_recommendation_group(low_priority, "Low", "ℹ️")

def run_sample_analysis_workflow():
    """Run a simplified workflow for demonstration"""
    
    print("🔬 Running Sample Analysis Workflow...")
    print("-" * 50)
    
    # Generate smaller dataset for quick demo
    generator = SampleDataGenerator(seed=42)
    
    print("Generating sample data (30 days)...")
    sample_data = generator.generate_all_sample_data(days=30)
    
    print("Initializing system...")
    orchestrator = AdvertisingROIOrchestrator()
    orchestrator.load_data(sample_data)
    
    print("Running attribution analysis...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    attribution_results = orchestrator.run_attribution_analysis(start_date, end_date)
    
    if attribution_results['status'] == 'success':
        print("✅ Attribution analysis completed!")
        print(f"   Average journey length: {attribution_results['insights'].get('avg_journey_length', 0):.1f}")
    else:
        print("❌ Attribution analysis failed")
    
    print("Running financial analysis...")
    financial_results = orchestrator.run_financial_analysis(start_date, end_date)
    
    if financial_results['status'] == 'success':
        print("✅ Financial analysis completed!")
        roi_metrics = financial_results['roi_metrics']
        print(f"   Average True ROAS: {roi_metrics['true_roas'].mean():.2f}x")
    else:
        print("❌ Financial analysis failed")
    
    print("Running budget optimization...")
    optimization_results = orchestrator.run_budget_optimization(total_budget=50000)
    
    if optimization_results['status'] == 'success':
        print("✅ Budget optimization completed!")
        opt_data = optimization_results['optimization']
        print(f"   Expected portfolio return: {opt_data.get('expected_return', 0):.2f}x")
    else:
        print("❌ Budget optimization failed")
    
    print("\n🎉 Sample workflow completed successfully!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advertising ROI Optimization Engine')
    parser.add_argument('--mode', choices=['full', 'sample'], default='full',
                       help='Run mode: full analysis or sample workflow')
    parser.add_argument('--days', type=int, default=90,
                       help='Number of days of sample data to generate')
    parser.add_argument('--budget', type=float, default=100000,
                       help='Total budget for optimization')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        run_sample_analysis_workflow()
    else:
        main()
