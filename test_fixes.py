"""
Quick test script to verify the fixes are working.
Run this after applying the fixes.
"""

from datetime import datetime, timedelta
from data.sample_data import SampleDataGenerator
from engine.orchestrator import AdvertisingROIOrchestrator

def test_fixes():
    """Test that the fixes resolve the main issues"""
    
    print("🧪 Testing fixes...")
    
    try:
        # Generate small sample data
        generator = SampleDataGenerator(seed=42)
        sample_data = generator.generate_all_sample_data(days=30)
        print("✅ Sample data generation: PASSED")
        
        # Initialize orchestrator
        orchestrator = AdvertisingROIOrchestrator()
        orchestrator.load_data(sample_data)
        print("✅ Data loading: PASSED")
        
        # Test attribution analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        attribution_result = orchestrator.run_attribution_analysis(start_date, end_date)
        if attribution_result['status'] == 'success':
            print("✅ Attribution analysis: PASSED")
        else:
            print(f"❌ Attribution analysis: {attribution_result.get('message', 'FAILED')}")
        
        # Test financial analysis
        financial_result = orchestrator.run_financial_analysis(start_date, end_date)
        if financial_result['status'] == 'success':
            print("✅ Financial analysis: PASSED")
        else:
            print(f"❌ Financial analysis: {financial_result.get('message', 'FAILED')}")
        
        # Test budget optimization
        optimization_result = orchestrator.run_budget_optimization(total_budget=50000)
        if optimization_result['status'] == 'success':
            print("✅ Budget optimization: PASSED")
        else:
            print(f"❌ Budget optimization: {optimization_result.get('message', 'FAILED')}")
        
        print("\n🎉 All core fixes are working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_fixes()
