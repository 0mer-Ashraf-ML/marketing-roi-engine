"""
Debug script to identify and fix model training issues.
Run this to diagnose the perfect score problem.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_attribution_model():
    """Debug the attribution model training issues"""
    
    print("🔍 DEBUGGING ATTRIBUTION MODEL")
    print("=" * 50)
    
    # Load sample data to examine
    from data.sample_data import SampleDataGenerator
    generator = SampleDataGenerator(seed=42)
    sample_data = generator.generate_all_sample_data(days=30)
    
    attribution_data = sample_data['attribution']
    campaigns = sample_data['campaigns']
    
    print(f"Attribution data shape: {attribution_data.shape}")
    print(f"Attribution columns: {list(attribution_data.columns)}")
    
    # Check for data leakage issues
    print("\n🚨 CHECKING FOR DATA LEAKAGE:")
    
    # 1. Check if target variable has variance
    if 'conversion_value' in attribution_data.columns:
        conv_values = attribution_data['conversion_value']
        print(f"Conversion value stats:")
        print(f"  Min: {conv_values.min()}, Max: {conv_values.max()}")
        print(f"  Unique values: {conv_values.nunique()}")
        print(f"  Std dev: {conv_values.std()}")
        
        if conv_values.std() == 0:
            print("❌ PROBLEM: All conversion values are identical!")
    
    # 2. Check attribution features
    from features.attribution import AttributionFeatureEngine
    
    try:
        attr_engine = AttributionFeatureEngine()
        features = attr_engine.create_attribution_features(attribution_data, campaigns)
        
        print(f"\nFeature matrix shape: {features.shape}")
        print(f"Features created: {list(features.columns)}")
        
        # Check target variable
        if 'attributed_revenue' in features.columns:
            target = features['attributed_revenue']
            print(f"\nTarget variable 'attributed_revenue':")
            print(f"  Min: {target.min()}, Max: {target.max()}")
            print(f"  Unique values: {target.nunique()}")
            print(f"  Std dev: {target.std()}")
            
            if target.std() == 0:
                print("❌ PROBLEM: All attributed revenue values are identical!")
                return False
        
        # Check for leakage - target in features
        feature_cols = [col for col in features.columns if col not in ['attributed_revenue', 'customer_id', 'touchpoint_id']]
        
        print(f"\nFeature columns for ML: {len(feature_cols)}")
        
        # Check if any features perfectly correlate with target
        if 'attributed_revenue' in features.columns:
            target = features['attributed_revenue']
            
            perfect_correlations = []
            for col in feature_cols:
                if features[col].dtype in ['float64', 'int64']:
                    corr = np.corrcoef(features[col], target)[0, 1]
                    if abs(corr) > 0.99:
                        perfect_correlations.append((col, corr))
            
            if perfect_correlations:
                print("\n❌ PROBLEM: Perfect correlations found (data leakage):")
                for col, corr in perfect_correlations:
                    print(f"  {col}: {corr:.6f}")
                return False
            else:
                print("\n✅ No obvious data leakage detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in attribution features: {e}")
        return False

def debug_performance_model():
    """Debug the performance model initialization"""
    
    print("\n🔍 DEBUGGING PERFORMANCE MODEL")
    print("=" * 50)
    
    from models.ml_engine import PerformancePredictionModel
    
    try:
        model = PerformancePredictionModel()
        
        # Check if models attribute exists
        if hasattr(model, 'models'):
            print(f"✅ Models attribute exists: {list(model.models.keys())}")
        else:
            print("❌ Models attribute missing")
            
        # Check if required libraries are available
        try:
            import xgboost as xgb
            print("✅ XGBoost available")
        except ImportError:
            print("❌ XGBoost not available")
            
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            print("✅ Scikit-learn available")
        except ImportError:
            print("❌ Scikit-learn not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing performance model: {e}")
        return False

def debug_data_quality():
    """Debug data quality issues"""
    
    print("\n🔍 DEBUGGING DATA QUALITY")
    print("=" * 50)
    
    from data.sample_data import SampleDataGenerator
    generator = SampleDataGenerator(seed=42)
    sample_data = generator.generate_all_sample_data(days=30)
    
    for data_name, df in sample_data.items():
        print(f"\n{data_name.upper()}:")
        print(f"  Shape: {df.shape}")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {df.duplicated().sum()}")
        
        # Check for constant columns
        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"  ❌ Constant columns: {constant_cols}")
        else:
            print(f"  ✅ No constant columns")

def create_fixed_attribution_training():
    """Create a fixed version that prevents data leakage"""
    
    print("\n🔧 CREATING FIXED ATTRIBUTION TRAINING")
    print("=" * 50)
    
    fixed_code = '''
def train_attribution_model_fixed(attribution_features):
    """Fixed training that prevents data leakage"""
    
    # CRITICAL FIX: Ensure no target leakage
    feature_cols = [
        'time_decay_weight', 'position_weight', 'touchpoint_position', 
        'journey_length', 'journey_duration_days', 'campaign_spend', 
        'campaign_roas', 'campaign_ctr', 'is_first_touch', 'is_last_touch',
        'platform_amazon', 'platform_walmart', 'touchpoint_click'
    ]
    
    # Only use features that exist
    available_features = [col for col in feature_cols if col in attribution_features.columns]
    
    if len(available_features) < 3:
        print("❌ Insufficient features for training")
        return None
    
    X = attribution_features[available_features]
    y = attribution_features['attributed_revenue']
    
    # Check for data quality
    if y.std() == 0:
        print("❌ No variance in target variable")
        return None
    
    # Proper train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Target variance: {y.std():.4f}")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'features_used': available_features
    }
    '''
    
    print("Fixed training code created above ☝️")
    return fixed_code

def main():
    """Run all debugging checks"""
    
    print("🚨 MODEL TRAINING DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Run diagnostics
    attr_ok = debug_attribution_model()
    perf_ok = debug_performance_model()
    debug_data_quality()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    print(f"Attribution Model: {'✅ OK' if attr_ok else '❌ ISSUES'}")
    print(f"Performance Model: {'✅ OK' if perf_ok else '❌ ISSUES'}")
    
    if not attr_ok or not perf_ok:
        print("\n🔧 RECOMMENDED FIXES:")
        if not attr_ok:
            print("1. Fix attribution data leakage")
            print("2. Ensure target variable has variance") 
            print("3. Implement proper train/test split")
        if not perf_ok:
            print("4. Fix performance model initialization")
            print("5. Check ML library dependencies")
        
        create_fixed_attribution_training()
    else:
        print("\n✅ All models appear to be working correctly")

if __name__ == "__main__":
    main()