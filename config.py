"""
Configuration file for the advertising ROI optimization system.
File: config.py
"""

from typing import Dict, Any
import os
from pathlib import Path

class Config:
    """Configuration management for the ROI optimization system"""
    
    # Data Configuration
    DATA_CONFIG = {
        'sample_data': {
            'default_days': 90,
            'campaigns_count': 25,
            'seed': 42
        },
        'data_sources': {
            'campaigns_table': 'campaigns',
            'keywords_table': 'keywords', 
            'products_table': 'products',
            'financial_table': 'financial_data',
            'attribution_table': 'attribution_events'
        },
        'data_quality': {
            'min_records_for_training': 50,
            'max_missing_rate': 0.3,
            'outlier_threshold': 3.0
        }
    }
    
    # Attribution Model Configuration  
    ATTRIBUTION_CONFIG = {
        'models': {
            'time_decay': {
                'decay_rate': 0.1,
                'window_days': 30
            },
            'position_based': {
                'first_touch_weight': 0.4,
                'last_touch_weight': 0.4,
                'middle_touch_weight': 0.2
            },
            'shapley_value': {
                'max_coalition_size': 10,
                'monte_carlo_iterations': 1000
            },
            'markov_chain': {
                'transition_window': 14,
                'convergence_threshold': 0.001
            }
        },
        'ensemble': {
            'default_weights': {
                'time_decay': 0.3,
                'position': 0.2,
                'shapley': 0.3,
                'markov': 0.2
            },
            'adaptive_weighting': True,
            'performance_threshold': 0.7
        }
    }
    
    # Financial Analysis Configuration
    FINANCIAL_CONFIG = {
        'discount_rates': {
            'annual_rate': 0.12,
            'daily_rate': 0.12 / 365,
            'risk_free_rate': 0.03
        },
        'tax_settings': {
            'corporate_tax_rate': 0.25,
            'state_tax_rate': 0.05,
            'effective_tax_rate': 0.28
        },
        'working_capital': {
            'carrying_cost_rate': 0.15,
            'default_payment_terms': 30,
            'default_inventory_days': 45,
            'default_payable_days': 30
        },
        'platform_fees': {
            'amazon': {
                'referral_fee_rate': 0.15,
                'fba_fee_per_unit': 3.5,
                'storage_fee_per_day': 0.85
            },
            'walmart': {
                'referral_fee_rate': 0.12,
                'fulfillment_fee_per_unit': 3.0,
                'storage_fee_per_day': 0.65
            }
        },
        'customer_metrics': {
            'default_repeat_rate': 0.3,
            'default_order_frequency': 4.0,
            'default_retention_periods': 3,
            'default_gross_margin': 0.35
        }
    }
    
    # Machine Learning Configuration
    ML_CONFIG = {
        'models': {
            'attribution_models': {
                'gradient_boost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                },
                'xgboost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8
                }
            },
            'prediction_models': {
                'xgboost': {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 8
                },
                'gradient_boost': {
                    'n_estimators': 200,
                    'learning_rate': 0.05,
                    'max_depth': 8
                }
            }
        },
        'training': {
            'test_size': 0.2,
            'validation_split': 0.2,
            'cross_validation_folds': 5,
            'random_state': 42
        },
        'retraining': {
            'frequency': 'weekly',
            'performance_threshold': 0.7,
            'min_improvement': 0.05
        },
        'feature_engineering': {
            'lag_periods': [1, 3, 7],
            'rolling_windows': [7, 14, 30, 90],
            'max_features': 100
        }
    }
    
    # Optimization Configuration
    OPTIMIZATION_CONFIG = {
        'portfolio_optimization': {
            'risk_aversion': 0.5,
            'min_allocation': 0.05,
            'max_allocation': 0.40,
            'rebalance_frequency': 'daily',
            'transaction_cost': 0.001
        },
        'constraints': {
            'max_campaign_allocation': 0.40,
            'min_campaign_allocation': 0.05,
            'max_platform_allocation': 0.70,
            'diversification_threshold': 0.8
        },
        'algorithms': {
            'mean_variance': {
                'solver': 'ECOS',
                'max_iterations': 1000
            },
            'multi_armed_bandit': {
                'epsilon': 0.1,
                'decay_rate': 0.99
            },
            'reinforcement_learning': {
                'learning_rate': 0.1,
                'exploration_rate': 0.05,
                'discount_factor': 0.95
            }
        }
    }
    
    # Reporting Configuration
    REPORTING_CONFIG = {
        'output_formats': ['json', 'csv', 'excel'],
        'report_types': [
            'executive_summary',
            'attribution_analysis', 
            'financial_performance',
            'optimization_recommendations',
            'predictive_insights'
        ],
        'kpi_thresholds': {
            'roas_excellent': 4.0,
            'roas_good': 2.5,
            'roas_poor': 1.5,
            'health_score_excellent': 80,
            'health_score_good': 60,
            'health_score_poor': 40
        },
        'export_settings': {
            'include_raw_data': False,
            'include_model_details': True,
            'decimal_places': 3
        }
    }
    
    # System Configuration
    SYSTEM_CONFIG = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_path': 'logs/roi_engine.log'
        },
        'performance': {
            'max_workers': 4,
            'chunk_size': 10000,
            'memory_limit_gb': 8
        },
        'alerts': {
            'performance_decline_threshold': 0.15,
            'budget_overrun_threshold': 0.10,
            'roas_decline_threshold': 0.20,
            'model_accuracy_threshold': 0.60
        }
    }
    
    # Environment-specific configurations
    ENVIRONMENT_CONFIGS = {
        'development': {
            'debug': True,
            'sample_data_size': 30,
            'quick_mode': True
        },
        'staging': {
            'debug': False,
            'sample_data_size': 90,
            'quick_mode': False
        },
        'production': {
            'debug': False,
            'sample_data_size': 365,
            'quick_mode': False,
            'backup_enabled': True
        }
    }
    
    @classmethod
    def get_config(cls, environment: str = 'development') -> Dict[str, Any]:
        """Get complete configuration for specified environment"""
        
        base_config = {
            'data': cls.DATA_CONFIG,
            'attribution': cls.ATTRIBUTION_CONFIG,
            'financial': cls.FINANCIAL_CONFIG,
            'ml': cls.ML_CONFIG,
            'optimization': cls.OPTIMIZATION_CONFIG,
            'reporting': cls.REPORTING_CONFIG,
            'system': cls.SYSTEM_CONFIG
        }
        
        # Merge environment-specific settings
        env_config = cls.ENVIRONMENT_CONFIGS.get(environment, {})
        base_config['environment'] = env_config
        
        return base_config
    
    @classmethod
    def get_database_config(cls) -> Dict[str, str]:
        """Get database configuration from environment variables"""
        
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'advertising_roi'),
            'username': os.getenv('DB_USER', 'roi_user'),
            'password': os.getenv('DB_PASSWORD', ''),
            'connection_pool_size': int(os.getenv('DB_POOL_SIZE', '10'))
        }
    
    @classmethod 
    def get_api_config(cls) -> Dict[str, str]:
        """Get API configuration for external data sources"""
        
        return {
            'amazon_advertising': {
                'client_id': os.getenv('AMAZON_CLIENT_ID', ''),
                'client_secret': os.getenv('AMAZON_CLIENT_SECRET', ''),
                'refresh_token': os.getenv('AMAZON_REFRESH_TOKEN', ''),
                'api_endpoint': 'https://advertising-api.amazon.com'
            },
            'walmart_dsp': {
                'client_id': os.getenv('WALMART_CLIENT_ID', ''),
                'client_secret': os.getenv('WALMART_CLIENT_SECRET', ''),
                'api_endpoint': 'https://api.walmart.com/v3/dsp'
            },
            'rate_limits': {
                'requests_per_minute': 60,
                'requests_per_hour': 1000,
                'retry_attempts': 3,
                'backoff_factor': 2
            }
        }
