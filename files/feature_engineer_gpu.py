# 02_feature_engineer_gpu.py - GPU-Accelerated Feature Engineering - FIXED VERSION
# Target: Create robust features without data leakage and with proper NaN handling

import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Try GPU imports but gracefully fallback to CPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - using CPU for feature engineering")

class GPUPrecisionFeatureEngineer:
    """ROBUST feature engineering without data leakage"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.df = None
        self.metadata = None
        self.feature_names = []
        self.amount_stats = {}
        self.gpu_available = GPU_AVAILABLE
        
    def _check_gpu(self):
        """Enhanced GPU check"""
        if not self.gpu_available:
            return False
        try:
            cp.cuda.runtime.getDeviceCount()
            test_array = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
            _ = cp.mean(test_array)
            del test_array
            return True
        except:
            return False
    
    def load_checkpoint(self):
        """Load validated data from Step 1"""
        print("📄 STEP 2: ROBUST FEATURE ENGINEERING") 
        print("=" * 60)
        
        try:
            # Load validated data
            data_path = f"{self.checkpoint_dir}/01_loaded_data_gpu.pkl"
            self.df = pd.read_pickle(data_path)
            print(f"✅ Loaded validated data: {self.df.shape}")
            
            # Load metadata
            meta_path = f"{self.checkpoint_dir}/01_metadata_gpu.pkl" 
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Data quality status: {self.metadata.get('data_quality', 'UNKNOWN')}")
            
            # Verify GPU availability
            self.gpu_available = self._check_gpu()
            print(f"🚀 GPU Available: {self.gpu_available}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            return False
    
    def engineer_features(self):
        """ROBUST feature engineering WITHOUT data leakage"""
        print(f"\n🔧 ROBUST FEATURE ENGINEERING (NO DATA LEAKAGE)...")
        
        # Start with copy
        data = self.df.copy()
        
        # CRITICAL FIX: Create hour features BEFORE filtering
        if 'step' in data.columns:
            data = self._create_critical_hour_features(data)
        
        # CRITICAL: Remove ALL potential leakage features upfront
        print("   🛡️ PREVENTING DATA LEAKAGE...")
        leakage_features = [
            'nameOrig', 'nameDest',        # High-cardinality IDs
            'isFlaggedFraud',             # System flag (leakage)
            'oldbalanceOrg', 'newbalanceOrig',   # Balance columns
            'oldbalanceDest', 'newbalanceDest'   # Balance columns
        ]
        
        # Track which leakage features actually exist
        existing_leakage = [col for col in leakage_features if col in data.columns]
        if existing_leakage:
            data = data.drop(columns=existing_leakage)
            print(f"      ✅ Removed leakage features: {existing_leakage}")
        
        # Filter to fraud-prone transactions for better signal-to-noise
        if 'type' in data.columns:
            print("   🎯 Filtering to fraud-prone transaction types...")
            original_size = len(data)
            fraud_prone_types = ['CASH_OUT', 'TRANSFER']
            available_types = data['type'].unique()
            
            valid_types = [t for t in fraud_prone_types if t in available_types]
            if valid_types:
                data = data[data['type'].isin(valid_types)].copy()
                print(f"      📉 Reduced from {original_size:,} to {len(data):,} transactions")
                print(f"      🎯 Using types: {valid_types}")
        
        # Continue with rest of feature engineering...
        print("   ⚡ Creating SAFE features (no leakage risk)...")
        
        # 1. Amount-based features (SAFE)
        if 'amount' in data.columns:
            data = self._create_safe_amount_features(data)
        
        # 2. Time features already created above
        
        # 3. Transaction type features (SAFE)
        if 'type' in data.columns:
            data = self._encode_transaction_types(data)
        
        # 4. Interaction features between safe variables
        data = self._create_safe_interactions(data)
        
        # 5. Risk scoring based on patterns (SAFE)
        data = self._create_pattern_risk_features(data)
        
        
        # Prepare final dataset
        if 'isFraud' not in data.columns:
            raise ValueError("Target variable 'isFraud' not found!")
            
        y = data['isFraud'].copy()
        
        # Remove non-feature columns
        feature_columns = data.columns.tolist()
        non_feature_cols = ['isFraud', 'type'] if 'type' in feature_columns else ['isFraud']
        
        X = data.drop(columns=non_feature_cols)
        
        # CRITICAL: Final data validation and cleaning
        X, y = self._final_validation_and_cleaning(X, y)
        
        self.feature_names = list(X.columns)
        
        print(f"\n✅ ROBUST FEATURE ENGINEERING COMPLETED:")
        print(f"   📊 Features created: {len(self.feature_names)}")
        print(f"   📊 Final dataset shape: {X.shape}")
        print(f"   🎯 Target distribution: {y.sum():,} fraud / {len(y):,} total ({y.mean()*100:.3f}%)")
        print(f"   🛡️ Data leakage: PREVENTED")
        print(f"   🧹 NaN values: HANDLED")
        
        # Memory cleanup
        del data
        gc.collect()
        if self.gpu_available:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        
        return X, y
    
    def _create_safe_amount_features(self, data):
        """Create amount features WITHOUT using balance information"""
        print("      💰 Safe amount features...")
        
        amount = data['amount'].copy()
        
        # GPU-accelerated if available
        if self.gpu_available:
            try:
                amount_gpu = cp.array(amount.values, dtype=cp.float32)
                
                # GPU computations
                stats = {
                    'mean': float(cp.mean(amount_gpu)),
                    'median': float(cp.median(amount_gpu)),
                    'std': float(cp.std(amount_gpu)),
                    'p25': float(cp.percentile(amount_gpu, 25)),
                    'p75': float(cp.percentile(amount_gpu, 75)),
                    'p90': float(cp.percentile(amount_gpu, 90)),
                    'p99': float(cp.percentile(amount_gpu, 99))
                }
                
                # GPU feature creation
                data['amount_log'] = cp.asnumpy(cp.log1p(amount_gpu))
                data['amount_sqrt'] = cp.asnumpy(cp.sqrt(amount_gpu))
                
                del amount_gpu
                cp.get_default_memory_pool().free_all_blocks()
                print("        🚀 GPU-accelerated amount features")
                
            except Exception as e:
                print(f"        ⚠️ GPU failed ({e}), using CPU")
                self.gpu_available = False
        
        if not self.gpu_available:
            # CPU fallback
            stats = {
                'mean': amount.mean(),
                'median': amount.median(),
                'std': amount.std(),
                'p25': amount.quantile(0.25),
                'p75': amount.quantile(0.75),
                'p90': amount.quantile(0.90),
                'p99': amount.quantile(0.99)
            }
            
            data['amount_log'] = np.log1p(amount)
            data['amount_sqrt'] = np.sqrt(amount)
        
        # Safe amount categorization
        data['amount_is_zero'] = (amount == 0).astype('int8')
        data['amount_is_round_hundred'] = (amount % 100 == 0).astype('int8')
        data['amount_is_round_thousand'] = (amount % 1000 == 0).astype('int8')
        
        # Amount size categories
        data['amount_tiny'] = (amount <= stats['p25']).astype('int8')
        data['amount_small'] = ((amount > stats['p25']) & (amount <= stats['median'])).astype('int8')
        data['amount_medium'] = ((amount > stats['median']) & (amount <= stats['p75'])).astype('int8')
        data['amount_large'] = ((amount > stats['p75']) & (amount <= stats['p90'])).astype('int8')
        data['amount_huge'] = (amount > stats['p90']).astype('int8')
        data['amount_extreme'] = (amount > stats['p99']).astype('int8')
        
        self.amount_stats = stats
        return data
    
    def _create_critical_hour_features(self, data):
        """Create CRITICAL hour features for fraud detection"""
        print("      🕐 Creating critical hour features...")
        
        if 'step' in data.columns:
            # Basic hour extraction
            data['hour'] = (data['step'] % 24).astype('int8')
            data['day'] = (data['step'] // 24).astype('int16')
            
            # CRITICAL: Fraud concentration hours (from your data analysis)
            data['is_fraud_hour_3_5'] = data['hour'].isin([3, 4, 5]).astype('int8')
            data['is_hour_3'] = (data['hour'] == 3).astype('int8')
            data['is_hour_4'] = (data['hour'] == 4).astype('int8')
            data['is_hour_5'] = (data['hour'] == 5).astype('int8')
            
            # Your data shows 22% fraud rate at hour 5!
            data['is_peak_fraud_hour'] = (data['hour'] == 5).astype('int8')
            
            print(f"        ✅ Added critical hour features (hours 3-5 have 16-22% fraud!)")
        
        return data
    
    def _create_time_features(self, data):
        """Create temporal features from step column"""
        print("      ⏰ Time-based features...")
        
        step = data['step']
        
        # Assume step represents time units (hours)
        data['hour'] = (step % 24).astype('int8')
        data['day'] = (step // 24).astype('int16') 
        data['day_of_week'] = (data['day'] % 7).astype('int8')
        
        # Time-based risk patterns
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype('int8')
        data['is_business_hours'] = ((data['hour'] >= 9) & (data['hour'] <= 17)).astype('int8')
        data['is_weekend'] = (data['day_of_week'].isin([5, 6])).astype('int8')
        
        # Cyclical encoding for better ML performance
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24).astype('float32')
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24).astype('float32')
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7).astype('float32')
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7).astype('float32')
        
        return data
    
    def _encode_transaction_types(self, data):
        """One-hot encode transaction types"""
        print("      🏷️ Transaction type encoding...")
        
        # One-hot encoding for transaction types
        type_dummies = pd.get_dummies(data['type'], prefix='type', dtype='int8')
        data = pd.concat([data, type_dummies], axis=1)
        
        # Create type risk scores based on domain knowledge
        type_risk = {
            'CASH_OUT': 0.8,    # High fraud risk
            'TRANSFER': 0.7,    # High fraud risk
            'CASH_IN': 0.2,     # Low fraud risk
            'PAYMENT': 0.1,     # Very low fraud risk
            'DEBIT': 0.3        # Medium fraud risk
        }
        
        # 🔧 FIX: Convert categorical to string first, then map
        data['type_risk_score'] = data['type'].astype(str).map(type_risk).fillna(0.5).astype('float32')
        
        return data
    
    def _create_safe_interactions(self, data):
        """Create interaction features between safe variables"""
        print("      🔗 Safe interaction features...")
        
        # Amount-time interactions
        if all(col in data.columns for col in ['amount_log', 'is_night']):
            data['night_amount_risk'] = (data['amount_log'] * data['is_night']).astype('float32')
        
        if all(col in data.columns for col in ['amount_log', 'is_weekend']):
            data['weekend_amount_risk'] = (data['amount_log'] * data['is_weekend']).astype('float32')
        
        # Amount-type interactions
        if 'type_risk_score' in data.columns and 'amount_log' in data.columns:
            data['amount_type_risk'] = (data['amount_log'] * data['type_risk_score']).astype('float32')
        
        return data
    
    def _create_pattern_risk_features(self, data):
        """Create pattern-based risk features"""
        print("      🎯 Pattern risk features...")
        
        # High-risk patterns
        risk_conditions = []
        
        # Night + large amount
        if all(col in data.columns for col in ['is_night', 'amount_large']):
            data['high_risk_night_large'] = (data['is_night'] & data['amount_large']).astype('int8')
            risk_conditions.append('high_risk_night_large')
        
        # Weekend + extreme amount
        if all(col in data.columns for col in ['is_weekend', 'amount_extreme']):
            data['high_risk_weekend_extreme'] = (data['is_weekend'] & data['amount_extreme']).astype('int8')
            risk_conditions.append('high_risk_weekend_extreme')
        
        # Round amounts (common in fraud)
        if all(col in data.columns for col in ['amount_is_round_thousand', 'amount_large']):
            data['suspicious_round_large'] = (data['amount_is_round_thousand'] & data['amount_large']).astype('int8')
            risk_conditions.append('suspicious_round_large')
        
        # Aggregate risk score
        if risk_conditions:
            data['total_risk_flags'] = data[risk_conditions].sum(axis=1).astype('int8')
        
        return data
    
    def _final_validation_and_cleaning(self, X, y):
        """Final validation and cleaning to ensure no NaN/Inf values"""
        print("   🧹 FINAL DATA VALIDATION & CLEANING...")
        
        # Check for NaN values
        nan_counts = X.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"      ⚠️ Found {nan_counts.sum()} NaN values across {(nan_counts > 0).sum()} columns")
            for col in nan_counts[nan_counts > 0].index:
                print(f"        {col}: {nan_counts[col]} NaNs")
                
            # Fill NaN values intelligently
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if X[col].dtype in ['float32', 'float64']:
                        fill_value = 0.0
                    else:
                        fill_value = 0
                    
                    X[col].fillna(fill_value, inplace=True)
                    print(f"        ✅ Filled {col} NaNs with {fill_value}")
        
        # Check for infinite values
        inf_counts = {}
        for col in X.select_dtypes(include=[np.number]).columns:
            inf_count = np.isinf(X[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            print(f"      ⚠️ Found infinite values: {inf_counts}")
            X = X.replace([np.inf, -np.inf], 0)
            print("        ✅ Replaced infinite values with 0")
        
        # Ensure consistent data types
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"      🔧 Converting object column {col} to numeric")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str)).astype('int32')
            elif X[col].dtype in ['int64', 'float64']:
                # Convert to memory-efficient types
                if X[col].dtype == 'float64':
                    X[col] = X[col].astype('float32')
                elif X[col].dtype == 'int64':
                    X[col] = pd.to_numeric(X[col], downcast='integer')
        
        # Validate target variable
        if y.isnull().sum() > 0:
            print(f"      ⚠️ Found {y.isnull().sum()} NaN values in target")
            y = y.fillna(0)
            print("        ✅ Filled target NaNs with 0")
        
        y = y.astype('int8')  # Ensure binary target
        
        # Final validation
        print(f"      ✅ Final dataset validated:")
        print(f"        📊 Features: {X.shape}")
        print(f"        🎯 Target: fraud rate {y.mean()*100:.3f}%")
        print(f"        🧹 NaN values: {X.isnull().sum().sum()} (should be 0)")
        print(f"        🧹 Inf values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum()} (should be 0)")
        
        return X, y
    
    def save_checkpoint(self, X, y, output_dir='/kaggle/working'):
        """Save engineered features and metadata"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save features and target
        X.to_pickle(f"{output_dir}/02_features_gpu.pkl")
        y.to_pickle(f"{output_dir}/02_target_gpu.pkl")
        
        # Save feature metadata
        feature_metadata = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'amount_stats': self.amount_stats,
            'gpu_used': self.gpu_available,
            'leakage_prevented': True,
            'data_validated': True,
            'feature_types': dict(X.dtypes),
            'target_distribution': {
                'fraud_count': int(y.sum()),
                'total_count': len(y),
                'fraud_rate': float(y.mean())
            }
        }
        
        with open(f"{output_dir}/02_feature_metadata_gpu.pkl", 'wb') as f:
            pickle.dump(feature_metadata, f)
        
        print(f"\n💾 STEP 2 CHECKPOINT SAVED:")
        print(f"   📊 Features: 02_features_gpu.pkl ({X.shape})")
        print(f"   🎯 Target: 02_target_gpu.pkl")
        print(f"   📋 Metadata: 02_feature_metadata_gpu.pkl")
        print(f"   🛡️ Leakage prevention: ENABLED")
        
        return True

def main():
    """Main execution for Step 2"""
    print("🚀 STARTING STEP 2: ROBUST FEATURE ENGINEERING")
    
    engineer = GPUPrecisionFeatureEngineer()
    
    try:
        if not engineer.load_checkpoint():
            print("❌ STEP 2 FAILED: Could not load data from Step 1")
            return False
        
        X, y = engineer.engineer_features()
        engineer.save_checkpoint(X, y)
        
        print("\n✅ STEP 2 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 3: Data Splitting & Sampling")
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)