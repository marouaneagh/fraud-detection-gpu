# 02_feature_engineer_gpu.py - GPU-Accelerated Precision-Focused Feature Engineering
# Target: Create features using GPU acceleration for maximum performance

import pandas as pd
import numpy as np
import cupy as cp  # GPU-accelerated NumPy
import pickle
import gc
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class GPUPrecisionFeatureEngineer:
    """GPU-accelerated feature engineering optimized for precision"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.df = None
        self.metadata = None
        self.feature_names = []
        self.amount_stats = {}
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self):
        """Check GPU availability"""
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            print("⚠️ CuPy not available - using CPU for feature engineering")
            return False
    
    def load_checkpoint(self):
        """Load GPU-optimized data from Step 1"""
        print("🔄 STEP 2: GPU-ACCELERATED PRECISION FEATURE ENGINEERING") 
        print("=" * 60)
        
        try:
            # Load GPU-optimized data
            data_path = f"{self.checkpoint_dir}/01_loaded_data_gpu.pkl"
            self.df = pd.read_pickle(data_path)
            print(f"✅ Loaded GPU-ready data: {self.df.shape}")
            
            # Load metadata
            meta_path = f"{self.checkpoint_dir}/01_metadata_gpu.pkl" 
            with open(meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded metadata with GPU status: {self.metadata.get('gpu_used', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            return False
    
    def engineer_features(self):
        """GPU-accelerated feature engineering"""
        print(f"\n🚀 GPU-ACCELERATED FEATURE ENGINEERING...")
        print(f"   GPU Available: {self.gpu_available}")
        
        # Start with copy
        data = self.df.copy()
        
        # Step 1: Remove leakage features
        print("   🧹 Removing data leakage features...")
        leakage_features = ['nameOrig', 'nameDest', 'isFlaggedFraud', 
                           'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        data = data.drop([col for col in leakage_features if col in data.columns], axis=1)
        
        # Step 2: Filter to fraud-prone transactions
        print("   🎯 Filtering to CASH_OUT and TRANSFER transactions...")
        original_size = len(data)
        data = data[data['type'].isin(['CASH_OUT', 'TRANSFER'])].copy()
        print(f"      Reduced from {original_size:,} to {len(data):,} transactions")
        
        # Step 3: GPU-accelerated amount statistics
        print("   📊 GPU-accelerated amount statistics...")
        if self.gpu_available:
            self.amount_stats = self._gpu_calculate_amount_stats(data['amount'])
        else:
            self.amount_stats = self._cpu_calculate_amount_stats(data['amount'])
        
        # Step 4: GPU-accelerated feature creation
        print("   ⚡ Creating GPU-accelerated features...")
        
        if self.gpu_available:
            data = self._gpu_create_amount_features(data)
        else:
            data = self._cpu_create_amount_features(data)
        
        # Step 5: Time features (optimized)
        data = self._create_time_features(data)
        
        # Step 6: Transaction type encoding
        data = self._encode_transaction_types(data)
        
        # Step 7: GPU-accelerated interaction features
        if self.gpu_available:
            data = self._gpu_create_interaction_features(data)
        else:
            data = self._cpu_create_interaction_features(data)
        
        # Step 8: Risk scoring features
        data = self._create_risk_features(data)
        
        # Step 9: Prepare final dataset
        y = data['isFraud'].copy()
        X = data.drop(['isFraud', 'type'], axis=1)
        
        # Ensure all features are float32 for GPU efficiency
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].astype('float32')
            elif X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str)).astype('float32')
        
        self.feature_names = list(X.columns)
        
        print(f"\n✅ GPU-ACCELERATED FEATURE ENGINEERING COMPLETED:")
        print(f"   📊 Features created: {len(self.feature_names)}")
        print(f"   📊 Final dataset shape: {X.shape}")
        print(f"   🎯 Target distribution: {y.sum():,} fraud / {len(y):,} total ({y.mean()*100:.3f}%)")
        print(f"   🚀 GPU acceleration used: {self.gpu_available}")
        
        # Memory cleanup
        del data, self.df
        gc.collect()
        if self.gpu_available:
            cp.get_default_memory_pool().free_all_blocks()
        
        return X, y
    
    def _gpu_calculate_amount_stats(self, amount_series):
        """GPU-accelerated amount statistics calculation"""
        print("      🚀 Using GPU for amount statistics...")
        
        # Convert to GPU array
        amount_gpu = cp.array(amount_series.values, dtype=cp.float32)
        
        # Calculate percentiles and stats on GPU
        stats = {
            'p1': float(cp.percentile(amount_gpu, 1)),
            'p5': float(cp.percentile(amount_gpu, 5)),
            'p25': float(cp.percentile(amount_gpu, 25)),
            'p75': float(cp.percentile(amount_gpu, 75)),
            'p95': float(cp.percentile(amount_gpu, 95)),
            'p99': float(cp.percentile(amount_gpu, 99)),
            'mean': float(cp.mean(amount_gpu)),
            'median': float(cp.median(amount_gpu)),
            'std': float(cp.std(amount_gpu))
        }
        
        # Clear GPU memory
        del amount_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return stats
    
    def _cpu_calculate_amount_stats(self, amount_series):
        """CPU fallback for amount statistics"""
        print("      💻 Using CPU for amount statistics...")
        return {
            'p1': amount_series.quantile(0.01),
            'p5': amount_series.quantile(0.05),
            'p25': amount_series.quantile(0.25),
            'p75': amount_series.quantile(0.75),
            'p95': amount_series.quantile(0.95),
            'p99': amount_series.quantile(0.99),
            'mean': amount_series.mean(),
            'median': amount_series.median(),
            'std': amount_series.std()
        }
    
    def _gpu_create_amount_features(self, data):
        """GPU-accelerated amount feature creation"""
        print("      🚀 GPU-accelerated amount features...")
        
        # Convert amount to GPU
        amount_gpu = cp.array(data['amount'].values, dtype=cp.float32)
        
        # GPU computations
        amount_log_gpu = cp.log1p(amount_gpu)
        amount_zscore_gpu = (amount_gpu - self.amount_stats['mean']) / self.amount_stats['std']
        is_round_gpu = (amount_gpu % 1000 == 0).astype(cp.float32)
        is_extreme_gpu = ((amount_gpu >= self.amount_stats['p99']) | 
                         (amount_gpu <= self.amount_stats['p1'])).astype(cp.float32)
        
        # Convert back to CPU and assign
        data['amount_log'] = cp.asnumpy(amount_log_gpu)
        data['amount_zscore'] = cp.asnumpy(amount_zscore_gpu)
        data['is_round_amount'] = cp.asnumpy(is_round_gpu)
        data['is_extreme_amount'] = cp.asnumpy(is_extreme_gpu)
        
        # Amount categories (done on CPU for cut function)
        data['amount_category'] = pd.cut(data['amount'], 
                                       bins=[0, 1000, 10000, 100000, 1000000, np.inf],
                                       labels=[0, 1, 2, 3, 4]).astype('float32')
        
        # Clear GPU memory
        del amount_gpu, amount_log_gpu, amount_zscore_gpu, is_round_gpu, is_extreme_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return data
    
    def _cpu_create_amount_features(self, data):
        """CPU fallback for amount features"""
        print("      💻 CPU amount features...")
        
        data['amount_log'] = np.log1p(data['amount'])
        data['amount_zscore'] = (data['amount'] - self.amount_stats['mean']) / self.amount_stats['std']
        data['is_round_amount'] = (data['amount'] % 1000 == 0).astype(int)
        data['is_extreme_amount'] = ((data['amount'] >= self.amount_stats['p99']) | 
                                   (data['amount'] <= self.amount_stats['p1'])).astype(int)
        
        data['amount_category'] = pd.cut(data['amount'], 
                                       bins=[0, 1000, 10000, 100000, 1000000, np.inf],
                                       labels=[0, 1, 2, 3, 4]).astype('float32')
        return data
    
    def _create_time_features(self, data):
        """Time-based features (vectorized for speed)"""
        print("      ⏰ Creating temporal features...")
        
        # Vectorized operations
        data['hour'] = (data['step'] % 24).astype('float32')
        data['day'] = ((data['step'] // 24) + 1).astype('float32')
        data['is_night'] = ((data['hour'] >= 22) | (data['hour'] <= 6)).astype('float32')
        data['is_weekend'] = (data['day'] % 7 >= 5).astype('float32')
        data['is_deep_night'] = ((data['hour'] >= 1) & (data['hour'] <= 4)).astype('float32')
        
        return data
    
    def _encode_transaction_types(self, data):
        """Encode transaction types efficiently"""
        print("      📝 Encoding transaction types...")
        
        le = LabelEncoder()
        data['type_encoded'] = le.fit_transform(data['type']).astype('float32')
        data['is_cash_out'] = (data['type'] == 'CASH_OUT').astype('float32')
        
        return data
    
    def _gpu_create_interaction_features(self, data):
        """GPU-accelerated interaction features"""
        print("      🚀 GPU-accelerated interaction features...")
        
        # Convert relevant columns to GPU
        amount_log_gpu = cp.array(data['amount_log'].values, dtype=cp.float32)
        hour_gpu = cp.array(data['hour'].values, dtype=cp.float32)
        is_round_gpu = cp.array(data['is_round_amount'].values, dtype=cp.float32)
        is_night_gpu = cp.array(data['is_night'].values, dtype=cp.float32)
        is_extreme_gpu = cp.array(data['is_extreme_amount'].values, dtype=cp.float32)
        is_cash_out_gpu = cp.array(data['is_cash_out'].values, dtype=cp.float32)
        
        # GPU computations
        amount_hour_gpu = amount_log_gpu * hour_gpu
        round_night_gpu = is_round_gpu * is_night_gpu
        extreme_cashout_gpu = is_extreme_gpu * is_cash_out_gpu
        
        # Convert back to CPU
        data['amount_hour_interaction'] = cp.asnumpy(amount_hour_gpu)
        data['round_amount_night'] = cp.asnumpy(round_night_gpu)
        data['extreme_amount_cashout'] = cp.asnumpy(extreme_cashout_gpu)
        
        # Clear GPU memory
        del amount_log_gpu, hour_gpu, is_round_gpu, is_night_gpu, is_extreme_gpu, is_cash_out_gpu
        del amount_hour_gpu, round_night_gpu, extreme_cashout_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return data
    
    def _cpu_create_interaction_features(self, data):
        """CPU fallback for interaction features"""
        print("      💻 CPU interaction features...")
        
        data['amount_hour_interaction'] = data['amount_log'] * data['hour']
        data['round_amount_night'] = data['is_round_amount'] * data['is_night']
        data['extreme_amount_cashout'] = data['is_extreme_amount'] * data['is_cash_out']
        
        return data
    
    def _create_risk_features(self, data):
        """Risk scoring features (optimized boolean operations)"""
        print("      🎲 Creating risk scoring features...")
        
        # Vectorized boolean operations
        data['high_risk_pattern'] = (
            (data['is_round_amount'] == 1) & 
            (data['amount'] >= 100000) & 
            (data['is_night'] == 1)
        ).astype('float32')
        
        data['suspicious_timing'] = (
            (data['is_deep_night'] == 1) | 
            (data['is_weekend'] == 1)
        ).astype('float32')
        
        return data
    
    def save_checkpoint(self, X, y, output_dir='/kaggle/working'):
        """Save GPU-optimized features"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save in GPU-friendly format
        features_path = f"{output_dir}/02_features_gpu.pkl"
        target_path = f"{output_dir}/02_target_gpu.pkl"
        
        X.to_pickle(features_path)
        y.to_pickle(target_path)
        
        # Enhanced feature metadata
        feature_metadata = {
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'amount_stats': self.amount_stats,
            'data_shape': X.shape,
            'fraud_rate': float(y.mean()),
            'feature_types': {col: str(X[col].dtype) for col in X.columns},
            'gpu_optimized': True,
            'gpu_used_for_creation': self.gpu_available
        }
        
        metadata_path = f"{output_dir}/02_feature_metadata_gpu.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(feature_metadata, f)
        
        print(f"\n💾 GPU-OPTIMIZED CHECKPOINT SAVED:")
        print(f"   Features: {features_path}")
        print(f"   Target: {target_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   🚀 GPU-optimized: {self.gpu_available}")
        
        return features_path, target_path, metadata_path

def main():
    """Main execution for GPU-optimized Step 2"""
    print("🚀 STARTING STEP 2: GPU-ACCELERATED FEATURE ENGINEERING")
    
    engineer = GPUPrecisionFeatureEngineer()
    
    if not engineer.load_checkpoint():
        print("❌ STEP 2 FAILED: Could not load data from Step 1")
        return False
    
    try:
        X, y = engineer.engineer_features()
        engineer.save_checkpoint(X, y)
        
        print("\n✅ STEP 2 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 3: GPU Data Splitting")
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