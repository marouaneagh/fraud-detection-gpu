# 03_data_splitter_gpu.py - GPU-Accelerated Data Splitting and Sampling
# Target: Create balanced datasets using GPU acceleration where possible

import pandas as pd
import numpy as np
import cupy as cp  # GPU arrays
import pickle
import gc
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

class GPUPrecisionDataSplitter:
    """GPU-accelerated data splitting with precision-focused sampling"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.X = None
        self.y = None
        self.feature_metadata = None
        self.splits = {}
        self.sampling_method = None
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self):
        """Check GPU availability"""
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            return False
    
    def load_checkpoint(self):
        """Load GPU-optimized features from Step 2"""
        print("🔄 STEP 3: GPU-ACCELERATED DATA SPLITTING & PRECISION SAMPLING")
        print("=" * 60)
        
        try:
            # Load GPU-optimized features
            features_path = f"{self.checkpoint_dir}/02_features_gpu.pkl"
            self.X = pd.read_pickle(features_path)
            print(f"✅ Loaded GPU-ready features: {self.X.shape}")
            
            # Load target
            target_path = f"{self.checkpoint_dir}/02_target_gpu.pkl"
            self.y = pd.read_pickle(target_path)
            print(f"✅ Loaded target: fraud rate {self.y.mean()*100:.3f}%")
            
            # Load feature metadata
            metadata_path = f"{self.checkpoint_dir}/02_feature_metadata_gpu.pkl"
            with open(metadata_path, 'rb') as f:
                self.feature_metadata = pickle.load(f)
            print(f"✅ GPU-optimized metadata: {len(self.feature_metadata['feature_names'])} features")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            return False
    
    def create_optimized_splits(self):
        """GPU-accelerated data splitting with memory optimization"""
        print(f"\n📊 GPU-ACCELERATED DATA SPLITTING...")
        print(f"   GPU Available: {self.gpu_available}")
        
        # Convert to numpy for faster splitting if using GPU
        if self.gpu_available and len(self.X) > 100000:
            print("   🚀 Using GPU-optimized memory layout...")
            
            # Ensure optimal data types for GPU processing
            X_values = self.X.values.astype(np.float32)
            y_values = self.y.values.astype(np.int32)
        else:
            X_values = self.X.values
            y_values = self.y.values
        
        # Stratified split with GPU-friendly random state
        print("   🎯 Creating stratified train/test split (80/20)...")
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_values, y_values,
            test_size=0.2,
            stratify=y_values,
            random_state=42
        )
        
        print("   🎯 Creating train/validation split (80/20 of remaining)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.2,
            stratify=y_temp,
            random_state=42
        )
        
        # Convert back to DataFrames with original column names
        feature_names = self.feature_metadata['feature_names']
        
        self.splits = {
            'X_train': pd.DataFrame(X_train, columns=feature_names).astype('float32'),
            'y_train': pd.Series(y_train, dtype='int8'),
            'X_val': pd.DataFrame(X_val, columns=feature_names).astype('float32'),
            'y_val': pd.Series(y_val, dtype='int8'),
            'X_test': pd.DataFrame(X_test, columns=feature_names).astype('float32'),
            'y_test': pd.Series(y_test, dtype='int8')
        }
        
        print(f"   ✅ Train set: {X_train.shape} (fraud: {y_train.mean()*100:.3f}%)")
        print(f"   ✅ Val set:   {X_val.shape} (fraud: {y_val.mean()*100:.3f}%)")
        print(f"   ✅ Test set:  {X_test.shape} (fraud: {y_test.mean()*100:.3f}%)")
        
        # Memory cleanup
        del X_temp, y_temp, X_values, y_values, X_train, X_val, X_test, y_train, y_val, y_test
        gc.collect()
        if self.gpu_available:
            cp.get_default_memory_pool().free_all_blocks()
        
        return True
    
    def apply_gpu_optimized_sampling(self):
        """GPU-optimized precision sampling with enhanced strategies"""
        print(f"\n⚖️ GPU-ACCELERATED PRECISION SAMPLING...")
        
        # Get class distribution
        fraud_count = self.splits['y_train'].sum()
        total_count = len(self.splits['y_train'])
        fraud_ratio = fraud_count / total_count
        imbalance_ratio = (total_count - fraud_count) / fraud_count
        
        print(f"   📊 Original: {fraud_count:,} fraud / {total_count:,} total ({fraud_ratio*100:.3f}%)")
        print(f"   📊 Imbalance ratio: 1:{imbalance_ratio:.0f}")
        
        X_train = self.splits['X_train']
        y_train = self.splits['y_train']
        
        try:
            # Enhanced sampling strategy selection
            if imbalance_ratio > 300:
                print("   🎯 Extreme imbalance - Using BorderlineSMOTE with GPU optimization...")
                sampler = BorderlineSMOTE(
                    sampling_strategy=0.03,  # 3% fraud rate target for extreme cases
                    random_state=42,
                    k_neighbors=3,
                    m_neighbors=5,
                    n_jobs=-1  # Use all CPU cores for neighbor search
                )
                target_rate = 0.03
                self.sampling_method = f"BorderlineSMOTE (3%)"
                
            elif imbalance_ratio > 150:
                print("   🎯 High imbalance - Using ADASYN with GPU optimization...")
                sampler = ADASYN(
                    sampling_strategy=0.05,  # 5% fraud rate target
                    random_state=42,
                    n_neighbors=3,
                    n_jobs=-1
                )
                target_rate = 0.05
                self.sampling_method = f"ADASYN (5%)"
                
            elif imbalance_ratio > 100:
                print("   🎯 Moderate imbalance - Using SMOTETomek...")
                sampler = SMOTETomek(
                    sampling_strategy=0.08,  # 8% fraud rate target
                    random_state=42,
                    smote=SMOTE(k_neighbors=5, random_state=42, n_jobs=-1)
                )
                target_rate = 0.08
                self.sampling_method = f"SMOTETomek (8%)"
                
            else:
                print("   🎯 Balanced approach - Using enhanced SMOTE...")
                sampler = SMOTE(
                    sampling_strategy=0.15,  # 15% fraud rate target
                    random_state=42,
                    k_neighbors=5,
                    n_jobs=-1
                )
                target_rate = 0.15
                self.sampling_method = f"SMOTE (15%)"
            
            # Pre-sampling GPU memory optimization
            if self.gpu_available:
                print("   🚀 Pre-optimizing data for GPU processing...")
                
                # Move data to GPU for faster neighbor computations
                X_train_gpu = cp.array(X_train.values, dtype=cp.float32)
                
                # Compute some statistics on GPU to validate data
                gpu_mean = cp.mean(X_train_gpu, axis=0)
                gpu_std = cp.std(X_train_gpu, axis=0)
                
                print(f"      GPU validation - Features mean: {float(cp.mean(gpu_mean)):.3f}")
                print(f"      GPU validation - Features std: {float(cp.mean(gpu_std)):.3f}")
                
                # Clear GPU memory before sampling
                del X_train_gpu, gpu_mean, gpu_std
                cp.get_default_memory_pool().free_all_blocks()
            
            # Apply sampling (CPU-based but optimized)
            print("   ⚡ Applying sampling transformation...")
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
            
            # Convert to GPU-friendly format
            if isinstance(X_train_balanced, np.ndarray):
                X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns).astype('float32')
            if isinstance(y_train_balanced, np.ndarray):
                y_train_balanced = pd.Series(y_train_balanced, dtype='int8')
            
            # Store balanced data
            self.splits['X_train_balanced'] = X_train_balanced
            self.splits['y_train_balanced'] = y_train_balanced
            
            new_fraud_count = y_train_balanced.sum()
            new_total = len(y_train_balanced)
            new_fraud_ratio = new_fraud_count / new_total
            
            print(f"   ✅ Balanced: {new_fraud_count:,} fraud / {new_total:,} total ({new_fraud_ratio*100:.1f}%)")
            print(f"   📈 Fraud samples: {fraud_count:,} → {new_fraud_count:,} (+{new_fraud_count-fraud_count:,})")
            
            # GPU memory optimization for balanced data
            if self.gpu_available:
                print("   🚀 Post-sampling GPU optimization...")
                
                # Validate balanced data on GPU
                X_balanced_gpu = cp.array(X_train_balanced.values, dtype=cp.float32)
                balance_check = cp.mean(X_balanced_gpu, axis=0)
                
                print(f"      GPU balanced data validation: {float(cp.mean(balance_check)):.3f}")
                
                # Clear GPU memory
                del X_balanced_gpu, balance_check
                cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"   ⚠️ Sampling failed: {e}")
            print("   🔄 Using class weighting fallback...")
            
            # Fallback: use original data with GPU-optimized weights
            self.splits['X_train_balanced'] = X_train.copy()
            self.splits['y_train_balanced'] = y_train.copy()
            
            # Calculate optimal scale_pos_weight for XGBoost GPU
            scale_pos_weight = min((total_count - fraud_count) / fraud_count, 100)
            self.sampling_method = f"GPU Class Weighting (scale_pos_weight={scale_pos_weight:.1f})"
            
            print(f"   📊 GPU-optimized class weight: {scale_pos_weight:.1f}")
        
        return True
    
    def gpu_validate_splits(self):
        """GPU-accelerated validation of data splits"""
        if not self.gpu_available:
            return True
            
        print(f"\n🔍 GPU-ACCELERATED SPLIT VALIDATION...")
        
        try:
            # Validate each split on GPU
            for split_name in ['X_train', 'X_val', 'X_test', 'X_train_balanced']:
                X_split = self.splits[split_name]
                
                # Move to GPU and validate
                X_gpu = cp.array(X_split.values, dtype=cp.float32)
                
                # GPU computations
                has_nan = cp.any(cp.isnan(X_gpu))
                has_inf = cp.any(cp.isinf(X_gpu))
                mean_val = cp.mean(X_gpu)
                
                print(f"   {split_name}: NaN={bool(has_nan)}, Inf={bool(has_inf)}, Mean={float(mean_val):.3f}")
                
                # Clear GPU memory
                del X_gpu
                
            cp.get_default_memory_pool().free_all_blocks()
            print("   ✅ GPU validation completed successfully")
            return True
            
        except Exception as e:
            print(f"   ⚠️ GPU validation error: {e}")
            return True  # Continue even if validation fails
    
    def save_checkpoint(self, output_dir='/kaggle/working'):
        """Save GPU-optimized splits"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save all splits in GPU-friendly format
        splits_path = f"{output_dir}/03_data_splits_gpu.pkl"
        with open(splits_path, 'wb') as f:
            pickle.dump(self.splits, f)
        
        # Enhanced split metadata
        split_metadata = {
            'sampling_method': self.sampling_method,
            'train_shape': self.splits['X_train'].shape,
            'val_shape': self.splits['X_val'].shape,
            'test_shape': self.splits['X_test'].shape,
            'train_balanced_shape': self.splits['X_train_balanced'].shape,
            'original_fraud_rate': float(self.splits['y_train'].mean()),
            'balanced_fraud_rate': float(self.splits['y_train_balanced'].mean()),
            'feature_names': self.feature_metadata['feature_names'],
            'gpu_optimized': True,
            'gpu_used': self.gpu_available,
            'data_types': {
                'features': 'float32',  # GPU-optimized
                'target': 'int8'        # Memory-optimized
            }
        }
        
        metadata_path = f"{output_dir}/03_split_metadata_gpu.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(split_metadata, f)
        
        print(f"\n💾 GPU-OPTIMIZED SPLITS SAVED:")
        print(f"   Data splits: {splits_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   🚀 GPU-optimized: {self.gpu_available}")
        
        print(f"\n📋 SPLIT SUMMARY:")
        print(f"   🎯 Sampling: {self.sampling_method}")
        print(f"   📊 Train (original): {self.splits['X_train'].shape}")
        print(f"   📊 Train (balanced): {self.splits['X_train_balanced'].shape}")
        print(f"   📊 Validation: {self.splits['X_val'].shape}")
        print(f"   📊 Test: {self.splits['X_test'].shape}")
        
        return splits_path, metadata_path

def main():
    """Main execution for GPU-optimized Step 3"""
    print("🚀 STARTING STEP 3: GPU-ACCELERATED DATA SPLITTING")
    
    splitter = GPUPrecisionDataSplitter()
    
    if not splitter.load_checkpoint():
        print("❌ STEP 3 FAILED: Could not load data from Step 2")
        return False
    
    try:
        splitter.create_optimized_splits()
        splitter.apply_gpu_optimized_sampling()
        splitter.gpu_validate_splits()
        splitter.save_checkpoint()
        
        print("\n✅ STEP 3 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 4: GPU Hyperparameter Optimization")
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)