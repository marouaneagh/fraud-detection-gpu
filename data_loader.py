# 01_data_loader_gpu.py - GPU-Optimized Data Loading Module
# Target: Load and validate data with GPU memory optimization

import pandas as pd
import numpy as np
import gc
import os
import pickle
from pathlib import Path
import cupy as cp  # GPU-accelerated NumPy
import warnings
warnings.filterwarnings('ignore')

class GPUFraudDataLoader:
    """GPU-optimized data loading with memory management"""
    
    def __init__(self, data_path='/kaggle/input/my-dataset/fraud_data.csv'):
        self.data_path = data_path
        self.df = None
        self.metadata = {}
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self):
        """Check GPU availability"""
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return True
        except:
            print("⚠️ CuPy not available - using CPU for data loading")
            return False
        
    def load_and_validate(self):
        """GPU-optimized data loading with validation"""
        print("🔄 STEP 1: GPU-OPTIMIZED DATA LOADING & VALIDATION")
        print("=" * 50)
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            print(f"❌ ERROR: File not found at {self.data_path}")
            return False
            
        try:
            print(f"📂 Loading data from: {self.data_path}")
            print("🚀 Using GPU-optimized loading strategy...")
            
            # GPU-optimized dtypes
            dtype_map = {
                'step': 'int32',
                'type': 'category', 
                'amount': 'float32',  # Use float32 for GPU efficiency
                'nameOrig': 'string',
                'oldbalanceOrg': 'float32',
                'newbalanceOrig': 'float32',
                'nameDest': 'string',
                'oldbalanceDest': 'float32', 
                'newbalanceDest': 'float32',
                'isFraud': 'int8',
                'isFlaggedFraud': 'int8'
            }
            
            # Load with larger chunks for GPU efficiency
            chunk_size = 500000  # Larger chunks for GPU processing
            chunks = []
            
            print(f"📊 Loading in GPU-optimized chunks of {chunk_size:,} rows...")
            
            for chunk in pd.read_csv(self.data_path, chunksize=chunk_size, dtype=dtype_map):
                chunks.append(chunk)
                
                # GPU memory check
                if self.gpu_available:
                    gpu_memory_used = cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0]
                    gpu_memory_used_gb = gpu_memory_used / 1024**3
                    if gpu_memory_used_gb > 6:  # 6GB threshold for T4
                        print(f"⚠️ GPU memory usage: {gpu_memory_used_gb:.1f}GB - processing chunk...")
                        break
                
            self.df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            
            # Move numerical data to GPU for validation
            if self.gpu_available:
                self._gpu_validate_data()
            else:
                self._cpu_validate_data()
            
            print(f"✅ Data loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"💾 Memory usage: {self.df.memory_usage().sum() / 1024**2:.1f} MB")
            
            # Generate metadata
            self._generate_metadata()
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading data: {e}")
            return False
    
    def _gpu_validate_data(self):
        """GPU-accelerated data validation"""
        print("🚀 GPU-ACCELERATED VALIDATION:")
        
        # Convert numerical columns to GPU arrays for fast computation
        numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        
        for col in numerical_cols:
            if col in self.df.columns:
                # Move to GPU, compute stats, move back
                gpu_array = cp.array(self.df[col].values)
                
                # Fast GPU computations
                missing_count = cp.isnan(gpu_array).sum()
                mean_val = cp.nanmean(gpu_array)
                
                print(f"   {col}: Missing={int(missing_count)}, Mean={float(mean_val):.2f}")
                
                # Clear GPU memory
                del gpu_array
                cp.cuda.runtime.deviceSynchronize()
        
        # Fraud distribution (keep on CPU)
        fraud_count = self.df['isFraud'].sum()
        fraud_rate = fraud_count / len(self.df) * 100
        
        print(f"   Total transactions: {len(self.df):,}")
        print(f"   Fraudulent: {fraud_count:,} ({fraud_rate:.3f}%)")
        print(f"   Legitimate: {len(self.df) - fraud_count:,}")
        
        if fraud_rate < 0.1 or fraud_rate > 5:
            print(f"⚠️ WARNING: Unusual fraud rate: {fraud_rate:.3f}%")
        else:
            print("✅ Fraud distribution validated on GPU")
    
    def _cpu_validate_data(self):
        """Fallback CPU validation"""
        print("🔍 CPU DATA VALIDATION:")
        
        missing = self.df.isnull().sum()
        print(f"   Missing values: {missing.sum():,}")
        
        fraud_count = self.df['isFraud'].sum()
        fraud_rate = fraud_count / len(self.df) * 100
        
        print(f"   Total transactions: {len(self.df):,}")
        print(f"   Fraudulent: {fraud_count:,} ({fraud_rate:.3f}%)")
        print(f"   Legitimate: {len(self.df) - fraud_count:,}")
        
        type_dist = self.df['type'].value_counts()
        print(f"   Transaction types: {list(type_dist.index)}")
    
    def _generate_metadata(self):
        """Generate metadata with GPU acceleration where possible"""
        
        if self.gpu_available:
            # GPU-accelerated statistics for amount
            amount_gpu = cp.array(self.df['amount'].values)
            amount_stats = {
                'min': float(cp.min(amount_gpu)),
                'max': float(cp.max(amount_gpu)),
                'mean': float(cp.mean(amount_gpu)),
                'median': float(cp.median(amount_gpu))
            }
            del amount_gpu
        else:
            amount_stats = {
                'min': float(self.df['amount'].min()),
                'max': float(self.df['amount'].max()),
                'mean': float(self.df['amount'].mean()),
                'median': float(self.df['amount'].median())
            }
        
        self.metadata = {
            'total_rows': len(self.df),
            'fraud_count': int(self.df['isFraud'].sum()),
            'fraud_rate': float(self.df['isFraud'].mean()),
            'memory_mb': self.df.memory_usage().sum() / 1024**2,
            'transaction_types': list(self.df['type'].unique()),
            'amount_stats': amount_stats,
            'data_quality': {
                'missing_values': int(self.df.isnull().sum().sum()),
                'duplicates': int(self.df.duplicated().sum())
            },
            'gpu_used': self.gpu_available
        }
        
        print(f"\n📋 GPU-OPTIMIZED METADATA GENERATED:")
        for key, value in self.metadata.items():
            if isinstance(value, dict):
                print(f"   {key}: {len(value)} items")
            else:
                print(f"   {key}: {value}")
    
    def save_checkpoint(self, output_dir='/kaggle/working'):
        """Save processed data optimized for GPU pipeline"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save data in GPU-friendly format
        data_path = f"{output_dir}/01_loaded_data_gpu.pkl"
        
        # Convert to optimal dtypes for GPU processing
        gpu_optimized_df = self.df.copy()
        
        # Ensure float32 for GPU efficiency
        float_cols = gpu_optimized_df.select_dtypes(include=['float64']).columns
        gpu_optimized_df[float_cols] = gpu_optimized_df[float_cols].astype('float32')
        
        gpu_optimized_df.to_pickle(data_path)
        
        # Save metadata  
        meta_path = f"{output_dir}/01_metadata_gpu.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        print(f"\n💾 GPU-OPTIMIZED CHECKPOINT SAVED:")
        print(f"   Data: {data_path}")
        print(f"   Metadata: {meta_path}")
        print(f"   Data size: {os.path.getsize(data_path) / 1024**2:.1f} MB")
        print(f"   GPU-ready: {self.gpu_available}")
        
        return data_path, meta_path

def main():
    """Main execution for GPU-optimized Step 1"""
    print("🚀 STARTING STEP 1: GPU-OPTIMIZED DATA LOADING")
    
    loader = GPUFraudDataLoader()
    
    if loader.load_and_validate():
        loader.save_checkpoint()
        print("\n✅ STEP 1 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 2: GPU Feature Engineering")
        return True
    else:
        print("\n❌ STEP 1 FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)