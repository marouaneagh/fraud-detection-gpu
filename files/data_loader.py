# 01_data_loader_gpu.py - GPU-Optimized Data Loading Module - FIXED VERSION
# Target: Load and validate data with robust error handling and proper NaN management

import pandas as pd
import numpy as np
import gc
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try GPU imports but gracefully fallback to CPU
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ CuPy not available - using CPU optimizations")

class GPUFraudDataLoader:
    """Robust data loading with comprehensive validation and NaN handling"""
    
    def __init__(self, data_path='/kaggle/input/my-dataset/fraud_data.csv'):
        self.data_path = data_path
        self.df = None
        self.metadata = {}
        self.gpu_available = GPU_AVAILABLE
        
    def _check_gpu(self):
        """Enhanced GPU availability check"""
        if not self.gpu_available:
            return False
        try:
            cp.cuda.runtime.getDeviceCount()
            # Test basic GPU operation
            test_array = cp.array([1, 2, 3])
            _ = cp.mean(test_array)
            del test_array
            return True
        except Exception as e:
            print(f"⚠️ GPU check failed: {e}")
            return False
        
    def load_and_validate(self):
        """ROBUST data loading with comprehensive validation"""
        print("📄 STEP 1: ROBUST DATA LOADING & VALIDATION")
        print("=" * 50)
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            print(f"❌ ERROR: File not found at {self.data_path}")
            print("💡 Available paths to check:")
            potential_paths = [
                '/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv',
                '/kaggle/input/fraud-detection-dataset/fraudTrain.csv',
                '/kaggle/input/my-dataset/fraud_data.csv',
                '/kaggle/input/*/fraud*.csv'
            ]
            for path in potential_paths:
                if '*' in path:
                    print(f"   📂 Search pattern: {path}")
                elif os.path.exists(path):
                    print(f"   ✅ Found: {path}")
                    self.data_path = path
                    break
            else:
                return False
            
        try:
            print(f"📂 Loading data from: {self.data_path}")
            print("🚀 Using robust loading strategy...")
            
            # ROBUST dtype mapping - handle any dataset structure
            dtype_map = {
                'step': 'int32',
                'type': 'category', 
                'amount': 'float32',  # Use float32 for memory efficiency
                'nameOrig': 'string',
                'oldbalanceOrg': 'float32',
                'newbalanceOrig': 'float32',
                'nameDest': 'string',
                'oldbalanceDest': 'float32', 
                'newbalanceDest': 'float32',
                'isFraud': 'int8',
                'isFlaggedFraud': 'int8'
            }
            
            # Load with error handling
            chunk_size = 500000
            chunks = []
            
            print(f"   📊 Loading in chunks of {chunk_size:,} rows...")
            
            try:
                # Try with specified dtypes first
                chunk_iter = pd.read_csv(
                    self.data_path,
                    chunksize=chunk_size,
                    dtype=dtype_map,
                    na_values=['', 'null', 'NULL', 'None', 'none', 'NaN', 'nan']
                )
                
                for i, chunk in enumerate(chunk_iter):
                    print(f"   📦 Processing chunk {i+1}...")
                    chunks.append(chunk)
                    if i >= 13:  # Limit to reasonable size
                        print("   ⚠️ Large dataset - truncating to first 5M rows")
                        break
                        
            except (ValueError, TypeError) as e:
                print(f"   ⚠️ Dtype specification failed: {e}")
                print("   🔄 Retrying with automatic dtype inference...")
                
                # Fallback: load without dtype specification
                chunk_iter = pd.read_csv(
                    self.data_path,
                    chunksize=chunk_size,
                    na_values=['', 'null', 'NULL', 'None', 'none', 'NaN', 'nan']
                )
                
                for i, chunk in enumerate(chunk_iter):
                    chunks.append(chunk)
                    if i >= 13:
                        break
            
            # Combine chunks
            print("   🔗 Combining data chunks...")
            self.df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            
            print(f"   ✅ Data loaded successfully!")
            print(f"   📊 Shape: {self.df.shape}")
            print(f"   💾 Memory usage: {self.df.memory_usage().sum() / 1024**2:.1f} MB")
            
            # CRITICAL: Comprehensive data validation and cleaning
            self._comprehensive_validation()
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading data: {e}")
            return False
    
    def _comprehensive_validation(self):
        """COMPREHENSIVE data validation and cleaning"""
        print("\n🔍 COMPREHENSIVE DATA VALIDATION & CLEANING...")
        
        # 1. Check for completely empty DataFrame
        if self.df.empty:
            raise ValueError("Dataset is empty!")
        
        # 2. Identify and handle missing values
        missing_summary = self.df.isnull().sum()
        missing_pct = (missing_summary / len(self.df)) * 100
        
        print("   🔍 Missing value analysis:")
        for col in missing_summary.index:
            if missing_summary[col] > 0:
                print(f"      {col}: {missing_summary[col]:,} ({missing_pct[col]:.1f}%)")
        
        # 3. Remove columns with >90% missing values
        high_missing_cols = missing_pct[missing_pct > 90].index.tolist()
        if high_missing_cols:
            print(f"   🗑️ Removing high-missing columns: {high_missing_cols}")
            self.df = self.df.drop(columns=high_missing_cols)
        
        # 4. Handle remaining missing values intelligently
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                    # For numerical: fill with 0 (common in financial data for balances)
                    self.df[col].fillna(0, inplace=True)
                    print(f"      ✅ Filled {col} numerical NaNs with 0")
                else:
                    # For categorical: fill with 'UNKNOWN'
                    self.df[col].fillna('UNKNOWN', inplace=True)
                    print(f"      ✅ Filled {col} categorical NaNs with 'UNKNOWN'")
        
        # 5. Validate fraud target column
        fraud_cols = ['isFraud', 'Class', 'fraud', 'is_fraud']
        target_col = None
        
        for col in fraud_cols:
            if col in self.df.columns:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("No fraud target column found! Expected: isFraud, Class, fraud, or is_fraud")
        
        # Standardize target column name
        if target_col != 'isFraud':
            self.df['isFraud'] = self.df[target_col]
            if target_col != 'isFraud':
                self.df = self.df.drop(columns=[target_col])
            print(f"   ✅ Standardized target column: {target_col} → isFraud")
        
        # 6. Validate fraud distribution
        fraud_count = self.df['isFraud'].sum()
        fraud_rate = fraud_count / len(self.df) * 100
        
        print(f"   📊 Fraud distribution: {fraud_count:,} fraud / {len(self.df):,} total ({fraud_rate:.3f}%)")
        
        if fraud_rate < 0.01:
            print("   ⚠️ WARNING: Very low fraud rate (<0.01%) - check data quality")
        elif fraud_rate > 10:
            print("   ⚠️ WARNING: Very high fraud rate (>10%) - unusual for fraud detection")
        else:
            print("   ✅ Fraud rate looks reasonable for fraud detection")
        
        # 7. Basic data type optimization
        print("   🔧 Optimizing data types...")
        
        # Convert boolean-like integers to int8
        bool_cols = ['isFraud', 'isFlaggedFraud']
        for col in bool_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('int8')
        
        # Convert large integers to smaller types where possible
        int_cols = self.df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            if col not in bool_cols:
                col_max = self.df[col].max()
                col_min = self.df[col].min()
                
                if col_min >= 0 and col_max <= 255:
                    self.df[col] = self.df[col].astype('uint8')
                elif col_min >= -128 and col_max <= 127:
                    self.df[col] = self.df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    self.df[col] = self.df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    self.df[col] = self.df[col].astype('int32')
        
        # Convert float64 to float32 for memory efficiency
        float_cols = self.df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        memory_after = self.df.memory_usage().sum() / 1024**2
        print(f"   💾 Memory optimized to: {memory_after:.1f} MB")
        
        # 8. Store comprehensive metadata
        self.metadata = {
            'original_shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': dict(self.df.dtypes),
            'memory_mb': memory_after,
            'fraud_count': int(fraud_count),
            'fraud_rate': float(fraud_rate),
            'missing_handled': True,
            'gpu_available': self.gpu_available,
            'data_quality': 'VALIDATED'
        }
        
        print("   ✅ Comprehensive validation completed!")
    
    def save_checkpoint(self, output_dir='/kaggle/working'):
        """Save validated data and metadata"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save main dataset
        data_path = f"{output_dir}/01_loaded_data_gpu.pkl"
        self.df.to_pickle(data_path)
        
        # Save metadata
        meta_path = f"{output_dir}/01_metadata_gpu.pkl"
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"\n💾 CHECKPOINT SAVED:")
        print(f"   📊 Data: {data_path}")
        print(f"   📋 Metadata: {meta_path}")
        print(f"   💾 Size: {os.path.getsize(data_path) / 1024**2:.1f} MB")
        
        return data_path, meta_path

def main():
    """Main execution for Step 1"""
    print("🚀 STARTING STEP 1: ROBUST DATA LOADING")
    
    loader = GPUFraudDataLoader()
    
    try:
        if not loader.load_and_validate():
            print("❌ STEP 1 FAILED: Could not load data")
            return False
        
        loader.save_checkpoint()
        
        print("\n✅ STEP 1 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 2: Feature Engineering")
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)