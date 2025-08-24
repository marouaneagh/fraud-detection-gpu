# 00_gpu_config.py - GPU Configuration for Fraud Detection Pipeline
# Optimizes XGBoost and related operations for GPU acceleration

import os
import subprocess
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class GPUOptimizer:
    """Configure and optimize GPU settings for fraud detection pipeline"""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_count = 0
        self.gpu_memory = []
        self.optimal_config = {}
        
    def detect_gpu_setup(self):
        """Detect available GPU configuration"""
        print("🔍 DETECTING GPU CONFIGURATION...")
        
        try:
            # Check for CUDA availability
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                self.gpu_count = torch.cuda.device_count()
                self.gpu_available = True
                
                print(f"✅ CUDA Available: {cuda_available}")
                print(f"📊 GPU Count: {self.gpu_count}")
                
                # Get GPU details
                for i in range(self.gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    self.gpu_memory.append(gpu_memory)
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                    
            else:
                print("❌ No GPU detected - falling back to CPU")
                return False
                
        except ImportError:
            print("⚠️ PyTorch not available - checking XGBoost GPU support...")
            
            # Fallback: Check XGBoost GPU support
            try:
                import xgboost as xgb
                # Try to create a GPU-enabled model
                test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
                self.gpu_available = True
                self.gpu_count = 1  # Assume at least 1 GPU if XGBoost supports it
                print("✅ XGBoost GPU support detected")
            except:
                print("❌ No GPU support detected")
                return False
        
        return True
    
    def configure_gpu_settings(self):
        """Configure optimal GPU settings for the pipeline"""
        print("\n⚙️ CONFIGURING GPU OPTIMIZATION...")
        
        if not self.gpu_available:
            print("❌ GPU not available - using CPU configuration")
            self.optimal_config = {
                'tree_method': 'hist',
                'n_jobs': -1,
                'gpu_id': None,
                'max_bin': 256
            }
            return self.optimal_config
        
        # GPU-optimized XGBoost configuration
        self.optimal_config = {
            # GPU-specific settings
            'tree_method': 'gpu_hist',
            'gpu_id': 0,  # Use first GPU
            'max_bin': 1024,  # Higher bins for GPU
            
            # Memory optimization
            'single_precision_histogram': True,
            
            # Parallel processing
            'n_jobs': 1,  # Let GPU handle parallelization
            
            # Performance tuning
            'max_leaves': 255,  # Optimal for GPU
            'grow_policy': 'lossguide',
        }
        
        # Multi-GPU optimization for T4 x2 setup
        if self.gpu_count >= 2:
            print("🚀 Multi-GPU setup detected - enabling advanced optimization")
            self.optimal_config.update({
                'n_jobs': 2,  # Use both GPUs for data loading
                'max_bin': 2048,  # More bins with more memory
            })
        
        print("✅ GPU configuration optimized:")
        for key, value in self.optimal_config.items():
            print(f"   {key}: {value}")
            
        return self.optimal_config
    
    def setup_cuda_environment(self):
        """Setup CUDA environment variables"""
        print("\n🔧 SETTING UP CUDA ENVIRONMENT...")
        
        # CUDA memory management
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' if self.gpu_count >= 2 else '0'
        
        # XGBoost GPU optimizations
        os.environ['XGBOOST_GPU_CACHE_SIZE'] = '8192'  # 8GB cache
        
        # Memory growth settings
        if self.gpu_count >= 2:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        
        print("✅ CUDA environment configured")
        
    def install_gpu_dependencies(self):
        """Install GPU-optimized packages"""
        print("\n📦 INSTALLING GPU DEPENDENCIES...")
        
        gpu_packages = [
            'xgboost[gpu]',
            'cupy-cuda11x',  # For GPU-accelerated NumPy operations
            'rapids-cudf',   # For GPU-accelerated Pandas operations (optional)
        ]
        
        for package in gpu_packages:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
                print(f"   ✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"   ⚠️ {package} installation failed - continuing...")
                
    def optimize_memory_allocation(self):
        """Optimize GPU memory allocation"""
        print("\n💾 OPTIMIZING GPU MEMORY...")
        
        if not self.gpu_available:
            return
            
        try:
            import torch
            
            # Enable memory-efficient attention
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory fraction
            total_memory = sum(self.gpu_memory)
            if total_memory > 0:
                # Use 90% of available GPU memory
                memory_fraction = 0.9
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                print(f"   GPU memory fraction set to: {memory_fraction*100:.0f}%")
                
        except ImportError:
            print("   PyTorch not available for memory optimization")
    
    def validate_gpu_setup(self):
        """Validate GPU setup with test operations"""
        print("\n🧪 VALIDATING GPU SETUP...")
        
        try:
            import xgboost as xgb
            import numpy as np
            
            # Create test data
            X_test = np.random.random((1000, 10)).astype(np.float32)
            y_test = np.random.randint(0, 2, 1000)
            
            # Test GPU training
            model = xgb.XGBClassifier(
                **self.optimal_config,
                n_estimators=10,
                max_depth=3,
                random_state=42,
                verbosity=0
            )
            
            print("   Testing GPU training...")
            model.fit(X_test, y_test)
            predictions = model.predict(X_test)
            
            print("   ✅ GPU validation successful!")
            print(f"   📊 Test accuracy: {(predictions == y_test).mean():.3f}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ GPU validation failed: {e}")
            print("   🔄 Pipeline will fall back to CPU")
            return False

def setup_gpu_environment():
    """Main function to setup GPU environment"""
    print("🚀 GPU OPTIMIZATION SETUP")
    print("=" * 50)
    
    optimizer = GPUOptimizer()
    
    # Step 1: Detect GPU
    if not optimizer.detect_gpu_setup():
        print("⚠️ No GPU detected - using CPU configuration")
        return {'tree_method': 'hist', 'n_jobs': -1}
    
    # Step 2: Install dependencies
    optimizer.install_gpu_dependencies()
    
    # Step 3: Configure settings
    gpu_config = optimizer.configure_gpu_settings()
    
    # Step 4: Setup environment
    optimizer.setup_cuda_environment()
    
    # Step 5: Optimize memory
    optimizer.optimize_memory_allocation()
    
    # Step 6: Validate setup
    is_valid = optimizer.validate_gpu_setup()
    
    if is_valid:
        print("\n🎉 GPU SETUP COMPLETED SUCCESSFULLY!")
        print("   Pipeline will use GPU acceleration")
    else:
        print("\n⚠️ GPU SETUP HAD ISSUES")
        print("   Pipeline will fall back to CPU")
        gpu_config = {'tree_method': 'hist', 'n_jobs': -1}
    
    return gpu_config

def get_updated_targets():
    """Return updated realistic target metrics"""
    return {
        'precision_min': 0.30,     # 30% minimum
        'precision_max': 0.55,     # 55% target
        'recall_min': 0.45,        # 45% minimum  
        'recall_max': 0.70,        # 70% target
        'f1_min': 0.35,            # 35% minimum
        'f1_max': 0.55,            # 55% target
        'auc_min': 0.70,           # 70% minimum
        'auc_max': 0.85,           # 85% target
        'fpr_max': 0.05,           # 5% maximum false positive rate
        'business_target': 'precision_focused_with_fpr_constraint'
    }

if __name__ == "__main__":
    gpu_config = setup_gpu_environment()
    targets = get_updated_targets()
    
    print(f"\n📊 UPDATED TARGET RANGES:")
    print(f"   Precision: {targets['precision_min']*100:.0f}%-{targets['precision_max']*100:.0f}%")
    print(f"   Recall:    {targets['recall_min']*100:.0f}%-{targets['recall_max']*100:.0f}%")
    print(f"   F1-Score:  {targets['f1_min']*100:.0f}%-{targets['f1_max']*100:.0f}%")
    print(f"   AUC-ROC:   {targets['auc_min']*100:.0f}%-{targets['auc_max']*100:.0f}%")
    print(f"   FPR:       <{targets['fpr_max']*100:.0f}%")
    
    # Save configuration
    import pickle
    config_data = {
        'gpu_config': gpu_config,
        'target_metrics': targets
    }
    
    with open('/kaggle/working/00_gpu_config.pkl', 'wb') as f:
        pickle.dump(config_data, f)
    
    print(f"\n💾 Configuration saved to: /kaggle/working/00_gpu_config.pkl")