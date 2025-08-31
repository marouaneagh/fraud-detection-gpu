# 04_hyperparameter_optimizer_gpu.py - FIXED: Realistic Hyperparameter Optimization
# Target: Achieve realistic 30-55% precision with proper validation

import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# XGBoost with proper error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("❌ XGBoost not available!")
    XGBOOST_AVAILABLE = False

# Optuna with error handling
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not available - using grid search fallback")
    OPTUNA_AVAILABLE = False

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class GPUHyperparameterOptimizer:
    """REALISTIC hyperparameter optimization targeting 30-55% precision"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.splits = None
        self.split_metadata = None
        self.best_params = None
        self.optimization_history = []
        
        # 🔧 FIX: Set gpu_available BEFORE calling _check_gpu()
        self.gpu_available = GPU_AVAILABLE  # Set first
        if self.gpu_available:
            self.gpu_available = self._check_gpu()  # Then check
        
        # REALISTIC targets for fraud detection
        self.targets = {
            'precision_min': 0.30,      # 30% minimum (realistic)
            'precision_ideal': 0.45,    # 45% ideal target
            'precision_max': 0.60,      # 60% excellent
            'recall_min': 0.40,         # 40% minimum
            'recall_ideal': 0.60,       # 60% ideal
            'f1_min': 0.30,             # 30% minimum
            'auc_min': 0.70,            # 70% minimum
            'fpr_max': 0.05             # 5% maximum false positive rate
        }
        
    def _check_gpu(self):
        """Check GPU availability for XGBoost"""
        # 🔧 FIX: Don't reference self.gpu_available here
        try:
            # Test GPU XGBoost
            test_model = xgb.XGBClassifier(
                tree_method='gpu_hist', 
                n_estimators=10, 
                gpu_id=0,
                verbosity=0
            )
            # Test with dummy data
            X_test = np.random.random((100, 5)).astype(np.float32)
            y_test = np.random.randint(0, 2, 100)
            test_model.fit(X_test, y_test)
            return True
        except Exception as e:
            print(f"⚠️ GPU XGBoost test failed: {e}")
            return False
        
    def load_checkpoint(self):
        """Load data splits from Step 3"""
        print("📄 STEP 4: REALISTIC HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        
        try:
            # Load data splits
            splits_path = f"{self.checkpoint_dir}/03_data_splits_gpu.pkl"
            with open(splits_path, 'rb') as f:
                self.splits = pickle.load(f)
            print(f"✅ Loaded data splits")
            
            # Load metadata
            metadata_path = f"{self.checkpoint_dir}/03_split_metadata_gpu.pkl"
            with open(metadata_path, 'rb') as f:
                self.split_metadata = pickle.load(f)
            
            sampling_method = self.split_metadata.get('sampling_method', 'Unknown')
            print(f"✅ Sampling method: {sampling_method}")
            
            # Validate splits
            print(f"   📊 Train (balanced): {self.split_metadata['train_balanced_shape']}")
            print(f"   📊 Validation: {self.split_metadata['val_shape']}")
            print(f"   📊 Balanced fraud rate: {self.split_metadata['balanced_fraud_rate']*100:.1f}%")
            print(f"   🚀 GPU Available: {self.gpu_available}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            return False
    
    def create_realistic_objective_function(self):
        """FIXED objective function for extreme imbalance"""
        
        X_train = self.splits['X_train_balanced']
        y_train = self.splits['y_train_balanced']
        X_val = self.splits['X_val']
        y_val = self.splits['y_val']
        
        def realistic_precision_objective(trial):
                """Objective targeting realistic 30-45% precision"""
                
                try:
                    # CRITICAL FIX: Higher scale_pos_weight for 1:336 imbalance
                    if self.gpu_available:
                        params = {
                            'tree_method': 'gpu_hist',
                            'gpu_id': 0,
                            'objective': 'binary:logistic',
                            'eval_metric': 'aucpr',
                            'verbosity': 0,
                            'random_state': 42,
                            
                            # Core hyperparameters
                            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                            'max_depth': trial.suggest_int('max_depth', 4, 10),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 2.0),
                            
                            # CRITICAL FIX: Much higher scale_pos_weight for extreme imbalance
                            'scale_pos_weight': self.scale_pos_weight
                        }
                    else:
                        # CPU fallback parameters
                        params = {
                            'tree_method': 'hist',
                            'objective': 'binary:logistic',
                            'eval_metric': 'aucpr',
                            'verbosity': 0,
                            'random_state': 42,
                            'n_jobs': -1,
                            
                            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                            'max_depth': trial.suggest_int('max_depth', 4, 8),
                            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                            'subsample': trial.suggest_float('subsample', 0.7, 0.9),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),
                            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
                            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
                            
                            # CRITICAL FIX: Higher range for CPU too
                            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 150)
                        }
                    
                    # Train model
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_train, y_train)
                    
                    # CRITICAL FIX: Use default threshold, don't optimize here
                    y_val_pred = model.predict(X_val)  # Uses 0.5 threshold
                    
                    # Calculate metrics with default threshold
                    precision = precision_score(y_val, y_val_pred, zero_division=0)
                    recall = recall_score(y_val, y_val_pred, zero_division=0)
                    f1 = f1_score(y_val, y_val_pred, zero_division=0)
                    
                    # Store metrics
                    trial.set_user_attr('precision', precision)
                    trial.set_user_attr('recall', recall)
                    trial.set_user_attr('f1', f1)
                    trial.set_user_attr('threshold', 0.5)  # Fixed threshold
                    
                    # SIMPLE SCORING: Maximize precision with minimum recall constraint
                    if recall < 0.40:  # Minimum 40% recall required
                        return -1.0
                    
                    # Return precision as objective (we want to maximize it)
                    return precision
                            
                except Exception as e:
                    print(f"Trial failed: {e}")
                    return -1.0
        return realistic_precision_objective
    
    def run_optimization(self, n_trials=50, timeout_minutes=30):
        """Run hyperparameter optimization with realistic expectations"""
        print(f"\n🎯 REALISTIC HYPERPARAMETER OPTIMIZATION")
        print(f"   🎯 Target: 30-55% precision, >40% recall, <5% FPR")
        print(f"   🔢 Trials: {n_trials}")
        print(f"   ⏱️ Timeout: {timeout_minutes} minutes")
        print(f"   🚀 GPU Acceleration: {self.gpu_available}")
        
        if not OPTUNA_AVAILABLE:
            return self._run_grid_search_fallback()
        
        try:
            # Create study with realistic settings
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=min(10, n_trials//5)
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=5
                )
            )
            
            # Create objective function
            objective_func = self.create_realistic_objective_function()
            
            # Run optimization
            print(f"\n⚡ Starting optimization...")
            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout_minutes * 60
            )
            
            # Extract best results
            if len(study.trials) == 0:
                raise ValueError("No trials completed successfully")
            
            best_trial = study.best_trial
            self.best_params = best_trial.params.copy()
            
            # Add fixed parameters based on GPU availability
            if self.gpu_available:
                self.best_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'random_state': 42,
                    'verbosity': 0
                })
            else:
                self.best_params.update({
                    'tree_method': 'hist',
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                })
            
            # Store optimization results
            self.optimization_history = {
                'best_score': float(study.best_value),
                'best_threshold': best_trial.user_attrs.get('threshold', 0.5),
                'best_precision': best_trial.user_attrs.get('precision', 0.0),
                'best_recall': best_trial.user_attrs.get('recall', 0.0),
                'best_f1': best_trial.user_attrs.get('f1', 0.0),
                'best_fpr': best_trial.user_attrs.get('fpr', 1.0),
                'n_trials': len(study.trials),
                'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'optimization_time_minutes': timeout_minutes,
                'gpu_used': self.gpu_available,
                'targets_met': self._check_targets_met(best_trial.user_attrs)
            }
            
            print(f"\n🏆 OPTIMIZATION COMPLETED!")
            print(f"   📊 Best score: {study.best_value:.3f}")
            print(f"   🎯 Best precision: {self.optimization_history['best_precision']*100:.1f}%")
            print(f"   📈 Best recall: {self.optimization_history['best_recall']*100:.1f}%")
            print(f"   📐 Best F1: {self.optimization_history['best_f1']*100:.1f}%")
            print(f"   🚨 Best FPR: {self.optimization_history['best_fpr']*100:.2f}%")
            print(f"   🎛️ Best threshold: {self.optimization_history['best_threshold']:.3f}")
            print(f"   ✅ Trials completed: {self.optimization_history['n_completed_trials']}/{n_trials}")
            
            return True
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return False
    
    def _check_targets_met(self, user_attrs):
        """Check if optimization targets were met"""
        precision = user_attrs.get('precision', 0.0)
        recall = user_attrs.get('recall', 0.0)
        fpr = user_attrs.get('fpr', 1.0)
        
        targets_met = {
            'precision_min': precision >= self.targets['precision_min'],
            'recall_min': recall >= self.targets['recall_min'],
            'fpr_max': fpr <= self.targets['fpr_max'],
            'overall': (precision >= self.targets['precision_min'] and 
                       recall >= self.targets['recall_min'] and 
                       fpr <= self.targets['fpr_max'])
        }
        
        return targets_met
    
    def _run_grid_search_fallback(self):
        """Fallback grid search when Optuna is not available"""
        print("🔍 Running grid search fallback...")
        
        # Define parameter grid
        if self.gpu_available:
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'scale_pos_weight': [5.0, 15.0, 30.0]
            }
            base_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'objective': 'binary:logistic',
                'random_state': 42,
                'verbosity': 0
            }
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'scale_pos_weight': [10.0, 25.0]
            }
            base_params = {
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        
        # Simple grid search
        X_train = self.splits['X_train_balanced']
        y_train = self.splits['y_train_balanced']
        X_val = self.splits['X_val']
        y_val = self.splits['y_val']
        
        best_score = -1
        best_params = None
        best_metrics = {}
        
        print(f"Testing {np.prod([len(v) for v in param_grid.values()])} parameter combinations...")
        
        # Generate all combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for i, param_combo in enumerate(product(*param_values)):
            trial_params = dict(zip(param_names, param_combo))
            trial_params.update(base_params)
            
            try:
                model = xgb.XGBClassifier(**trial_params)
                model.fit(X_train, y_train)
                y_val_proba = model.predict_proba(X_val)[:, 1]
                
                # Find best threshold
                for threshold in np.arange(0.2, 0.8, 0.1):
                    y_val_pred = (y_val_proba >= threshold).astype(int)
                    
                    if y_val_pred.sum() == 0:
                        continue
                    
                    precision = precision_score(y_val, y_val_pred, zero_division=0)
                    recall = recall_score(y_val, y_val_pred, zero_division=0)
                    f1 = f1_score(y_val, y_val_pred, zero_division=0)
                    
                    if (precision >= self.targets['precision_min'] and 
                        recall >= self.targets['recall_min']):
                        
                        score = precision * 0.6 + recall * 0.3 + f1 * 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_params = trial_params.copy()
                            best_metrics = {
                                'threshold': threshold,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1
                            }
                
                print(f"   Trial {i+1}: Score = {best_score:.3f}")
                
            except Exception as e:
                print(f"   Trial {i+1} failed: {e}")
                continue
        
        if best_params is None:
            print("❌ Grid search failed to find good parameters")
            return False
        
        self.best_params = best_params
        self.optimization_history = {
            'best_score': best_score,
            'method': 'grid_search',
            **best_metrics,
            'gpu_used': self.gpu_available
        }
        
        print(f"✅ Grid search completed - Best precision: {best_metrics['precision']*100:.1f}%")
        return True
    
    def save_checkpoint(self, output_dir='/kaggle/working'):
        """Save optimization results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save best parameters
        params_path = f"{output_dir}/04_best_params_gpu.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(self.best_params, f)
        
        # Save optimization history
        history_path = f"{output_dir}/04_optimization_history_gpu.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.optimization_history, f)
        
        print(f"\n💾 STEP 4 CHECKPOINT SAVED:")
        print(f"   🎛️ Best params: {params_path}")
        print(f"   📊 History: {history_path}")
        print(f"   🎯 Targets met: {self.optimization_history.get('targets_met', {}).get('overall', False)}")
        
        return params_path, history_path

def main():
    """Main execution for Step 4"""
    print("🚀 STARTING STEP 4: REALISTIC HYPERPARAMETER OPTIMIZATION")
    
    if not XGBOOST_AVAILABLE:
        print("❌ STEP 4 FAILED: XGBoost not available")
        return False
    
    optimizer = GPUHyperparameterOptimizer()
    
    try:
        if not optimizer.load_checkpoint():
            print("❌ STEP 4 FAILED: Could not load data from Step 3")
            return False
        
        if not optimizer.run_optimization(n_trials=50, timeout_minutes=30):
            print("❌ STEP 4 FAILED: Optimization failed")
            return False
        
        optimizer.save_checkpoint()
        
        print("\n✅ STEP 4 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 5: Final Model Training")
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
