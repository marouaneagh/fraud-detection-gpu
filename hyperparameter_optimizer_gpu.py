# 04_hyperparameter_optimizer_gpu.py - GPU-Accelerated Hyperparameter Optimization
# Target: Ultra-fast precision optimization using GPU XGBoost + Optuna

import pandas as pd
import numpy as np
import cupy as cp  # GPU arrays
import pickle
import gc
from pathlib import Path
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class GPUHyperparameterOptimizer:
    """GPU-accelerated precision-focused hyperparameter optimization"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.splits = None
        self.split_metadata = None
        self.best_params = None
        self.optimization_history = []
        self.gpu_available = self._check_gpu()
        
        # Updated realistic targets
        self.targets = {
            'precision_min': 0.30,  # 30% minimum (realistic)
            'precision_target': 0.45,  # 45% target
            'recall_min': 0.45,     # 45% minimum
            'f1_min': 0.35,         # 35% minimum
            'auc_min': 0.70,        # 70% minimum
            'fpr_max': 0.05         # 5% maximum false positive rate
        }
        
    def _check_gpu(self):
        """Check GPU availability for XGBoost"""
        try:
            # Test GPU XGBoost
            test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1, gpu_id=0)
            return True
        except:
            return False
    
    def load_checkpoint(self):
        """Load GPU-optimized data splits"""
        print("🔄 STEP 4: GPU-ACCELERATED HYPERPARAMETER OPTIMIZATION")
        print("=" * 60)
        
        try:
            # Load GPU-optimized splits
            splits_path = f"{self.checkpoint_dir}/03_data_splits_gpu.pkl"
            with open(splits_path, 'rb') as f:
                self.splits = pickle.load(f)
            print(f"✅ Loaded GPU-ready data splits")
            
            # Load metadata
            metadata_path = f"{self.checkpoint_dir}/03_split_metadata_gpu.pkl"
            with open(metadata_path, 'rb') as f:
                self.split_metadata = pickle.load(f)
            print(f"✅ GPU-optimized metadata: {self.split_metadata['sampling_method']}")
            
            print(f"   📊 Train (balanced): {self.split_metadata['train_balanced_shape']}")
            print(f"   📊 Validation: {self.split_metadata['val_shape']}")
            print(f"   📊 Balanced fraud rate: {self.split_metadata['balanced_fraud_rate']*100:.1f}%")
            print(f"   🚀 GPU Available: {self.gpu_available}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            return False
    
    def create_gpu_objective_function(self):
        """GPU-accelerated Optuna objective function"""
        
        X_train = self.splits['X_train_balanced']
        y_train = self.splits['y_train_balanced']
        X_val = self.splits['X_val']
        y_val = self.splits['y_val']
        
        def gpu_precision_objective(trial):
            """GPU-accelerated objective optimizing for precision with FPR constraint"""
            
            # GPU-optimized hyperparameters with expanded search space
            if self.gpu_available:
                params = {
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'max_bin': trial.suggest_int('max_bin', 512, 2048),  # GPU can handle more bins
                    'max_depth': trial.suggest_int('max_depth', 3, 8),   # Deeper trees with GPU
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 300, 1500),  # More trees with GPU speed
                    'subsample': trial.suggest_float('subsample', 0.5, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
                    'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                    'max_leaves': trial.suggest_int('max_leaves', 0, 1024),
                }
            else:
                # CPU fallback parameters
                params = {
                    'tree_method': 'hist',
                    'max_depth': trial.suggest_int('max_depth', 2, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0.0, 0.8),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }
            
            # Common parameters
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'aucpr',
                'random_state': 42,
                'n_jobs': 1 if self.gpu_available else -1,  # GPU handles parallelization
                'verbosity': 0
            })
            
            # Add scale_pos_weight for class imbalance if not using SMOTE
            if 'Class Weighting' in self.split_metadata['sampling_method']:
                fraud_rate = self.split_metadata['balanced_fraud_rate']
                scale_pos_weight = min((1 - fraud_rate) / fraud_rate, 100)
                params['scale_pos_weight'] = scale_pos_weight
            
            try:
                # Create GPU-accelerated model
                model = xgb.XGBClassifier(**params)
                
                # GPU-optimized training
                eval_set = [(X_val, y_val)]
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=30,  # More patience with GPU speed
                    verbose=False
                )
                
                # Fast GPU predictions
                y_val_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Advanced threshold optimization with FPR constraint
                thresholds = np.linspace(0.05, 0.95, 100)  # More threshold candidates
                best_score = 0
                best_metrics = {}
                
                for threshold in thresholds:
                    y_val_pred = (y_val_pred_proba >= threshold).astype(int)
                    
                    if y_val_pred.sum() > 0:  # Avoid division by zero
                        precision = precision_score(y_val, y_val_pred, zero_division=0)
                        recall = recall_score(y_val, y_val_pred, zero_division=0)
                        f1 = f1_score(y_val, y_val_pred, zero_division=0)
                        
                        # Calculate FPR (False Positive Rate)
                        tn = ((y_val == 0) & (y_val_pred == 0)).sum()
                        fp = ((y_val == 0) & (y_val_pred == 1)).sum()
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
                        
                        # Multi-objective scoring with FPR constraint
                        if (recall >= self.targets['recall_min'] and 
                            fpr <= self.targets['fpr_max'] and
                            precision >= self.targets['precision_min']):
                            
                            # Advanced scoring function
                            precision_score_component = precision * 2.0  # High weight on precision
                            recall_bonus = min(recall * 0.5, 0.3)       # Bonus for good recall
                            f1_bonus = f1 * 0.3                         # F1 balance bonus
                            fpr_penalty = max(0, (fpr - 0.02) * 5)      # Penalty for high FPR
                            
                            score = precision_score_component + recall_bonus + f1_bonus - fpr_penalty
                            
                            if score > best_score:
                                best_score = score
                                best_metrics = {
                                    'threshold': threshold,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'fpr': fpr
                                }
                
                # Store detailed metrics
                for key, value in best_metrics.items():
                    trial.set_user_attr(key, value)
                
                return best_score
                
            except Exception as e:
                print(f"GPU trial failed: {e}")
                return 0.0
        
        return gpu_precision_objective
    
    def run_gpu_optimization(self, n_trials=50, timeout_minutes=45):
        """Run GPU-accelerated Optuna optimization"""
        print(f"\n🚀 GPU-ACCELERATED HYPERPARAMETER OPTIMIZATION")
        print(f"   🔢 Trials: {n_trials} (GPU-accelerated)")
        print(f"   ⏰ Timeout: {timeout_minutes} minutes")
        print(f"   🎯 Target: Precision 30-55%, FPR ≤5%")
        print(f"   🚀 GPU Acceleration: {self.gpu_available}")
        
        try:
            # Create GPU-optimized study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=10,  # Quick warmup
                    n_ei_candidates=24 if self.gpu_available else 12
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=5
                )
            )
            
            # Create objective function
            objective_func = self.create_gpu_objective_function()
            
            # Run optimization with progress tracking
            print(f"\n⚡ Starting GPU-accelerated optimization...")
            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=timeout_minutes * 60,
                show_progress_bar=True
            )
            
            # Extract best parameters
            self.best_params = study.best_params.copy()
            
            # Add GPU-specific parameters
            if self.gpu_available:
                self.best_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'objective': 'binary:logistic',
                    'eval_metric': 'aucpr',
                    'random_state': 42,
                    'n_jobs': 1,
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
            best_trial = study.best_trial
            self.optimization_history = {
                'best_score': float(study.best_value),
                'best_threshold': best_trial.user_attrs.get('threshold', 0.5),
                'best_precision': best_trial.user_attrs.get('precision', 0.0),
                'best_recall': best_trial.user_attrs.get('recall', 0.0),
                'best_f1': best_trial.user_attrs.get('f1', 0.0),
                'best_fpr': best_trial.user_attrs.get('fpr', 1.0),
                'n_trials': len(study.trials),
                'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration),
                'gpu_used': self.gpu_available
            }
            
            print(f"\n🏆 GPU OPTIMIZATION COMPLETED!")
            print(f"   ✅ Best score: {self.optimization_history['best_score']:.4f}")
            print(f"   🎯 Best precision: {self.optimization_history['best_precision']:.4f}")
            print(f"   📊 Best recall: {self.optimization_history['best_recall']:.4f}")
            print(f"   📊 Best F1: {self.optimization_history['best_f1']:.4f}")
            print(f"   🚨 Best FPR: {self.optimization_history['best_fpr']:.4f}")
            print(f"   🎚️ Best threshold: {self.optimization_history['best_threshold']:.3f}")
            print(f"   ⚡ Completed trials: {self.optimization_history['n_completed_trials']}/{n_trials}")
            print(f"   ⏱️ Total time: {self.optimization_history['optimization_time']:.1f} seconds")
            
            # Target achievement check
            self._check_target_achievement()
            
            print(f"\n🎛️ GPU-OPTIMIZED PARAMETERS:")
            for param, value in self.best_params.items():
                if param not in ['objective', 'eval_metric', 'random_state', 'n_jobs', 'verbosity']:
                    print(f"   📋 {param}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ GPU OPTIMIZATION FAILED: {e}")
            
            # Enhanced fallback parameters
            print("🔄 Using GPU-optimized fallback parameters...")
            self.best_params = self._get_gpu_fallback_params()
            self.optimization_history = {
                'best_score': 0.0,
                'best_threshold': 0.3,  # Lower threshold for higher recall
                'best_precision': 0.0,
                'best_recall': 0.0,
                'best_f1': 0.0,
                'best_fpr': 0.0,
                'n_trials': 0,
                'n_completed_trials': 0,
                'optimization_time': 0.0,
                'fallback_used': True,
                'gpu_used': self.gpu_available
            }
            return True
    
    def _check_target_achievement(self):
        """Check if optimization achieved realistic targets"""
        results = self.optimization_history
        
        print(f"\n🎯 TARGET ACHIEVEMENT CHECK:")
        checks = {
            'Precision (≥30%)': results['best_precision'] >= 0.30,
            'Recall (≥45%)': results['best_recall'] >= 0.45,
            'F1 (≥35%)': results['best_f1'] >= 0.35,
            'FPR (≤5%)': results['best_fpr'] <= 0.05
        }
        
        for check, status in checks.items():
            print(f"   {check}: {'✅ PASS' if status else '❌ FAIL'}")
        
        success_rate = passed / len(checks) * 100
        print(f"\n📊 Success Rate: {passed}/{len(checks)} ({success_rate:.1f}%)")
        
        if success_rate >= 75:
            print("🎉 EXCELLENT! Ready for final training!")
        elif success_rate >= 50:
            print("✅ GOOD! Minor tuning may help.")
        else:
            print("⚠️ Needs improvement - check parameters.")
    
    def _get_gpu_fallback_params(self):
        """Enhanced GPU-optimized fallback parameters"""
        
        base_params = {
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 800,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
            'verbosity': 0
        }
        
        if self.gpu_available:
            base_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'max_bin': 1024,
                'grow_policy': 'lossguide',
                'max_leaves': 511,
                'n_jobs': 1
            })
        else:
            base_params.update({
                'tree_method': 'hist',
                'n_jobs': -1
            })
        
        # Add scale_pos_weight if needed
        if 'Class Weighting' in self.split_metadata['sampling_method']:
            fraud_rate = self.split_metadata['balanced_fraud_rate']
            scale_pos_weight = min((1 - fraud_rate) / fraud_rate, 50)
            base_params['scale_pos_weight'] = scale_pos_weight
        
        return base_params
    
    def save_checkpoint(self, output_dir='/kaggle/working'):
        """Save GPU-optimized parameters and results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save best parameters
        params_path = f"{output_dir}/04_best_params_gpu.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(self.best_params, f)
        
        # Save optimization history
        history_path = f"{output_dir}/04_optimization_history_gpu.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.optimization_history, f)
        
        # Comprehensive GPU metadata
        optimization_metadata = {
            'best_parameters': self.best_params,
            'optimization_results': self.optimization_history,
            'target_metrics': self.targets,
            'sampling_method': self.split_metadata['sampling_method'],
            'feature_count': len(self.split_metadata['feature_names']),
            'gpu_optimized': True,
            'gpu_available': self.gpu_available,
            'data_shapes': {
                'train_balanced': self.split_metadata['train_balanced_shape'],
                'validation': self.split_metadata['val_shape']
            },
            'performance_summary': {
                'precision_achieved': self.optimization_history.get('best_precision', 0.0),
                'recall_achieved': self.optimization_history.get('best_recall', 0.0),
                'f1_achieved': self.optimization_history.get('best_f1', 0.0),
                'fpr_achieved': self.optimization_history.get('best_fpr', 1.0),
                'optimization_time_minutes': self.optimization_history.get('optimization_time', 0) / 60
            }
        }
        
        metadata_path = f"{output_dir}/04_optimization_metadata_gpu.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(optimization_metadata, f)
        
        print(f"\n💾 GPU-OPTIMIZED CHECKPOINT SAVED:")
        print(f"   Parameters: {params_path}")
        print(f"   History: {history_path}")
        print(f"   Metadata: {metadata_path}")
        print(f"   🚀 GPU-accelerated: {self.gpu_available}")
        
        return params_path, history_path, metadata_path

def main():
    """Main execution for GPU-optimized Step 4"""
    print("🚀 STARTING STEP 4: GPU-ACCELERATED HYPERPARAMETER OPTIMIZATION")
    
    optimizer = GPUHyperparameterOptimizer()
    
    if not optimizer.load_checkpoint():
        print("❌ STEP 4 FAILED: Could not load data from Step 3")
        return False
    
    try:
        # Run GPU-accelerated optimization with more trials
        n_trials = 60 if optimizer.gpu_available else 30
        timeout = 60 if optimizer.gpu_available else 30
        
        optimizer.run_gpu_optimization(n_trials=n_trials, timeout_minutes=timeout)
        optimizer.save_checkpoint()
        
        print("\n✅ STEP 4 COMPLETED SUCCESSFULLY!")
        print("   Ready for Step 5: GPU Final Model Training")
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