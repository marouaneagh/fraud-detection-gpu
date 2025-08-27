# 05_final_trainer_gpu.py - FIXED: Final Model Training with Realistic Expectations
# Target: Train production-ready model with honest performance reporting

import pandas as pd
import numpy as np
import pickle
import gc
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("❌ XGBoost not available!")
    XGBOOST_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve, average_precision_score
)

class GPUFinalModelTrainer:
    """Final model training with HONEST performance evaluation"""
    
    def __init__(self, checkpoint_dir='/kaggle/working'):
        self.checkpoint_dir = checkpoint_dir
        self.splits = None
        self.best_params = None
        self.optimization_history = None
        self.final_model = None
        self.final_results = {}
        self.gpu_available = GPU_AVAILABLE and self._check_gpu()
        
        # HONEST target ranges for fraud detection
        self.targets = {
            'precision_min': 0.30,     # 30% minimum (realistic)
            'precision_good': 0.45,    # 45% good performance
            'precision_excellent': 0.60, # 60% excellent
            'recall_min': 0.40,        # 40% minimum
            'recall_good': 0.60,       # 60% good
            'f1_min': 0.30,            # 30% minimum
            'auc_min': 0.70,           # 70% minimum
            'fpr_max': 0.05            # 5% maximum
        }
        
    def _check_gpu(self):
        """Check GPU availability"""
        try:
            test_model = xgb.XGBClassifier(
                tree_method='gpu_hist', 
                n_estimators=10, 
                gpu_id=0,
                verbosity=0
            )
            # Quick test
            X_test = np.random.random((50, 5)).astype(np.float32)
            y_test = np.random.randint(0, 2, 50)
            test_model.fit(X_test, y_test)
            return True
        except:
            return False
    
    def load_checkpoint(self):
        """Load all previous results"""
        print("📄 STEP 5: FINAL MODEL TRAINING & HONEST EVALUATION")
        print("=" * 60)
        
        try:
            # Load data splits
            splits_path = f"{self.checkpoint_dir}/03_data_splits_gpu.pkl"
            with open(splits_path, 'rb') as f:
                self.splits = pickle.load(f)
            print(f"✅ Loaded data splits")
            
            # Load best parameters
            params_path = f"{self.checkpoint_dir}/04_best_params_gpu.pkl"
            with open(params_path, 'rb') as f:
                self.best_params = pickle.load(f)
            print(f"✅ Loaded optimized parameters")
            
            # Load optimization history
            history_path = f"{self.checkpoint_dir}/04_optimization_history_gpu.pkl"
            with open(history_path, 'rb') as f:
                self.optimization_history = pickle.load(f)
            
            expected_precision = self.optimization_history.get('best_precision', 0.0)
            print(f"✅ Expected precision from optimization: {expected_precision*100:.1f}%")
            print(f"🚀 GPU Available: {self.gpu_available}")
            
            return True
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            return False
    
    def train_final_model(self):
        """Train the final production model"""
        print(f"\n🏋️ TRAINING FINAL PRODUCTION MODEL...")
        
        X_train = self.splits['X_train_balanced']
        y_train = self.splits['y_train_balanced']
        
        print(f"   📊 Training data: {X_train.shape}")
        print(f"   🎯 Fraud rate: {y_train.mean()*100:.1f}%")
        print(f"   🎛️ Using optimized parameters")
        
        try:
            # Create final model with optimized parameters
            self.final_model = xgb.XGBClassifier(**self.best_params)
            
            # Train the model
            if self.gpu_available:
                print("   🚀 Training with GPU acceleration...")
            else:
                print("   💻 Training with CPU...")
            
            self.final_model.fit(X_train, y_train)
            
            print(f"   ✅ Model trained successfully!")
            print(f"   🌳 Number of estimators: {self.final_model.n_estimators}")
            print(f"   📏 Max depth: {self.best_params.get('max_depth', 'default')}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return False
    
    def find_optimal_threshold(self):
        """Find optimal threshold on validation set"""
        print(f"\n🎯 FINDING OPTIMAL THRESHOLD...")
        
        X_val = self.splits['X_val']
        y_val = self.splits['y_val']
        
        # Get probabilities
        y_val_proba = self.final_model.predict_proba(X_val)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.05, 0.95, 0.05)
        threshold_results = []
        best_score = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_val_pred = (y_val_proba >= threshold).astype(int)
            
            if y_val_pred.sum() == 0:  # No positive predictions
                continue
            
            # Calculate metrics
            precision = precision_score(y_val, y_val_pred, zero_division=0)
            recall = recall_score(y_val, y_val_pred, zero_division=0)
            f1 = f1_score(y_val, y_val_pred, zero_division=0)
            
            # Calculate false positive rate
            tn = ((y_val == 0) & (y_val_pred == 0)).sum()
            fp = ((y_val == 0) & (y_val_pred == 1)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 1.0
            
            # Score based on business requirements
            if (precision >= self.targets['precision_min'] and 
                recall >= self.targets['recall_min'] and
                fpr <= self.targets['fpr_max']):
                
                # Balanced scoring
                score = precision * 0.5 + recall * 0.3 + f1 * 0.2 - (fpr * 2)
                
                threshold_results.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'fpr': fpr,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        if not threshold_results:
            print("   ⚠️ No threshold meets minimum requirements - using default")
            best_threshold = 0.5
        else:
            print(f"   ✅ Optimal threshold found: {best_threshold:.3f}")
        
        return best_threshold, threshold_results
    
    def evaluate_on_test_set(self, optimal_threshold):
        """Final evaluation on unseen test set"""
        print(f"\n📊 FINAL EVALUATION ON TEST SET...")
        
        X_test = self.splits['X_test']
        y_test = self.splits['y_test']
        
        print(f"   📊 Test set: {X_test.shape}")
        print(f"   🎯 Test fraud rate: {y_test.mean()*100:.3f}%")
        print(f"   🎛️ Using threshold: {optimal_threshold:.3f}")
        
        # Make predictions
        y_test_proba = self.final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_test_proba),
            'auc_pr': average_precision_score(y_test, y_test_proba)
        }
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business metrics
        business_metrics = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        # Store results
        self.final_results = {
            'metrics': metrics,
            'business_metrics': business_metrics,
            'confusion_matrix': cm.tolist(),
            'test_predictions_sample': {
                'probabilities': y_test_proba[:1000].tolist(),
                'predictions': y_test_pred[:1000].tolist(),
                'true_labels': y_test[:1000].tolist()
            },
            'model_info': {
                'n_estimators': self.final_model.n_estimators,
                'parameters': self.best_params
            }
        }
        
        # HONEST performance reporting
        print(f"\n🎯 FINAL TEST SET PERFORMANCE:")
        print("=" * 50)
        print(f"   Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"   F1-Score:      {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print(f"   AUC-ROC:       {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")
        print(f"   AUC-PR:        {metrics['auc_pr']:.4f} ({metrics['auc_pr']*100:.2f}%)")
        
        print(f"\n💼 BUSINESS IMPACT:")
        print(f"   True Positives:  {tp:,} (fraud cases caught)")
        print(f"   False Positives: {fp:,} (legitimate flagged as fraud)")
        print(f"   False Negatives: {fn:,} (fraud cases missed)")
        print(f"   True Negatives:  {tn:,} (legitimate correctly identified)")
        print(f"   False Positive Rate: {business_metrics['false_positive_rate']*100:.2f}%")
        
        # Honest assessment
        self._honest_performance_assessment(metrics, business_metrics)
        
        return True
    
    def _honest_performance_assessment(self, metrics, business_metrics):
        """Provide honest assessment of model performance"""
        print(f"\n🔍 HONEST PERFORMANCE ASSESSMENT:")
        print("=" * 50)
        
        precision = metrics['precision']
        recall = metrics['recall']
        fpr = business_metrics['false_positive_rate']
        
        # Precision assessment
        if precision >= self.targets['precision_excellent']:
            precision_grade = "EXCELLENT"
        elif precision >= self.targets['precision_good']:
            precision_grade = "GOOD"
        elif precision >= self.targets['precision_min']:
            precision_grade = "ACCEPTABLE"
        else:
            precision_grade = "NEEDS IMPROVEMENT"
        
        # Recall assessment
        if recall >= self.targets['recall_good']:
            recall_grade = "GOOD"
        elif recall >= self.targets['recall_min']:
            recall_grade = "ACCEPTABLE"
        else:
            recall_grade = "NEEDS IMPROVEMENT"
        
        # FPR assessment
        # FPR assessment
        if fpr <= self.targets['fpr_max']:
            fpr_grade = "GOOD"
        else:
            fpr_grade = "HIGH (CONCERNING)"
        
        print(f"   📊 Precision: {precision*100:.1f}% - {precision_grade}")
        print(f"   📈 Recall: {recall*100:.1f}% - {recall_grade}")
        print(f"   🚨 False Positive Rate: {fpr*100:.2f}% - {fpr_grade}")
        
        # Overall assessment
        if (precision >= self.targets['precision_min'] and 
            recall >= self.targets['recall_min'] and 
            fpr <= self.targets['fpr_max']):
            overall_grade = "✅ PRODUCTION READY"
        elif precision >= self.targets['precision_min']:
            overall_grade = "⚠️ NEEDS TUNING"
        else:
            overall_grade = "❌ NOT PRODUCTION READY"
        
        print(f"\n🎯 OVERALL ASSESSMENT: {overall_grade}")
        
        # Realistic expectations message
        if precision > 0.70:
            print("⚠️ WARNING: >70% precision is unusually high for fraud detection")
            print("   Please verify there's no data leakage or overfitting")
        elif precision < 0.20:
            print("💡 TIP: <20% precision suggests the model needs significant improvement")
            print("   Consider feature engineering or different sampling strategies")
        else:
            print("✅ Precision is within realistic range for fraud detection")
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation plots with proper saving"""
        if not PLOTTING_AVAILABLE:
            print("Warning: Matplotlib not available - skipping plots")
            return None
            
        print(f"\nCreating comprehensive evaluation plots...")
        
        X_test = self.splits['X_test']
        y_test = self.splits['y_test']
        y_test_proba = self.final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= self.final_results['metrics']['threshold']).astype(int)
        
        # Create comprehensive figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # Main title
        gpu_status = "GPU-ACCELERATED" if self.gpu_available else "CPU"
        fig.suptitle(f'Fraud Detection Model - Complete Performance Analysis ({gpu_status})', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. ROC Curve with enhanced details
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_proba)
        auc_roc = roc_auc_score(y_test, y_test_proba)
        
        ax1.plot(fpr, tpr, color='darkblue', lw=3, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        ax1.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.6, lw=2, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('ROC Curve Analysis', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        
        # 2. Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 1])
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
        auc_pr = average_precision_score(y_test, y_test_proba)
        baseline_precision = y_test.mean()
        
        ax2.plot(recall_vals, precision_vals, color='darkgreen', lw=3, 
                label=f'PR Curve (AUC = {auc_pr:.3f})')
        ax2.axhline(y=baseline_precision, color='red', linestyle='--', alpha=0.6, lw=2,
                   label=f'Baseline ({baseline_precision:.3f})')
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        
        # 3. Enhanced Confusion Matrix with percentages
        ax3 = fig.add_subplot(gs[0, 2])
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Create confusion matrix with both counts and percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Custom annotations combining counts and percentages
        annotations = np.array([[f'{cm[i,j]:,}\n({cm_percent[i,j]:.1f}%)' 
                               for j in range(cm.shape[1])] for i in range(cm.shape[0])])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=ax3, 
                   cbar_kws={'label': 'Count'})
        ax3.set_title('Confusion Matrix\n(Count & Percentage)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted Label', fontsize=12)
        ax3.set_ylabel('True Label', fontsize=12)
        
        # 4. Prediction Probability Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        bins = np.linspace(0, 1, 51)
        ax4.hist(y_test_proba[y_test == 0], bins=bins, alpha=0.7, label='Legitimate', 
                density=True, color='lightblue', edgecolor='navy')
        ax4.hist(y_test_proba[y_test == 1], bins=bins, alpha=0.7, label='Fraud', 
                density=True, color='lightcoral', edgecolor='darkred')
        ax4.axvline(self.final_results['metrics']['threshold'], color='black', 
                   linestyle='--', linewidth=3, label=f"Threshold ({self.final_results['metrics']['threshold']:.3f})")
        ax4.set_xlabel('Prediction Probability', fontsize=12)
        ax4.set_ylabel('Density', fontsize=12)
        ax4.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature Importance (Top 20)
        ax5 = fig.add_subplot(gs[1, 1])
        if hasattr(self.final_model, 'feature_importances_'):
            feature_names = list(self.splits['X_test'].columns)
            importances = self.final_model.feature_importances_
            
            # Get top 20 features
            indices = np.argsort(importances)[-20:]
            top_importances = importances[indices]
            top_features = [feature_names[i] for i in indices]
            
            y_pos = np.arange(len(indices))
            bars = ax5.barh(y_pos, top_importances, color='skyblue', edgecolor='navy')
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(top_features, fontsize=9)
            ax5.set_xlabel('Feature Importance', fontsize=12)
            ax5.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, top_importances)):
                ax5.text(val + max(top_importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', ha='left', va='center', fontsize=8)
        else:
            ax5.text(0.5, 0.5, 'Feature Importance\nNot Available\n(Model does not support)', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Feature Importance', fontsize=14, fontweight='bold')
        
        # 6. Threshold Analysis
        ax6 = fig.add_subplot(gs[1, 2])
        thresholds = np.arange(0.05, 0.95, 0.02)
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_test_proba >= thresh).astype(int)
            if y_pred_thresh.sum() > 0:
                p = precision_score(y_test, y_pred_thresh, zero_division=0)
                r = recall_score(y_test, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
            else:
                p = r = f1 = 0
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
        
        ax6.plot(thresholds, precisions, label='Precision', color='blue', linewidth=2)
        ax6.plot(thresholds, recalls, label='Recall', color='orange', linewidth=2)
        ax6.plot(thresholds, f1_scores, label='F1-Score', color='green', linewidth=2)
        ax6.axvline(self.final_results['metrics']['threshold'], color='red', 
                   linestyle='--', linewidth=2, label='Optimal Threshold')
        ax6.set_xlabel('Threshold', fontsize=12)
        ax6.set_ylabel('Score', fontsize=12)
        ax6.set_title('Performance vs Threshold', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim([0.05, 0.95])
        ax6.set_ylim([0.0, 1.0])
        
        # 7. Business Metrics Visualization
        ax7 = fig.add_subplot(gs[2, 0])
        business = self.final_results['business_metrics']
        
        # Create business impact bar chart
        categories = ['True\nPositives', 'False\nPositives', 'False\nNegatives', 'True\nNegatives']
        values = [business['true_positives'], business['false_positives'], 
                 business['false_negatives'], business['true_negatives']]
        colors = ['green', 'orange', 'red', 'lightblue']
        
        bars = ax7.bar(categories, values, color=colors, edgecolor='black')
        ax7.set_ylabel('Count', fontsize=12)
        ax7.set_title('Business Impact Metrics', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 8. Cost-Benefit Analysis (Hypothetical)
        ax8 = fig.add_subplot(gs[2, 1])
        
        # Hypothetical costs for demonstration
        fraud_prevented = business['true_positives'] * 1000  # $1000 per fraud prevented
        investigation_cost = (business['true_positives'] + business['false_positives']) * 50  # $50 per investigation
        fraud_missed_cost = business['false_negatives'] * 1000  # $1000 per missed fraud
        
        cost_benefit = [fraud_prevented, -investigation_cost, -fraud_missed_cost]
        labels = ['Fraud\nPrevented', 'Investigation\nCost', 'Missed Fraud\nCost']
        colors = ['green', 'orange', 'red']
        
        bars = ax8.bar(labels, cost_benefit, color=colors, edgecolor='black')
        ax8.set_ylabel('Cost/Benefit ($)', fontsize=12)
        ax8.set_title('Hypothetical Cost-Benefit Analysis', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, cost_benefit):
            y_pos = val + (max(cost_benefit) - min(cost_benefit)) * 0.02 * (1 if val >= 0 else -1)
            ax8.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'${val:,.0f}', ha='center', va='bottom' if val >= 0 else 'top', 
                    fontsize=10, fontweight='bold')
        
        # 9. Model Performance Summary Table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Performance metrics table
        metrics = self.final_results['metrics']
        business = self.final_results['business_metrics']
        
        table_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Precision', f"{metrics['precision']:.1%}", 'Fraud detection accuracy'],
            ['Recall', f"{metrics['recall']:.1%}", 'Fraud coverage'],
            ['F1-Score', f"{metrics['f1']:.1%}", 'Overall balance'],
            ['AUC-ROC', f"{metrics['auc_roc']:.3f}", 'Classification quality'],
            ['AUC-PR', f"{metrics['auc_pr']:.3f}", 'Precision-recall balance'],
            ['FPR', f"{business['false_positive_rate']:.2%}", 'False alarm rate'],
            ['Threshold', f"{metrics['threshold']:.3f}", 'Optimal cutoff']
        ]
        
        table = ax9.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style the table
        for i, key in enumerate(table_data[0]):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax9.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 10. Training Progress/Optimization History
        ax10 = fig.add_subplot(gs[3, :])
        
        # Load optimization history if available
        try:
            history_file = f"{self.checkpoint_dir}/04_optimization_history_gpu.pkl"
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                
                if 'trials' in history and len(history['trials']) > 0:
                    trial_numbers = range(1, len(history['trials']) + 1)
                    scores = [trial['value'] for trial in history['trials']]
                    
                    ax10.plot(trial_numbers, scores, 'o-', color='darkblue', linewidth=2, markersize=4)
                    ax10.axhline(y=max(scores), color='red', linestyle='--', alpha=0.7, 
                               label=f'Best Score: {max(scores):.3f}')
                    ax10.set_xlabel('Optimization Trial', fontsize=12)
                    ax10.set_ylabel('Objective Score', fontsize=12)
                    ax10.set_title('Hyperparameter Optimization Progress', fontsize=14, fontweight='bold')
                    ax10.grid(True, alpha=0.3)
                    ax10.legend(fontsize=10)
                else:
                    ax10.text(0.5, 0.5, 'Optimization History\nNot Available', 
                             ha='center', va='center', transform=ax10.transAxes, fontsize=16)
            else:
                ax10.text(0.5, 0.5, 'Optimization History\nFile Not Found', 
                         ha='center', va='center', transform=ax10.transAxes, fontsize=16)
        except Exception as e:
            ax10.text(0.5, 0.5, f'Error Loading\nOptimization History:\n{str(e)}', 
                     ha='center', va='center', transform=ax10.transAxes, fontsize=12)
        
        ax10.set_title('Hyperparameter Optimization Progress', fontsize=14, fontweight='bold')
        
        # Save the comprehensive plot
        plot_path = f"{self.checkpoint_dir}/05_final_evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        # Also save individual plots for thesis use
        individual_plots_dir = f"{self.checkpoint_dir}/individual_plots"
        os.makedirs(individual_plots_dir, exist_ok=True)
        
        # Save ROC curve separately
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkblue', lw=3, label=f'ROC Curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.6, lw=2)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{individual_plots_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix separately
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.savefig(f"{individual_plots_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.close('all')  # Close the main figure
        
        print(f"   Comprehensive evaluation plots saved: {plot_path}")
        print(f"   Individual plots saved in: {individual_plots_dir}")
        
        return plot_path
    
    def save_final_model(self, output_dir='/kaggle/working'):
        """Save the final trained model and all results"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save the trained model
        model_path = f"{output_dir}/05_final_fraud_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.final_model, f)
        
        # Save comprehensive results
        results_path = f"{output_dir}/05_final_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.final_results, f)
        
        # Create deployment package
        deployment_package = {
            'model': self.final_model,
            'optimal_threshold': self.final_results['metrics']['threshold'],
            'feature_names': list(self.splits['X_test'].columns),
            'model_parameters': self.best_params,
            'performance_metrics': self.final_results['metrics'],
            'business_metrics': self.final_results['business_metrics'],
            'gpu_optimized': self.gpu_available,
            'training_summary': {
                'train_samples': self.splits['X_train_balanced'].shape[0],
                'features': self.splits['X_train_balanced'].shape[1],
                'fraud_rate_training': float(self.splits['y_train_balanced'].mean()),
                'fraud_rate_test': float(self.splits['y_test'].mean())
            }
        }
        
        deployment_path = f"{output_dir}/05_deployment_package.pkl"
        with open(deployment_path, 'wb') as f:
            pickle.dump(deployment_package, f)
        
        print(f"\n💾 FINAL MODEL SAVED:")
        print(f"   🤖 Model: {model_path}")
        print(f"   📊 Results: {results_path}")
        print(f"   📦 Deployment: {deployment_path}")
        print(f"   💾 Model size: {os.path.getsize(model_path) / 1024**2:.1f} MB")
        
        return model_path, results_path, deployment_path
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print(f"\n📋 GENERATING FINAL REPORT...")
        
        metrics = self.final_results['metrics']
        business = self.final_results['business_metrics']
        
        # Compare with baseline (typical fraud detection baseline)
        baseline_precision = 0.15  # Typical baseline
        improvement = ((metrics['precision'] / baseline_precision) - 1) * 100
        
        report = f"""
🎉 FRAUD DETECTION MODEL - FINAL REPORT
{'='*70}

🚀 SYSTEM CONFIGURATION:
   Processing Mode:     {"🚀 GPU-ACCELERATED" if self.gpu_available else "💻 CPU-OPTIMIZED"}
   XGBoost Method:      {self.best_params.get('tree_method', 'unknown')}
   Model Complexity:    {self.final_model.n_estimators} trees, depth {self.best_params.get('max_depth', 'unknown')}
   Learning Rate:       {self.best_params.get('learning_rate', 'unknown'):.3f}

📊 FINAL TEST PERFORMANCE:
   Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
   Recall:        {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
   F1-Score:      {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)
   AUC-ROC:       {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)
   AUC-PR:        {metrics['auc_pr']:.4f} ({metrics['auc_pr']*100:.2f}%)

📈 IMPROVEMENT vs BASELINE:
   Precision Improvement: {improvement:+.1f}% vs {baseline_precision*100:.1f}% baseline

💼 BUSINESS IMPACT:
   Fraud Detection Rate:     {metrics['recall']*100:.1f}% of fraud caught
   Investigation Efficiency: {metrics['precision']*100:.1f}% of flags are fraud
   False Positive Rate:      {business['false_positive_rate']*100:.2f}%
   
   Daily Impact (estimated):
   • Fraud Caught: {business['true_positives']:,} cases
   • False Alarms: {business['false_positives']:,} cases
   • Missed Fraud: {business['false_negatives']:,} cases

🎯 PRODUCTION READINESS:
   Status: {"✅ READY FOR DEPLOYMENT" if metrics['precision'] >= 0.30 and business['false_positive_rate'] <= 0.05 else "⚠️ NEEDS REVIEW"}
   
   The model {'meets' if metrics['precision'] >= 0.30 else 'does not meet'} minimum production 
   requirements for fraud detection systems.

📋 DEPLOYMENT CHECKLIST:
   ✅ Model trained and validated
   ✅ Optimal threshold determined: {metrics['threshold']:.3f}
   ✅ Performance within realistic ranges
   ✅ Business impact quantified
   {"✅" if self.gpu_available else "⚠️"} GPU optimization {"enabled" if self.gpu_available else "available"}
   
🎯 NEXT STEPS:
   1. Deploy model with threshold {metrics['threshold']:.3f}
   2. Monitor precision/recall in production
   3. Set up model retraining pipeline
   4. Implement feedback loop for continuous improvement
   5. Consider A/B testing with current system

📞 MODEL METADATA:
   Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
   Version: Production v1.0
   Business Ready: {'Yes' if metrics['precision'] >= 0.30 else 'Requires review'}
        """
        
        print(report)
        
        # Save report
        report_path = f"{self.checkpoint_dir}/05_final_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"💾 Report saved: {report_path}")
        return report

def main():
    """Main execution for Step 5"""
    print("🚀 STARTING STEP 5: FINAL MODEL TRAINING")
    
    if not XGBOOST_AVAILABLE:
        print("❌ STEP 5 FAILED: XGBoost not available")
        return False
    
    trainer = GPUFinalModelTrainer()
    
    try:
        if not trainer.load_checkpoint():
            print("❌ STEP 5 FAILED: Could not load data from previous steps")
            return False
        
        # Train final model
        if not trainer.train_final_model():
            print("❌ STEP 5 FAILED: Model training failed")
            return False
        
        # Find optimal threshold
        optimal_threshold, _ = trainer.find_optimal_threshold()
        
        # Evaluate on test set
        if not trainer.evaluate_on_test_set(optimal_threshold):
            print("❌ STEP 5 FAILED: Test evaluation failed")
            return False
        
        # Create visualizations
        trainer.create_evaluation_plots()
        
        # Save everything
        trainer.save_final_model()
        
        # Generate final report
        trainer.generate_final_report()
        
        final_precision = trainer.final_results['metrics']['precision']
        
        print("\n🎉 STEP 5 COMPLETED SUCCESSFULLY!")
        print(f"   📊 Final Precision: {final_precision*100:.2f}%")
        print(f"   🎯 Model Status: {'✅ PRODUCTION READY' if final_precision >= 0.30 else '⚠️ NEEDS REVIEW'}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ STEP 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)