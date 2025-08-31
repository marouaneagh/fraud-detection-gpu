import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
)
from pathlib import Path
import pickle

class GPUFraudEvaluator:
    """
    Scientific evaluation module integrated into your GPU pipeline
    Replaces your blob plots with individual publication-quality figures
    """
    
    def __init__(self, output_dir='./figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_publication_style()
        
    def _setup_publication_style(self):
        """Publication-quality matplotlib configuration"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def evaluate_and_plot(self, model, X_test, y_test, feature_names=None, threshold=0.5):
        """
        MAIN METHOD: Complete evaluation with individual plots
        Call this from your final_trainer_gpu.py
        """
        print("🔬 SCIENTIFIC EVALUATION WITH INDIVIDUAL PLOTS")
        print("=" * 60)
        
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'auc_pr': average_precision_score(y_test, y_proba)
        }
        
        # Generate individual plots (NOT blob plots)
        plot_paths = {}
        
        print("📊 Generating publication-quality individual plots...")
        
        # 1. ROC Curve
        plot_paths['roc'] = self._plot_roc_curve(y_test, y_proba, metrics['auc_roc'])
        
        # 2. Precision-Recall Curve
        plot_paths['pr'] = self._plot_pr_curve(y_test, y_proba, metrics['auc_pr'])
        
        # 3. Feature Importance
        if feature_names and hasattr(model, 'feature_importances_'):
            plot_paths['importance'] = self._plot_feature_importance(model, feature_names)
        
        # 4. Confusion Matrix
        plot_paths['confusion'] = self._plot_confusion_matrix(y_test, y_pred)
        
        # 5. Threshold Analysis
        plot_paths['threshold'] = self._plot_threshold_analysis(y_test, y_proba)
        
        # 6. Class Distribution
        plot_paths['distribution'] = self._plot_class_distribution(y_test, y_proba, threshold)
        
        # Print summary
        self._print_summary(metrics)
        
        return {
            'metrics': metrics,
            'plots': plot_paths,
            'predictions': {'y_pred': y_pred, 'y_proba': y_proba}
        }
    
    def _plot_roc_curve(self, y_true, y_proba, auc_score):
        """Individual ROC curve plot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        ax.plot(fpr, tpr, color='#2E8B57', linewidth=3, 
                label=f'XGBoost (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='#DC143C', linestyle='--', 
                linewidth=2, alpha=0.8, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Fraud Detection')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC Curve: {output_path}")
        return output_path
    
    def _plot_pr_curve(self, y_true, y_proba, auc_pr):
        """Individual Precision-Recall curve"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
        baseline = y_true.mean()
        
        ax.plot(recall_vals, precision_vals, color='#228B22', linewidth=3,
                label=f'XGBoost (AP = {auc_pr:.3f})')
        ax.axhline(y=baseline, color='#DC143C', linestyle='--', 
                  linewidth=2, alpha=0.8, 
                  label=f'Random (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - Fraud Detection')
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Add fraud rate annotation
        fraud_rate = y_true.mean() * 100
        ax.text(0.05, 0.95, f'Fraud Rate: {fraud_rate:.3f}%',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
               verticalalignment='top')
        
        output_path = self.output_dir / "precision_recall_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ PR Curve: {output_path}")
        return output_path
    
    def _plot_feature_importance(self, model, feature_names, top_n=15):
        """Individual feature importance plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
        bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'], fontsize=10)
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', va='center', ha='left', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        output_path = self.output_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature Importance: {output_path}")
        return output_path
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Individual confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=ax, cbar_kws={'shrink': 0.8},
                   annot_kws={'fontsize': 14})
        
        ax.set_title('Confusion Matrix - Fraud Detection')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_xticklabels(['Legitimate', 'Fraudulent'])
        ax.set_yticklabels(['Legitimate', 'Fraudulent'])
        
        output_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion Matrix: {output_path}")
        return output_path
    
    def _plot_threshold_analysis(self, y_true, y_proba):
        """Individual threshold analysis"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        thresholds = np.linspace(0.1, 0.9, 50)
        precisions, recalls, f1_scores = [], [], []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if np.sum(y_pred) > 0:
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            else:
                precision = recall = f1 = 0
                
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        ax.plot(thresholds, precisions, label='Precision', color='#1f77b4', linewidth=2.5)
        ax.plot(thresholds, recalls, label='Recall', color='#ff7f0e', linewidth=2.5)
        ax.plot(thresholds, f1_scores, label='F1-Score', color='#2ca02c', linewidth=2.5)
        
        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        ax.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.8,
                  label=f'Optimal = {optimal_threshold:.3f}')
        
        ax.set_xlabel('Classification Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Analysis')
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        output_path = self.output_dir / "threshold_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Threshold Analysis: {output_path}")
        return output_path
    
    def _plot_class_distribution(self, y_true, y_proba, threshold):
        """Individual class distribution plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fraud_probs = y_proba[y_true == 1]
        legit_probs = y_proba[y_true == 0]
        
        ax.hist(legit_probs, bins=50, alpha=0.7, label=f'Legitimate (n={len(legit_probs):,})', 
                color='lightblue', density=True, edgecolor='navy', linewidth=0.5)
        ax.hist(fraud_probs, bins=50, alpha=0.7, label=f'Fraudulent (n={len(fraud_probs):,})', 
                color='salmon', density=True, edgecolor='darkred', linewidth=0.5)
        
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold = {threshold:.3f}')
        
        ax.set_xlabel('Predicted Fraud Probability')
        ax.set_ylabel('Density')
        ax.set_title('Class Distribution Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / "class_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Class Distribution: {output_path}")
        return output_path
    
    def _print_summary(self, metrics):
        """Print evaluation summary"""
        print(f"\n📋 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"Precision:  {metrics['precision']:.3f}")
        print(f"Recall:     {metrics['recall']:.3f}")
        print(f"F1-Score:   {metrics['f1_score']:.3f}")
        print(f"AUC-ROC:    {metrics['auc_roc']:.3f}")
        print(f"AUC-PR:     {metrics['auc_pr']:.3f}")
        
        improvement = (metrics['precision'] / 0.1487 - 1) * 100
        print(f"\nImprovement: {improvement:.0f}x better than baseline")