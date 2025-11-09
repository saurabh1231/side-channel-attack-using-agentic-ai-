"""
Visualization tools for Side-Channel Attack Detection results
"""

import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class SCAVisualizer:
    """Visualization tools for SCA results"""
    
    def __init__(self, db_path: str = "sca_agent_memory.sqlite"):
        self.db_path = db_path
        self.conn = None
        if Path(db_path).exists():
            self.conn = sqlite3.connect(db_path)
    
    def plot_training_history(self, history_file: str = "sca_results.json"):
        """Plot training and validation metrics"""
        if not Path(history_file).exists():
            logger.warning(f"History file {history_file} not found")
            return
        
        with open(history_file, 'r') as f:
            results = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        if 'training' in results:
            axes[0].plot(results['training'].get('accuracy', []), label='Training')
            axes[0].plot(results['training'].get('val_accuracy', []), label='Validation')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Model Accuracy Over Time')
            axes[0].legend()
            axes[0].grid(True)
        
        # Loss plot
        if 'training' in results:
            axes[1].plot(results['training'].get('loss', []), label='Training')
            axes[1].plot(results['training'].get('val_loss', []), label='Validation')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Model Loss Over Time')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('sca_training_history.png', dpi=300, bbox_inches='tight')
        logger.info("Training history plot saved to 'sca_training_history.png'")
        plt.close()
    
    def plot_threat_distribution(self):
        """Plot distribution of threat levels"""
        if not self.conn:
            logger.warning("Database not available")
            return
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT threat_level, COUNT(*) as count
            FROM security_alerts
            GROUP BY threat_level
        """)
        
        data = cursor.fetchall()
        if not data:
            logger.warning("No alert data available")
            return
        
        threat_levels = [row[0] for row in data]
        counts = [row[1] for row in data]
        
        # Color mapping
        colors = {
            'safe': 'green',
            'low': 'yellow',
            'medium': 'orange',
            'high': 'red',
            'critical': 'darkred'
        }
        bar_colors = [colors.get(level, 'gray') for level in threat_levels]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(threat_levels, counts, color=bar_colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Threat Level', fontsize=12)
        plt.ylabel('Number of Alerts', fontsize=12)
        plt.title('Security Alert Distribution by Threat Level', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('sca_threat_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Threat distribution plot saved to 'sca_threat_distribution.png'")
        plt.close()
    
    def plot_confidence_distribution(self):
        """Plot distribution of prediction confidence"""
        if not self.conn:
            logger.warning("Database not available")
            return
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT confidence FROM security_alerts")
        confidences = [row[0] for row in cursor.fetchall()]
        
        if not confidences:
            logger.warning("No confidence data available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Prediction Confidence Scores', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sca_confidence_distribution.png', dpi=300, bbox_inches='tight')
        logger.info("Confidence distribution plot saved to 'sca_confidence_distribution.png'")
        plt.close()
    
    def plot_alerts_timeline(self):
        """Plot alerts over time"""
        if not self.conn:
            logger.warning("Database not available")
            return
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, threat_level, confidence
            FROM security_alerts
            ORDER BY timestamp
        """)
        
        data = cursor.fetchall()
        if not data:
            logger.warning("No timeline data available")
            return
        
        timestamps = [row[0] for row in data]
        threat_levels = [row[1] for row in data]
        confidences = [row[2] for row in data]
        
        # Normalize timestamps
        start_time = min(timestamps)
        relative_times = [(t - start_time) for t in timestamps]
        
        # Map threat levels to numeric values
        threat_map = {'safe': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        threat_values = [threat_map.get(level, 0) for level in threat_levels]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Threat level timeline
        scatter = axes[0].scatter(relative_times, threat_values, 
                                 c=confidences, cmap='YlOrRd', 
                                 s=100, alpha=0.6, edgecolors='black')
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel('Threat Level', fontsize=12)
        axes[0].set_yticks(range(5))
        axes[0].set_yticklabels(['Safe', 'Low', 'Medium', 'High', 'Critical'])
        axes[0].set_title('Security Alerts Timeline', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Confidence')
        
        # Confidence timeline
        axes[1].plot(relative_times, confidences, 'o-', color='steelblue', 
                    markersize=4, alpha=0.6)
        axes[1].set_xlabel('Time (seconds)', fontsize=12)
        axes[1].set_ylabel('Confidence Score', fontsize=12)
        axes[1].set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('sca_alerts_timeline.png', dpi=300, bbox_inches='tight')
        logger.info("Alerts timeline plot saved to 'sca_alerts_timeline.png'")
        plt.close()
    
    def plot_key_predictions(self, top_n: int = 20):
        """Plot distribution of predicted key bytes"""
        if not self.conn:
            logger.warning("Database not available")
            return
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT predicted_key, COUNT(*) as count
            FROM security_alerts
            GROUP BY predicted_key
            ORDER BY count DESC
            LIMIT ?
        """, (top_n,))
        
        data = cursor.fetchall()
        if not data:
            logger.warning("No key prediction data available")
            return
        
        keys = [f"0x{row[0]:02X}" for row in data]
        counts = [row[1] for row in data]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(keys, counts, color='coral', alpha=0.7, edgecolor='black')
        plt.xlabel('Predicted Key Byte', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Most Frequently Predicted Key Bytes', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('sca_key_predictions.png', dpi=300, bbox_inches='tight')
        logger.info("Key predictions plot saved to 'sca_key_predictions.png'")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        logger.info("Generating all visualization plots...")
        
        self.plot_training_history()
        self.plot_threat_distribution()
        self.plot_confidence_distribution()
        self.plot_alerts_timeline()
        self.plot_key_predictions()
        
        logger.info("\nAll plots generated successfully!")
        logger.info("Generated files:")
        logger.info("  - sca_training_history.png")
        logger.info("  - sca_threat_distribution.png")
        logger.info("  - sca_confidence_distribution.png")
        logger.info("  - sca_alerts_timeline.png")
        logger.info("  - sca_key_predictions.png")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Generate visualizations"""
    visualizer = SCAVisualizer()
    visualizer.generate_all_plots()
    visualizer.close()


if __name__ == "__main__":
    main()
