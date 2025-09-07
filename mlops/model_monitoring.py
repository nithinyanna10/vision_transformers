#!/usr/bin/env python3
"""
Model Monitoring and Performance Tracking for Medical Vision Transformer
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ModelMonitor:
    """Comprehensive model monitoring system"""
    
    def __init__(self, monitoring_dir="monitoring"):
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(exist_ok=True)
        
        self.metrics_file = self.monitoring_dir / "metrics.json"
        self.alerts_file = self.monitoring_dir / "alerts.json"
        self.drift_file = self.monitoring_dir / "drift.json"
        
        # Load existing data
        self.metrics_history = self._load_metrics()
        self.alerts = self._load_alerts()
        self.drift_data = self._load_drift_data()
        
        # Monitoring thresholds
        self.thresholds = {
            'accuracy_drop': 0.05,  # 5% accuracy drop
            'latency_increase': 2.0,  # 2x latency increase
            'error_rate': 0.1,  # 10% error rate
            'data_drift': 0.3,  # 30% data drift
            'prediction_drift': 0.2  # 20% prediction drift
        }
    
    def _load_metrics(self):
        """Load metrics history"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_alerts(self):
        """Load alerts history"""
        if self.alerts_file.exists():
            with open(self.alerts_file, 'r') as f:
                return json.load(f)
        return []
    
    def _load_drift_data(self):
        """Load drift data"""
        if self.drift_file.exists():
            with open(self.drift_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metrics(self):
        """Save metrics history"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def _save_alerts(self):
        """Save alerts"""
        with open(self.alerts_file, 'w') as f:
            json.dump(self.alerts, f, indent=2)
    
    def _save_drift_data(self):
        """Save drift data"""
        with open(self.drift_file, 'w') as f:
            json.dump(self.drift_data, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], model_version: str = "latest"):
        """Log model performance metrics"""
        timestamp = datetime.now().isoformat()
        
        metric_entry = {
            'timestamp': timestamp,
            'model_version': model_version,
            'metrics': metrics
        }
        
        self.metrics_history.append(metric_entry)
        self._save_metrics()
        
        # Check for anomalies
        self._check_anomalies(metrics, timestamp)
        
        print(f"📊 Metrics logged: {timestamp}")
    
    def _check_anomalies(self, metrics: Dict[str, float], timestamp: str):
        """Check for performance anomalies"""
        if len(self.metrics_history) < 2:
            return
        
        # Get baseline metrics (previous entry)
        baseline = self.metrics_history[-2]['metrics']
        current = metrics
        
        alerts = []
        
        # Check accuracy drop
        if 'accuracy' in baseline and 'accuracy' in current:
            accuracy_drop = baseline['accuracy'] - current['accuracy']
            if accuracy_drop > self.thresholds['accuracy_drop']:
                alerts.append({
                    'type': 'accuracy_drop',
                    'severity': 'high',
                    'message': f"Accuracy dropped by {accuracy_drop:.3f}",
                    'timestamp': timestamp,
                    'baseline': baseline['accuracy'],
                    'current': current['accuracy']
                })
        
        # Check latency increase
        if 'latency' in baseline and 'latency' in current:
            latency_ratio = current['latency'] / baseline['latency']
            if latency_ratio > self.thresholds['latency_increase']:
                alerts.append({
                    'type': 'latency_increase',
                    'severity': 'medium',
                    'message': f"Latency increased by {latency_ratio:.2f}x",
                    'timestamp': timestamp,
                    'baseline': baseline['latency'],
                    'current': current['latency']
                })
        
        # Check error rate
        if 'error_rate' in current:
            if current['error_rate'] > self.thresholds['error_rate']:
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'high',
                    'message': f"Error rate {current['error_rate']:.3f} exceeds threshold",
                    'timestamp': timestamp,
                    'current': current['error_rate']
                })
        
        # Add alerts
        for alert in alerts:
            self.alerts.append(alert)
            print(f"🚨 Alert: {alert['message']}")
        
        if alerts:
            self._save_alerts()
    
    def detect_data_drift(self, current_data_stats: Dict[str, Any]):
        """Detect data drift in input data"""
        timestamp = datetime.now().isoformat()
        
        # Simulate drift detection (in real implementation, this would use statistical tests)
        drift_score = np.random.random()  # Simulated drift score
        
        drift_entry = {
            'timestamp': timestamp,
            'drift_score': drift_score,
            'data_stats': current_data_stats,
            'drift_detected': drift_score > self.thresholds['data_drift']
        }
        
        self.drift_data.append(drift_entry)
        self._save_drift_data()
        
        if drift_entry['drift_detected']:
            alert = {
                'type': 'data_drift',
                'severity': 'high',
                'message': f"Data drift detected: {drift_score:.3f}",
                'timestamp': timestamp,
                'drift_score': drift_score
            }
            self.alerts.append(alert)
            self._save_alerts()
            print(f"🚨 Data drift detected: {drift_score:.3f}")
        
        return drift_entry
    
    def detect_prediction_drift(self, current_predictions: List[float]):
        """Detect prediction drift"""
        timestamp = datetime.now().isoformat()
        
        # Simulate prediction drift detection
        prediction_drift = np.random.random()  # Simulated drift score
        
        if prediction_drift > self.thresholds['prediction_drift']:
            alert = {
                'type': 'prediction_drift',
                'severity': 'medium',
                'message': f"Prediction drift detected: {prediction_drift:.3f}",
                'timestamp': timestamp,
                'drift_score': prediction_drift
            }
            self.alerts.append(alert)
            self._save_alerts()
            print(f"🚨 Prediction drift detected: {prediction_drift:.3f}")
        
        return prediction_drift
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        print("📊 Generating monitoring report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'performance_trends': {},
            'alerts_summary': {},
            'drift_analysis': {},
            'recommendations': []
        }
        
        # Performance summary
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]['metrics']
            report['summary'] = {
                'latest_accuracy': latest_metrics.get('accuracy', 0),
                'latest_latency': latest_metrics.get('latency', 0),
                'latest_error_rate': latest_metrics.get('error_rate', 0),
                'total_metrics_logged': len(self.metrics_history)
            }
        
        # Performance trends
        if len(self.metrics_history) > 1:
            report['performance_trends'] = self._analyze_trends()
        
        # Alerts summary
        recent_alerts = [a for a in self.alerts if self._is_recent(a['timestamp'], days=7)]
        report['alerts_summary'] = {
            'total_alerts': len(self.alerts),
            'recent_alerts': len(recent_alerts),
            'high_severity': len([a for a in recent_alerts if a['severity'] == 'high']),
            'alert_types': self._count_alert_types(recent_alerts)
        }
        
        # Drift analysis
        if self.drift_data:
            recent_drift = [d for d in self.drift_data if self._is_recent(d['timestamp'], days=7)]
            report['drift_analysis'] = {
                'total_drift_checks': len(self.drift_data),
                'recent_drift_checks': len(recent_drift),
                'drift_incidents': len([d for d in recent_drift if d['drift_detected']]),
                'avg_drift_score': np.mean([d['drift_score'] for d in recent_drift]) if recent_drift else 0
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends"""
        if len(self.metrics_history) < 2:
            return {}
        
        trends = {}
        
        # Analyze accuracy trend
        accuracies = [m['metrics'].get('accuracy', 0) for m in self.metrics_history]
        if len(accuracies) > 1:
            trends['accuracy_trend'] = {
                'direction': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
                'change': accuracies[-1] - accuracies[0],
                'volatility': np.std(accuracies)
            }
        
        # Analyze latency trend
        latencies = [m['metrics'].get('latency', 0) for m in self.metrics_history]
        if len(latencies) > 1:
            trends['latency_trend'] = {
                'direction': 'improving' if latencies[-1] < latencies[0] else 'declining',
                'change': latencies[-1] - latencies[0],
                'volatility': np.std(latencies)
            }
        
        return trends
    
    def _count_alert_types(self, alerts: List[Dict]) -> Dict[str, int]:
        """Count alert types"""
        alert_types = {}
        for alert in alerts:
            alert_type = alert['type']
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        return alert_types
    
    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Check if timestamp is within recent days"""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt > datetime.now() - timedelta(days=days)
        except:
            return False
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on monitoring data"""
        recommendations = []
        
        # Check for performance issues
        if report['summary'].get('latest_accuracy', 1) < 0.8:
            recommendations.append("Model accuracy is below 80%. Consider retraining or hyperparameter tuning.")
        
        if report['summary'].get('latest_error_rate', 0) > 0.05:
            recommendations.append("Error rate is above 5%. Investigate model performance issues.")
        
        # Check for drift issues
        if report['drift_analysis'].get('drift_incidents', 0) > 0:
            recommendations.append("Data drift detected. Consider retraining with recent data.")
        
        # Check for alert frequency
        if report['alerts_summary'].get('recent_alerts', 0) > 5:
            recommendations.append("High number of recent alerts. Review model stability.")
        
        if not recommendations:
            recommendations.append("Model performance is stable. Continue monitoring.")
        
        return recommendations
    
    def plot_monitoring_dashboard(self):
        """Create monitoring dashboard visualization"""
        print("📊 Creating monitoring dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Performance metrics over time
        if self.metrics_history:
            timestamps = [m['timestamp'] for m in self.metrics_history]
            accuracies = [m['metrics'].get('accuracy', 0) for m in self.metrics_history]
            latencies = [m['metrics'].get('latency', 0) for m in self.metrics_history]
            
            # Convert timestamps to datetime for plotting
            dt_timestamps = [datetime.fromisoformat(ts) for ts in timestamps]
            
            # Accuracy trend
            axes[0, 0].plot(dt_timestamps, accuracies, 'b-', linewidth=2, marker='o')
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Latency trend
            axes[0, 1].plot(dt_timestamps, latencies, 'r-', linewidth=2, marker='s')
            axes[0, 1].set_title('Latency Over Time')
            axes[0, 1].set_ylabel('Latency (ms)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Alerts by type
        if self.alerts:
            alert_types = self._count_alert_types(self.alerts)
            if alert_types:
                axes[0, 2].pie(alert_types.values(), labels=alert_types.keys(), autopct='%1.1f%%')
                axes[0, 2].set_title('Alerts by Type')
            else:
                axes[0, 2].text(0.5, 0.5, 'No Alerts', ha='center', va='center', fontsize=14)
                axes[0, 2].set_title('Alerts by Type')
        else:
            axes[0, 2].text(0.5, 0.5, 'No Alerts', ha='center', va='center', fontsize=14)
            axes[0, 2].set_title('Alerts by Type')
        
        # Data drift over time
        if self.drift_data:
            drift_timestamps = [d['timestamp'] for d in self.drift_data]
            drift_scores = [d['drift_score'] for d in self.drift_data]
            
            dt_drift_timestamps = [datetime.fromisoformat(ts) for ts in drift_timestamps]
            
            axes[1, 0].plot(dt_drift_timestamps, drift_scores, 'g-', linewidth=2, marker='^')
            axes[1, 0].axhline(y=self.thresholds['data_drift'], color='r', linestyle='--', 
                              label=f"Threshold: {self.thresholds['data_drift']}")
            axes[1, 0].set_title('Data Drift Over Time')
            axes[1, 0].set_ylabel('Drift Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Alert severity distribution
        if self.alerts:
            severities = [a['severity'] for a in self.alerts]
            severity_counts = {'high': severities.count('high'), 
                             'medium': severities.count('medium'), 
                             'low': severities.count('low')}
            
            axes[1, 1].bar(severity_counts.keys(), severity_counts.values(), 
                          color=['red', 'orange', 'yellow'], alpha=0.7)
            axes[1, 1].set_title('Alert Severity Distribution')
            axes[1, 1].set_ylabel('Number of Alerts')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Alerts', ha='center', va='center', fontsize=14)
            axes[1, 1].set_title('Alert Severity Distribution')
        
        # System health summary
        report = self.generate_report()
        health_score = 100
        
        if report['alerts_summary'].get('recent_alerts', 0) > 0:
            health_score -= report['alerts_summary']['recent_alerts'] * 10
        
        if report['drift_analysis'].get('drift_incidents', 0) > 0:
            health_score -= report['drift_analysis']['drift_incidents'] * 15
        
        health_score = max(0, health_score)
        
        color = 'green' if health_score > 80 else 'orange' if health_score > 60 else 'red'
        axes[1, 2].bar(['System Health'], [health_score], color=color, alpha=0.7)
        axes[1, 2].set_title('System Health Score')
        axes[1, 2].set_ylabel('Health Score (0-100)')
        axes[1, 2].set_ylim(0, 100)
        axes[1, 2].text(0, health_score + 2, f'{health_score:.1f}', ha='center', fontweight='bold')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('monitoring/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Monitoring dashboard created!")


def create_monitoring_demo():
    """Create monitoring demonstration"""
    print("📊 Creating monitoring demonstration...")
    
    # Create sample monitoring data
    monitor = ModelMonitor()
    
    # Simulate metrics over time
    base_accuracy = 0.92
    base_latency = 150
    
    for i in range(20):
        # Simulate some variation in metrics
        accuracy = base_accuracy + np.random.normal(0, 0.02)
        latency = base_latency + np.random.normal(0, 20)
        error_rate = np.random.exponential(0.01)
        
        metrics = {
            'accuracy': max(0, min(1, accuracy)),
            'latency': max(50, latency),
            'error_rate': min(0.1, error_rate)
        }
        
        monitor.log_metrics(metrics, f"v1.{i}")
        
        # Simulate some drift detection
        if i % 5 == 0:
            data_stats = {'mean': np.random.normal(0, 1), 'std': np.random.exponential(1)}
            monitor.detect_data_drift(data_stats)
    
    # Generate report and dashboard
    report = monitor.generate_report()
    monitor.plot_monitoring_dashboard()
    
    print("✅ Monitoring demonstration created!")
    return report


def main():
    """Test model monitoring system"""
    print("🧪 Testing Model Monitoring System...")
    
    # Create monitoring demo
    report = create_monitoring_demo()
    
    print(f"📊 Monitoring Report Summary:")
    print(f"   Latest Accuracy: {report['summary'].get('latest_accuracy', 0):.3f}")
    print(f"   Recent Alerts: {report['alerts_summary'].get('recent_alerts', 0)}")
    print(f"   Drift Incidents: {report['drift_analysis'].get('drift_incidents', 0)}")
    print(f"   Recommendations: {len(report['recommendations'])}")
    
    print("🎉 Model monitoring system working!")


if __name__ == "__main__":
    main()
