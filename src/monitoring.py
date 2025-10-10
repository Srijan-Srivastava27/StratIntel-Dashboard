"""
System and Application Monitoring Module

Provides:
- System resource monitoring (CPU, memory, disk)
- Application metrics tracking
- Health check functionality
- Performance logging
"""

# System monitoring and logging imports
import psutil            # For system metrics collection
import logging          # For logging functionality
import time            # For timing operations
from datetime import datetime  # For timestamp handling
from typing import Dict       # For type hints
import pandas as pd          # For data handling

class SystemMonitor:
    """System resource monitoring and health checks"""
    
    def __init__(self):
        # Initialize logging for this component
        self.logger = logging.getLogger(__name__)
        # Record start time for uptime calculation
        self.start_time = datetime.now()
        
    def get_system_metrics(self) -> Dict:
        """
        Collect current system performance metrics
        
        Returns:
            Dictionary containing CPU, memory, disk usage and uptime
        """
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),     # CPU usage with 1s sampling
            'memory_percent': psutil.virtual_memory().percent, # RAM usage percentage
            'disk_usage': psutil.disk_usage('/').percent,     # Root disk usage
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600  # System uptime
        }
    
    def log_performance_metrics(self, metrics: Dict):
        """
        Log current system performance metrics
        
        Args:
            metrics: Dictionary containing system metrics
        """
        self.logger.info(
            f"System Performance - CPU: {metrics['cpu_percent']}%, "  # CPU utilization
            f"Memory: {metrics['memory_percent']}%, "                # Memory usage
            f"Disk: {metrics['disk_usage']}%, "                     # Disk usage
            f"Uptime: {metrics['uptime_hours']:.1f}h"              # System uptime
        )
    
    def check_system_health(self) -> bool:
        """
        Verify system health against thresholds
        
        Returns:
            Boolean indicating if system is healthy
        """
        # Get current metrics
        metrics = self.get_system_metrics()
        
        # Check against defined thresholds
        if metrics['cpu_percent'] > 80:  # CPU threshold: 80%
            self.logger.warning(f"High CPU usage: {metrics['cpu_percent']}%")
            return False
            
        if metrics['memory_percent'] > 85:  # Memory threshold: 85%
            self.logger.warning(f"High memory usage: {metrics['memory_percent']}%")
            return False
            
        if metrics['disk_usage'] > 90:  # Disk threshold: 90%
            self.logger.warning(f"High disk usage: {metrics['disk_usage']}%")
            return False
        
        return True  # All metrics within acceptable ranges

class ApplicationMonitor:
    """Application-specific monitoring and metrics"""
    
    def __init__(self):
        # Initialize logging for application monitoring
        self.logger = logging.getLogger(__name__)
        # Initialize metrics counters
        self.metrics = {
            'api_calls': 0,              # Total API requests
            'successful_collections': 0,  # Successful data collections
            'failed_collections': 0,      # Failed data collections
            'alerts_sent': 0,            # Number of alerts sent
            'errors': 0                  # Total error count
        }
    
    def increment_metric(self, metric_name: str):
        """
        Increment specified metric counter
        
        Args:
            metric_name: Name of metric to increment
        """
        if metric_name in self.metrics:
            self.metrics[metric_name] += 1
    
    def log_daily_summary(self):
        """Log summary of daily application metrics"""
        self.logger.info(
            f"Daily Summary - API Calls: {self.metrics['api_calls']}, "          # API usage
            f"Successful Collections: {self.metrics['successful_collections']}, " # Successes
            f"Failed Collections: {self.metrics['failed_collections']}, "        # Failures
            f"Alerts Sent: {self.metrics['alerts_sent']}, "                     # Alert count
            f"Errors: {self.metrics['errors']}"                                 # Error count
        )
    
    def get_success_rate(self) -> float:
        """
        Calculate data collection success rate
        
        Returns:
            Float representing success rate (0.0 to 1.0)
        """
        # Calculate total collection attempts
        total = self.metrics['successful_collections'] + self.metrics['failed_collections']
        # Return 0 if no collections attempted
        if total == 0:
            return 0.0
        # Calculate and return success rate
        return self.metrics['successful_collections'] / total
