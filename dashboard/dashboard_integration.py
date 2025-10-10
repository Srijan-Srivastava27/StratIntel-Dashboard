"""
Dashboard integration with existing Strategic Intelligence Platform

This module handles:
- Dashboard initialization and configuration
- System component integration
- Server deployment and monitoring
- Status reporting and logging
"""

# System path configuration for component imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import core platform components
from main_dashboard import StrategicIntelligenceDashboard
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from alert_system import AlertSystem

class DashboardLauncher:
    """Handles dashboard initialization and deployment"""
    
    def __init__(self):
        """Initialize dashboard with required components"""
        # Create main dashboard instance with default configuration
        self.dashboard = StrategicIntelligenceDashboard()
    
    def launch(self, host='127.0.0.1', port=5000, debug=True):
        """
        Launch the dashboard server with specified configuration
        
        Args:
            host: Server host address (default: localhost)
            port: Server port number (default: 5000)
            debug: Enable debug mode (default: True)
        """
        # Display startup banner with configuration details
        print("=" * 60)
        print("🚀 LAUNCHING STRATEGIC INTELLIGENCE DASHBOARD")
        print("=" * 60)
        print(f"📊 Dashboard URL: http://{host}:{port}")
        print(f"🔄 Auto-refresh: Every 5 minutes")
        print(f"📱 Mobile-responsive: Yes")
        print(f"📈 Real-time data: Yes")
        print("=" * 60)
        
        # Start dashboard server with specified configuration
        self.dashboard.run_dashboard(host=host, port=port, debug=debug)

# Entry point for direct execution
if __name__ == "__main__":
    # Initialize and launch dashboard
    launcher = DashboardLauncher()
    launcher.launch()
