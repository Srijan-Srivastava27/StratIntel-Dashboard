
"""
Strategic Intelligence Dashboard Launcher
"""
import sys
import os
import logging
from datetime import datetime
from dashboard.visualization_engine import VisualizationEngine
from dashboard.filter_manager import FilterManager
from dashboard.data_processor import DashboardDataProcessor
from dashboard.dashboard_config import DashboardConfig


def setup_environment():
    """Setup the environment for dashboard"""
    # Create necessary directories
    directories = [
        'dashboard/static/images/charts',
        'dashboard/config',
        'data/processed',
        'data/exports',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main function to launch the dashboard"""
    print("=" * 60)
    print("🚀 STRATEGIC INTELLIGENCE DASHBOARD LAUNCHER")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Configure logging
    log_filename = f"logs/dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Import dashboard components
        from dashboard.main_dashboard import StrategicIntelligenceDashboard
        
        print("✅ Dashboard modules imported successfully")
        
        # Initialize and launch dashboard
        dashboard = StrategicIntelligenceDashboard()
        
        print("✅ Dashboard initialized successfully")
        print(f"📊 Dashboard URL: http://127.0.0.1:5000")
        print(f"📝 Logs saved to: {log_filename}")
        print("=" * 60)
        print("🔄 Starting dashboard server...")
        print("   Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Launch the dashboard
        dashboard.run_dashboard(host='127.0.0.1', port=5000, debug=True)
        
    except ImportError as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dashboard modules are properly installed")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard failed to start: {e}")
        logging.error(f"Dashboard startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
