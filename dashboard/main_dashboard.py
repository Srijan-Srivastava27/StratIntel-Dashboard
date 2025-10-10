"""
Strategic Intelligence Dashboard Main Interface

This module orchestrates:
- Central dashboard coordination and state management  
- Data pipeline integration and processing
- Web interface and API endpoints
- Real-time updates and caching
- Component lifecycle management
"""

# Standard library imports for core functionality
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Data processing and web framework imports
import pandas as pd
from flask import Flask, render_template, jsonify, request

# Add source directory to Python path for component imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import core platform components with error handling
try:
    # Load data collection and analysis components
    from data_collector import DataCollector
    from sentiment_analyzer import SentimentAnalyzer  
    from alert_system import AlertSystem
    from trend_detector import TrendDetector
except ImportError as e:
    # Log import failures and provide guidance
    print(f"Warning: Could not import components: {e}")
    print("Make sure the src/ directory is in your Python path")

# Import dashboard-specific components
from dashboard.visualization_engine import VisualizationEngine  # Chart generation
from dashboard.data_processor import DashboardDataProcessor    # Data transformation
from dashboard.filter_manager import FilterManager            # Query/filter handling
from dashboard.dashboard_config import DashboardConfig        # Configuration management


class StrategicIntelligenceDashboard:
    """
    Main dashboard class that coordinates all dashboard components and serves the web interface
    
    Attributes:
        config: Dashboard configuration manager
        data_collector: Component for gathering data from various sources
        sentiment_analyzer: NLP component for sentiment analysis
        trend_detector: Component for identifying emerging trends
        viz_engine: Visualization generation engine
        data_processor: Data processing and transformation component
        filter_manager: Query and filter management
        app: Flask application instance
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize dashboard components and setup web application
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = DashboardConfig(config_file)
        
        # Initialize core components
        self.data_collector = DataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.alert_system = AlertSystem()
        try:
            from trend_detector import TrendDetector
            self.trend_detector = TrendDetector()
        except ImportError:
            self.trend_detector = None
            self.logger.warning("TrendDetector not available")
        
        # Initialize dashboard components
        self.viz_engine = VisualizationEngine(self.config)
        self.data_processor = DashboardDataProcessor()
        self.filter_manager = FilterManager()
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        
        # Setup routes
        self._setup_routes()
        
        # Cache for dashboard data
        self._data_cache = {}
        self._cache_timestamp = None
        self._cache_duration = timedelta(minutes=5)  # 5-minute cache
        
    def _setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard_home():
            """Main dashboard page"""
            return render_template('dashboard.html', 
                                title='Strategic Intelligence Dashboard')
        
        @self.app.route('/api/dashboard_data')
        def get_dashboard_data():
            """API endpoint for dashboard data"""
            try:
                filters = self._parse_filters(request.args)
                data = self._get_dashboard_data(filters)
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting dashboard data: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/competitor_analysis')
        def get_competitor_analysis():
            """API endpoint for competitor analysis data"""
            try:
                filters = self._parse_filters(request.args)
                data = self._get_competitor_analysis_data(filters)
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting competitor analysis: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/trend_evolution')
        def get_trend_evolution():
            """API endpoint for trend evolution data"""
            try:
                filters = self._parse_filters(request.args)
                data = self._get_trend_evolution_data(filters)
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting trend evolution: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/alert_history')
        def get_alert_history():
            """API endpoint for alert history"""
            try:
                filters = self._parse_filters(request.args)
                data = self._get_alert_history_data(filters)
                return jsonify(data)
            except Exception as e:
                self.logger.error(f"Error getting alert history: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/charts/<chart_type>')
        def get_chart(chart_type):
            """API endpoint for individual charts"""
            try:
                filters = self._parse_filters(request.args)
                chart_data = self._generate_chart(chart_type, filters)
                return jsonify(chart_data)
            except Exception as e:
                self.logger.error(f"Error generating chart {chart_type}: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _parse_filters(self, args) -> Dict:
        """Parse filter parameters from request arguments"""
        filters = {
            'time_range': args.get('time_range', '7d'),  # 7 days default
            'competitors': args.getlist('competitors'),
            'sectors': args.getlist('sectors'),
            'sentiment_filter': args.get('sentiment_filter', 'all'),
            'data_sources': args.getlist('data_sources'),
            'start_date': args.get('start_date'),
            'end_date': args.get('end_date')
        }
        
        # Convert time_range to actual dates if needed
        if not filters['start_date'] and not filters['end_date']:
            end_date = datetime.now()
            if filters['time_range'] == '1d':
                start_date = end_date - timedelta(days=1)
            elif filters['time_range'] == '7d':
                start_date = end_date - timedelta(days=7)
            elif filters['time_range'] == '30d':
                start_date = end_date - timedelta(days=30)
            elif filters['time_range'] == '90d':
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=7)  # Default
            
            filters['start_date'] = start_date.isoformat()
            filters['end_date'] = end_date.isoformat()
        
        return filters
    
    def _get_dashboard_data(self, filters: Dict) -> Dict:
        """Get comprehensive dashboard data"""
        # Check cache first
        cache_key = str(sorted(filters.items()))
        if (self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_duration and
            cache_key in self._data_cache):
            return self._data_cache[cache_key]
        
        try:
            # Collect fresh data
            queries = self.config.get_monitoring_queries()
            raw_data = self.data_collector.collect_all_data(queries)
            
            if raw_data.empty:
                return self._get_empty_dashboard_data()
            
            # Process data through sentiment analyzer
            processed_data = self.sentiment_analyzer.analyze_dataframe(raw_data)
            
            # Apply filters
            filtered_data = self.filter_manager.apply_filters(processed_data, filters)
            
            # Generate dashboard metrics
            dashboard_data = self.data_processor.process_for_dashboard(filtered_data, filters)
            
            # Cache the results
            self._data_cache[cache_key] = dashboard_data
            self._cache_timestamp = datetime.now()
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error processing dashboard data: {e}")
            return self._get_empty_dashboard_data()
    
    def _get_empty_dashboard_data(self) -> Dict:
        """Return empty dashboard data structure"""
        return {
            'kpis': {
                'total_data_points': 0,
                'average_sentiment': 0.0,
                'active_alerts': 0,
                'competitor_count': 0
            },
            'charts': {},
            'tables': [],
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_competitor_analysis_data(self, filters: Dict) -> Dict:
        """Get competitor analysis specific data"""
        try:
            return {
                'competitor_trajectories': [],
                'market_share_analysis': {},
                'competitive_positioning': {},
                'sentiment_comparison': {}
            }
        except Exception as e:
            self.logger.error(f"Error in competitor analysis: {e}")
            return {}
    
    def _get_trend_evolution_data(self, filters: Dict) -> Dict:
        """Get trend evolution data"""
        try:
            if self.trend_detector:
                # Use your existing trend detection logic here
                pass
            
            return {
                'trend_timeline': [],
                'emerging_trends': [],
                'trend_strength': {},
                'keyword_evolution': {}
            }
        except Exception as e:
            self.logger.error(f"Error in trend evolution: {e}")
            return {}
    
    def _get_alert_history_data(self, filters: Dict) -> Dict:
        """Get alert history and current alerts"""
        try:
            return {
                'current_alerts': [],
                'alert_history': [],
                'alert_statistics': {},
                'alert_trends': {}
            }
        except Exception as e:
            self.logger.error(f"Error getting alert history: {e}")
            return {}
    
    def _generate_chart(self, chart_type: str, filters: Dict) -> Dict:
        """Generate specific chart data"""
        try:
            # Get filtered data
            dashboard_data = self._get_dashboard_data(filters)
            
            # Use visualization engine to create chart
            chart_data = self.viz_engine.generate_chart(
                chart_type, 
                dashboard_data, 
                filters
            )
            
            return chart_data
            
        except Exception as e:
            self.logger.error(f"Error generating chart {chart_type}: {e}")
            return {'error': str(e)}
    
    def run_dashboard(self, host='127.0.0.1', port=5000, debug=True):
        """Run the dashboard server"""
        self.logger.info(f"Starting Strategic Intelligence Dashboard on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def export_report(self, filters: Dict, format: str = 'pdf') -> str:
        """Export dashboard data as a report"""
        try:
            dashboard_data = self._get_dashboard_data(filters)
            
            # Generate report using visualization engine
            report_path = self.viz_engine.generate_report(
                dashboard_data, 
                filters, 
                format
            )
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
            raise

# Example usage and testing
def test_dashboard():
    """Test the dashboard functionality"""
    print("Testing Strategic Intelligence Dashboard...")
    
    try:
        dashboard = StrategicIntelligenceDashboard()
        print("✅ Dashboard initialized successfully")
        
        # Test data retrieval
        test_filters = {
            'time_range': '7d',
            'competitors': [],
            'sectors': [],
            'sentiment_filter': 'all'
        }
        
        data = dashboard._get_dashboard_data(test_filters)
        print(f"✅ Retrieved dashboard data: {len(data)} sections")
        
        print("Dashboard ready to run!")
        print("Run with: dashboard.run_dashboard()")
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the dashboard
    test_dashboard()
