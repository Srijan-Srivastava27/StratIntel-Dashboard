"""
dashboard/dashboard_config.py - Dashboard configuration management

Handles:
- Loading and parsing YAML configuration
- Providing default configurations
- Configuration validation and access
- Configuration persistence
"""

# Required imports for configuration management
import os
import yaml
import logging
from typing import Dict, List, Optional

class DashboardConfig:
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        # Setup logging instance
        self.logger = logging.getLogger(__name__)
        
        # Set configuration file path (default or provided)
        self.config_file = config_file or os.path.join("dashboard", "config", "dashboard_config.yaml")
        
        # Load configuration settings
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file or use defaults
        
        Returns:
            Dictionary containing configuration settings
        """
        try:
            # Check if config file exists
            if os.path.exists(self.config_file):
                # Load and parse YAML configuration
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_file}")
                return config
            else:
                # Use default configuration if file not found
                self.logger.info("Using default configuration")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'dashboard': {
                'title': 'Strategic Intelligence Dashboard',
                'update_interval': 300,  # seconds
                'cache_duration': 300,   # seconds
                'max_data_points': 10000,
                'default_time_range': '7d'
            },
            'visualization': {
                'chart_height': 400,
                'chart_width': 600,
                'color_scheme': 'viridis',
                'font_size': 12,
                'show_grid': True,
                'animation': True
            },
            'data_sources': {
                'twitter': {
                    'enabled': True,
                    'weight': 1.0,
                    'color': '#1DA1F2'
                },
                'news': {
                    'enabled': True,
                    'weight': 1.2,
                    'color': '#FF6B35'
                },
                'reddit': {
                    'enabled': True,
                    'weight': 0.8,
                    'color': '#FF4500'
                }
            },
            'monitoring': {
                'queries': [
                    'artificial intelligence trends',
                    'market analysis',
                    'competitor intelligence',
                    'technology innovation',
                    'business strategy'
                ],
                'competitors': [
                    'Competitor A',
                    'Competitor B', 
                    'Competitor C'
                ],
                'sectors': [
                    'Technology',
                    'Finance',
                    'Healthcare',
                    'Retail'
                ]
            },
            'alerts': {
                'sentiment_threshold': 0.7,
                'confidence_threshold': 0.8,
                'trend_threshold': 0.6,
                'enable_notifications': True
            },
            'export': {
                'formats': ['pdf', 'xlsx', 'csv'],
                'include_charts': True,
                'include_raw_data': False
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_monitoring_queries(self) -> List[str]:
        """Get list of monitoring queries"""
        return self.get('monitoring.queries', [])
    
    def get_competitors(self) -> List[str]:
        """Get list of competitors to monitor"""
        return self.get('monitoring.competitors', [])
    
    def get_sectors(self) -> List[str]:
        """Get list of sectors to monitor"""
        return self.get('monitoring.sectors', [])
    
    def get_chart_config(self) -> Dict:
        """Get chart configuration"""
        return self.get('visualization', {})
    
    def get_alert_thresholds(self) -> Dict:
        """Get alert threshold configuration"""
        return self.get('alerts', {})
    
    def save_config(self, config_updates: Dict):
        """Save updated configuration"""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Update config
            self.config.update(config_updates)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            raise

# Test the configuration manager
def test_dashboard_config():
    """Test the dashboard configuration"""
    print("Testing Dashboard Configuration...")
    
    try:
        config = DashboardConfig()
        print("✅ Dashboard config initialized")
        
        # Test configuration access
        title = config.get('dashboard.title')
        queries = config.get_monitoring_queries()
        competitors = config.get_competitors()
        
        print(f"✅ Dashboard title: {title}")
        print(f"✅ Monitoring queries: {len(queries)}")
        print(f"✅ Competitors tracked: {len(competitors)}")
        
        print("Dashboard configuration ready!")
        
    except Exception as e:
        print(f"❌ Dashboard config test failed: {e}")

if __name__ == "__main__":
    test_dashboard_config()
