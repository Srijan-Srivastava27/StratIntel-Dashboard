"""
alert_system.py - Real-time alert system with Slack integration

This module provides:
- Alert generation and processing
- Slack integration and messaging
- Alert deduplication
- Alert severity determination
- Daily summary reporting
"""

# System imports for file and environment handling
import os
import requests
import json
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

class AlertSystem:
    def __init__(self):
        # Initialize logging first to ensure error tracking is available
        self.logger = logging.getLogger(__name__)
        
        # Load alert thresholds from environment with defaults
        self.sentiment_threshold = float(os.getenv('SENTIMENT_THRESHOLD', 0.7))  # 70% confidence for sentiment
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6)) # 60% general confidence
        
        # Initialize alert tracking system with cooldown
        self.recent_alerts = []  # List of (alert_key, timestamp) tuples
        self.alert_cooldown = 300  # Prevent duplicate alerts for 5 minutes
        
        # Setup Slack integration after logger initialization
        self.setup_slack()
        
    def setup_slack(self):
        """Initialize Slack webhook"""
        try:
            self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not self.slack_webhook_url or self.slack_webhook_url == 'https://hooks.slack.com/services/your/webhook/url':
                self.logger.warning("Slack webhook URL not found or using placeholder value")
                self.slack_webhook_url = None
            else:
                self.logger.info("Slack webhook initialized successfully")
        except Exception as e:
            self.logger.error(f"Slack setup failed: {e}")
            self.slack_webhook_url = None
    
    def create_alert(self, alert_type: str, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """Create alert based on analysis results"""
        timestamp = datetime.now()
        
        alert = {
            'id': f"{alert_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            'type': alert_type,
            'timestamp': timestamp.isoformat(),
            'data': data,
            'metadata': metadata or {},
            'severity': self.determine_severity(data),
            'message': self.generate_alert_message(alert_type, data)
        }
        
        return alert
    
    def determine_severity(self, data: Dict) -> str:
        """Determine alert severity based on data"""
        try:
            confidence = float(data.get('confidence', 0))
            market_relevance = str(data.get('market_relevance', 'low')).lower()
            sentiment = str(data.get('sentiment', 'neutral')).lower()
            
            # High severity conditions
            if (confidence >= 0.8 and market_relevance == 'high' and 
                sentiment in ['positive', 'negative']):
                return 'HIGH'
            
            # Medium severity conditions
            elif (confidence >= 0.6 and market_relevance in ['high', 'medium'] and
                  sentiment in ['positive', 'negative']):
                return 'MEDIUM'
            
            # Low severity - everything else
            else:
                return 'LOW'
        except Exception as e:
            self.logger.error(f"Error determining severity: {e}")
            return 'LOW'
    
    def generate_alert_message(self, alert_type: str, data: Dict) -> str:
        """Generate human-readable alert message"""
        try:
            sentiment = str(data.get('sentiment', 'neutral'))
            confidence = float(data.get('confidence', 0))
            market_relevance = str(data.get('market_relevance', 'low'))
            
            messages = {
                'SENTIMENT_CHANGE': f"Detected {sentiment} sentiment with {confidence:.1%} confidence (Market relevance: {market_relevance})",
                'TREND_DETECTED': f"New trend identified with {sentiment} sentiment",
                'ANOMALY_DETECTED': f"Anomaly detected in sentiment patterns",
                'HIGH_ENGAGEMENT': f"High engagement content detected with {sentiment} sentiment"
            }
            
            return messages.get(alert_type, f"Alert: {alert_type}")
        except Exception as e:
            self.logger.error(f"Error generating alert message: {e}")
            return f"Alert: {alert_type}"
    
    def send_slack_alert(self, alert: Dict) -> bool:
        """Send alert to Slack channel"""
        if not self.slack_webhook_url:
            self.logger.warning("Cannot send Slack alert - webhook URL not configured")
            return False
        
        try:
            # Check for alert cooldown to prevent spam
            if self._is_duplicate_alert(alert):
                self.logger.info(f"Skipping duplicate alert: {alert['type']}")
                return True
            
            severity_colors = {
                'HIGH': '#FF0000',    # Red
                'MEDIUM': '#FFA500',  # Orange  
                'LOW': '#FFFF00'      # Yellow
            }
            
            severity_emojis = {
                'HIGH': '🚨',
                'MEDIUM': '⚠️',
                'LOW': '💡'
            }
            
            color = severity_colors.get(alert['severity'], '#808080')
            emoji = severity_emojis.get(alert['severity'], '📊')
            
            # Safely extract data values
            data = alert.get('data', {})
            sentiment = str(data.get('sentiment', 'N/A')).title()
            confidence = float(data.get('confidence', 0))
            market_relevance = str(data.get('market_relevance', 'N/A')).title()
            
            # Prepare the message
            slack_message = {
                'text': f"{emoji} Strategic Intelligence Alert - {alert['severity']}",
                'attachments': [
                    {
                        'color': color,
                        'title': f"{alert['type']} Alert",
                        'fields': [
                            {
                                'title': 'Timestamp',
                                'value': datetime.fromisoformat(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
                                'short': True
                            },
                            {
                                'title': 'Severity',
                                'value': alert['severity'],
                                'short': True
                            },
                            {
                                'title': 'Sentiment',
                                'value': sentiment,
                                'short': True
                            },
                            {
                                'title': 'Confidence',
                                'value': f"{confidence:.1%}",
                                'short': True
                            },
                            {
                                'title': 'Market Relevance',
                                'value': market_relevance,
                                'short': True
                            }
                        ],
                        'text': alert.get('message', 'No message available')
                    }
                ]
            }
            
            # Add additional context if available
            keywords = data.get('keywords', [])
            if keywords and isinstance(keywords, list):
                slack_message['attachments'][0]['fields'].append({
                    'title': 'Keywords',
                    'value': ', '.join(str(k) for k in keywords[:5]),
                    'short': False
                })
            
            original_text = data.get('original_text', '')
            if original_text:
                text_preview = str(original_text)[:200] + '...' if len(str(original_text)) > 200 else str(original_text)
                slack_message['attachments'][0]['fields'].append({
                    'title': 'Content Preview',
                    'value': text_preview,
                    'short': False
                })
            
            # Send the message
            response = requests.post(
                self.slack_webhook_url,
                json=slack_message,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Slack alert sent successfully: {alert['type']} ({alert['severity']})")
                self._track_sent_alert(alert)
                return True
            else:
                self.logger.error(f"Failed to send Slack alert. Status code: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error sending Slack alert: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending Slack alert: {e}")
            return False
    
    def _is_duplicate_alert(self, alert: Dict) -> bool:
        """Check if this is a duplicate alert within the cooldown period"""
        try:
            current_time = datetime.now()
            alert_key = f"{alert['type']}_{alert['data'].get('sentiment', 'neutral')}"
            
            # Clean old alerts
            self.recent_alerts = [
                (key, timestamp) for key, timestamp in self.recent_alerts
                if (current_time - timestamp).total_seconds() < self.alert_cooldown
            ]
            
            # Check for duplicates
            for key, timestamp in self.recent_alerts:
                if key == alert_key:
                    return True
            
            return False
        except Exception as e:
            self.logger.error(f"Error checking duplicate alert: {e}")
            return False
    
    def _track_sent_alert(self, alert: Dict):
        """Track sent alert to prevent duplicates"""
        try:
            current_time = datetime.now()
            alert_key = f"{alert['type']}_{alert['data'].get('sentiment', 'neutral')}"
            self.recent_alerts.append((alert_key, current_time))
        except Exception as e:
            self.logger.error(f"Error tracking sent alert: {e}")
    
    def process_alerts(self, analysis_results: List[Dict], context: str = "general") -> int:
        """
        Process a batch of analysis results and send alerts if criteria are met
        
        Args:
            analysis_results: List of analysis result dictionaries
            context: Context identifier for the batch of alerts
            
        Returns:
            Number of alerts successfully sent
        """
        # Early return if no results to process
        if not analysis_results:
            self.logger.warning("No analysis results to process")
            return 0
            
        alerts_sent = 0  # Track number of successful alerts
        
        try:
            # Process each result individually
            for result in analysis_results:
                # Check if result meets alert criteria
                if self.should_alert(result):
                    # Create and send alert with context
                    alert = self.create_alert('SENTIMENT_CHANGE', result, {'context': context})
                    if self.send_slack_alert(alert):
                        alerts_sent += 1
                        
                    # Add delay between alerts to prevent rate limiting
                    time.sleep(1)
            
            # Log batch processing results
            self.logger.info(f"Processed {len(analysis_results)} results, sent {alerts_sent} alerts")
            return alerts_sent
            
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
            return alerts_sent
    
    def should_alert(self, result: Dict) -> bool:
        """Determine if result warrants an alert"""
        try:
            # Skip if there was an error in analysis
            if result.get('error', False):
                return False
            
            confidence = float(result.get('confidence', 0))
            market_relevance = str(result.get('market_relevance', 'low')).lower()
            sentiment = str(result.get('sentiment', 'neutral')).lower()
            
            # Alert conditions
            conditions = [
                # High confidence and market relevance
                confidence >= self.confidence_threshold and market_relevance in ['high', 'medium'],
                
                # Strong sentiment with good confidence
                sentiment in ['positive', 'negative'] and confidence >= self.sentiment_threshold,
                
                # High market relevance regardless of sentiment
                market_relevance == 'high' and confidence >= 0.5
            ]
            
            return any(conditions)
        except Exception as e:
            self.logger.error(f"Error determining if should alert: {e}")
            return False
    
    def send_daily_summary(self, summary_data: Dict) -> bool:
        """Send daily summary report to Slack"""
        if not self.slack_webhook_url:
            self.logger.warning("Cannot send daily summary - webhook URL not configured")
            return False
        
        try:
            summary_message = {
                'text': '📊 Daily Strategic Intelligence Summary',
                'attachments': [
                    {
                        'color': '#36a64f',  # Green
                        'title': 'Daily Analytics Report',
                        'fields': [
                            {
                                'title': 'Total Data Points',
                                'value': str(summary_data.get('total_items', 0)),
                                'short': True
                            },
                            {
                                'title': 'Sentiment Distribution',
                                'value': f"Positive: {summary_data.get('positive_count', 0)}\nNegative: {summary_data.get('negative_count', 0)}\nNeutral: {summary_data.get('neutral_count', 0)}",
                                'short': True
                            },
                            {
                                'title': 'Average Confidence',
                                'value': f"{summary_data.get('avg_confidence', 0):.1%}",
                                'short': True
                            },
                            {
                                'title': 'Alerts Sent',
                                'value': str(summary_data.get('alerts_sent', 0)),
                                'short': True
                            }
                        ],
                        'footer': 'Strategic Intelligence Platform',
                        'ts': int(time.time())
                    }
                ]
            }
            
            response = requests.post(self.slack_webhook_url, json=summary_message, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("Daily summary sent to Slack successfully")
                return True
            else:
                self.logger.error(f"Failed to send daily summary. Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {e}")
            return False
    
    def test_slack_connection(self) -> bool:
        """Test Slack webhook connection"""
        if not self.slack_webhook_url:
            self.logger.error("No Slack webhook URL configured")
            return False
        
        test_message = {
            'text': '✅ Strategic Intelligence Platform - Test Message',
            'attachments': [
                {
                    'color': '#36a64f',
                    'text': 'This is a test message to verify Slack integration is working correctly.',
                    'footer': 'Test completed at ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            ]
        }
        
        try:
            response = requests.post(self.slack_webhook_url, json=test_message, timeout=10)
            
            if response.status_code == 200:
                self.logger.info("Slack connection test successful")
                return True
            else:
                self.logger.error(f"Slack connection test failed. Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Slack connection test error: {e}")
            return False
    
    def get_status(self) -> Dict:
        """Get status of alert system"""
        return {
            'slack_configured': self.slack_webhook_url is not None,
            'sentiment_threshold': self.sentiment_threshold,
            'confidence_threshold': self.confidence_threshold,
            'recent_alerts_count': len(self.recent_alerts),
            'alert_cooldown_seconds': self.alert_cooldown
        }

# Test function with comprehensive error handling
def test_alert_system():
    """Test the alert system functionality"""
    print("Testing Alert System")
    print("=" * 50)
    
    try:
        alert_system = AlertSystem()
        
        # Print status
        status = alert_system.get_status()
        print("\nAlert System Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test Slack connection
        print(f"\nTesting Slack connection...")
        slack_configured = status['slack_configured']
        
        if slack_configured:
            print("Slack webhook URL is configured")
            if alert_system.test_slack_connection():
                print("✅ Slack connection successful!")
            else:
                print("❌ Slack connection failed!")
        else:
            print("⚠️  Slack webhook URL not configured - using mock mode")
        
        # Test alert creation and sending
        test_data = {
            'sentiment': 'positive',
            'confidence': 0.85,
            'market_relevance': 'high',
            'reasoning': 'Strong positive sentiment detected in market analysis',
            'keywords': ['growth', 'innovation', 'market', 'positive'],
            'original_text': 'This is a test message for alert system functionality testing.'
        }
        
        print(f"\nTesting alert creation...")
        alert = alert_system.create_alert('SENTIMENT_CHANGE', test_data)
        print(f"Alert created successfully:")
        print(f"  Type: {alert['type']}")
        print(f"  Severity: {alert['severity']}")
        print(f"  Message: {alert['message']}")
        
        # Test if alert should be sent
        should_send = alert_system.should_alert(test_data)
        print(f"\nAlert meets criteria for sending: {should_send}")
        
        if should_send:
            if slack_configured:
                if alert_system.send_slack_alert(alert):
                    print("✅ Alert sent successfully!")
                else:
                    print("❌ Failed to send alert")
            else:
                print("🔄 Would send alert if Slack was configured")
        else:
            print("ℹ️  Alert does not meet criteria for sending")
        
        # Test batch processing
        print(f"\nTesting batch alert processing...")
        test_results = [
            test_data,  # High confidence positive
            {
                'sentiment': 'negative',
                'confidence': 0.9,
                'market_relevance': 'high',
                'reasoning': 'Strong negative sentiment',
                'keywords': ['decline', 'loss']
            },
            {
                'sentiment': 'neutral',
                'confidence': 0.3,
                'market_relevance': 'low',
                'reasoning': 'Low confidence neutral'
            }
        ]
        
        alerts_sent = alert_system.process_alerts(test_results, "test_context")
        print(f"Batch processing completed: {alerts_sent} alerts sent out of {len(test_results)} results")
        
        print(f"\n✅ Alert system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_alert_system()
