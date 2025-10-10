"""
dashboard/data_processor.py - Data preparation for dashboard visualizations

This module handles:
- Data aggregation and summarization
- KPI calculation
- Time series preparation
- Competitor analysis data processing
- Alert generation and processing
"""

# Standard library imports for data handling and typing
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# System path configuration for module imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DashboardDataProcessor:
    def __init__(self):
        """Initialize logging for the data processor"""
        self.logger = logging.getLogger(__name__)
    
    def process_for_dashboard(self, data: pd.DataFrame, filters: Dict = None) -> Dict:
        """
        Transform raw data into dashboard-ready format
        
        Args:
            data: Raw DataFrame containing collected data
            filters: Optional filtering criteria
            
        Returns:
            Dictionary containing processed dashboard components
        """
        try:
            # Return empty structure if no data available
            if data.empty:
                return self._empty_dashboard_data()
            
            # Calculate main dashboard metrics
            kpis = self._calculate_kpis(data)
            
            # Process temporal data patterns
            time_series = self._prepare_time_series(data)
            
            # Analyze competitor information
            competitor_data = self._prepare_competitor_data(data)
            
            # Extract emerging topics and trends
            trending = self._extract_trending_topics(data)
            
            # Generate system alerts
            alert_data = self._prepare_alert_data(data)
            
            # Combine all processed components
            return {
                'kpis': kpis,
                'time_series': time_series,
                'competitor_data': competitor_data,
                'trending_topics': trending,
                'alert_data': alert_data,
                'raw_data_summary': self._summarize_raw_data(data),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing dashboard data: {e}")
            return self._empty_dashboard_data()
    
    def _calculate_kpis(self, data: pd.DataFrame) -> Dict:
        """Calculate key performance indicators"""
        try:
            kpis = {
                'total_data_points': len(data),
                'unique_sources': data['source'].nunique() if 'source' in data.columns else 0,
                'date_range': self._get_date_range(data),
            }
            
            # Sentiment KPIs
            if 'sentiment_sentiment' in data.columns:
                sentiment_counts = data['sentiment_sentiment'].value_counts()
                total = len(data)
                
                kpis.update({
                    'positive_sentiment_pct': (sentiment_counts.get('positive', 0) / total * 100) if total > 0 else 0,
                    'negative_sentiment_pct': (sentiment_counts.get('negative', 0) / total * 100) if total > 0 else 0,
                    'neutral_sentiment_pct': (sentiment_counts.get('neutral', 0) / total * 100) if total > 0 else 0,
                    'average_confidence': data['sentiment_confidence'].mean() if 'sentiment_confidence' in data.columns else 0
                })
            
            # Engagement KPIs
            if 'like_count' in data.columns:
                kpis.update({
                    'total_likes': data['like_count'].sum(),
                    'average_likes': data['like_count'].mean(),
                    'total_retweets': data['retweet_count'].sum() if 'retweet_count' in data.columns else 0
                })
            
            return kpis
            
        except Exception as e:
            self.logger.error(f"Error calculating KPIs: {e}")
            return {'total_data_points': 0}
    
    def _prepare_time_series(self, data: pd.DataFrame) -> Dict:
        """Prepare time series data for charts"""
        try:
            if 'created_at' not in data.columns:
                return {}
            
            # Convert to datetime
            data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
            
            # Group by hour for timeline
            hourly_data = data.set_index('created_at').resample('H').agg({
                'sentiment_sentiment': lambda x: (x == 'positive').sum() / len(x) if len(x) > 0 else 0,
                'sentiment_confidence': 'mean'
            }).fillna(0)
            
            return {
                'timestamps': hourly_data.index.tolist(),
                'positive_sentiment': hourly_data['sentiment_sentiment'].tolist(),
                'confidence': hourly_data['sentiment_confidence'].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing time series: {e}")
            return {}
    
    def _prepare_competitor_data(self, data: pd.DataFrame) -> Dict:
        """Prepare competitor analysis data"""
        try:
            # Mock competitor data - in real implementation, you'd identify competitors
            # from the data based on mentions, hashtags, or other identifiers
            
            competitors = {
                'Competitor A': {'sentiment': 0.65, 'mentions': 45, 'engagement': 1200},
                'Competitor B': {'sentiment': 0.72, 'mentions': 38, 'engagement': 980},
                'Competitor C': {'sentiment': 0.58, 'mentions': 52, 'engagement': 1450},
                'Our Company': {'sentiment': 0.68, 'mentions': 67, 'engagement': 1800}
            }
            
            return competitors
            
        except Exception as e:
            self.logger.error(f"Error preparing competitor data: {e}")
            return {}
    
    def _extract_trending_topics(self, data: pd.DataFrame) -> Dict:
        """Extract trending topics and keywords"""
        try:
            trending_topics = []
            
            if 'sentiment_keywords' in data.columns:
                # Flatten all keywords
                all_keywords = []
                for keywords in data['sentiment_keywords'].dropna():
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
                
                # Count keyword frequency
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                
                # Get top 20 trending topics
                trending_topics = [
                    {'keyword': keyword, 'count': count, 'trend': 'up'}
                    for keyword, count in keyword_counts.most_common(20)
                ]
            
            return {
                'trending_keywords': trending_topics,
                'total_unique_keywords': len(set([t['keyword'] for t in trending_topics]))
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting trending topics: {e}")
            return {'trending_keywords': [], 'total_unique_keywords': 0}
    
    def _prepare_alert_data(self, data: pd.DataFrame) -> Dict:
        """Prepare alert-related data"""
        try:
            alerts = []
            
            # Generate alerts based on data patterns
            if 'sentiment_confidence' in data.columns and 'sentiment_sentiment' in data.columns:
                high_confidence_data = data[data['sentiment_confidence'] > 0.8]
                
                for _, row in high_confidence_data.iterrows():
                    alerts.append({
                        'timestamp': row.get('created_at', datetime.now()).isoformat(),
                        'type': 'HIGH_CONFIDENCE_SENTIMENT',
                        'sentiment': row.get('sentiment_sentiment', 'neutral'),
                        'confidence': row.get('sentiment_confidence', 0),
                        'content': row.get('text', '')[:100] + '...' if row.get('text') else ''
                    })
            
            return {
                'recent_alerts': alerts[-10:],  # Last 10 alerts
                'total_alerts': len(alerts),
                'alert_summary': self._summarize_alerts(alerts)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing alert data: {e}")
            return {'recent_alerts': [], 'total_alerts': 0}
    
    def _summarize_alerts(self, alerts: List[Dict]) -> Dict:
        """Summarize alert statistics"""
        if not alerts:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        # Categorize alerts by confidence
        high_conf = len([a for a in alerts if a.get('confidence', 0) > 0.8])
        med_conf = len([a for a in alerts if 0.6 <= a.get('confidence', 0) <= 0.8])
        low_conf = len(alerts) - high_conf - med_conf
        
        return {'high': high_conf, 'medium': med_conf, 'low': low_conf}
    
    def _get_date_range(self, data: pd.DataFrame) -> Dict:
        """Get date range of the data"""
        try:
            if 'created_at' in data.columns:
                dates = pd.to_datetime(data['created_at'], errors='coerce').dropna()
                if len(dates) > 0:
                    return {
                        'start': dates.min().isoformat(),
                        'end': dates.max().isoformat(),
                        'days': (dates.max() - dates.min()).days
                    }
        except:
            pass
        
        return {'start': None, 'end': None, 'days': 0}
    
    def _summarize_raw_data(self, data: pd.DataFrame) -> Dict:
        """Create summary of raw data"""
        return {
            'total_rows': len(data),
            'columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'null_counts': data.isnull().sum().to_dict(),
            'sample_data': data.head(3).to_dict('records') if len(data) > 0 else []
        }
    
    def _empty_dashboard_data(self) -> Dict:
        """Return empty dashboard data structure"""
        return {
            'kpis': {'total_data_points': 0},
            'time_series': {},
            'competitor_data': {},
            'trending_topics': {'trending_keywords': [], 'total_unique_keywords': 0},
            'alert_data': {'recent_alerts': [], 'total_alerts': 0},
            'raw_data_summary': {'total_rows': 0, 'columns': []},
            'last_updated': datetime.now().isoformat()
        }

# Test the data processor
def test_data_processor():
    """Test the data processor functionality"""
    print("Testing Dashboard Data Processor...")
    
    try:
        processor = DashboardDataProcessor()
        print("✅ Data processor initialized")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'created_at': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'source': ['twitter'] * 50 + ['news'] * 50,
            'sentiment_sentiment': ['positive'] * 40 + ['negative'] * 30 + ['neutral'] * 30,
            'sentiment_confidence': np.random.uniform(0.5, 1.0, 100),
            'text': ['Sample text'] * 100
        })
        
        result = processor.process_for_dashboard(sample_data)
        print(f"✅ Processed data: {len(result)} sections")
        print(f"KPIs calculated: {result['kpis']['total_data_points']} data points")
        
        print("Data processor ready!")
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")

if __name__ == "__main__":
    test_data_processor()
