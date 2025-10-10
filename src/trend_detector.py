"""
TrendDetector - Advanced trend detection and pattern analysis module

This module provides:
- Real-time trend identification using NLP
- Statistical pattern detection
- Sentiment shift analysis
- Anomaly detection algorithms
"""

# Import required libraries with specific purposes
import pandas as pd          # For data manipulation and time series analysis
import numpy as np          # For numerical computations and statistics
from sklearn.cluster import KMeans  # For text clustering and trend grouping
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction
from collections import Counter     # For frequency analysis
import logging                      # For error tracking and debugging
from typing import Dict, List, Union  # For type annotations

class TrendDetector:
    """Analyzes temporal patterns and emerging trends in textual data"""
    
    def __init__(self):
        """Initialize trend detection components"""
        # Initialize logging system for error tracking
        self.logger = logging.getLogger(__name__)
        
        # Configure TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,     # Limit features to prevent memory issues
            stop_words='english'   # Remove common English words for better signal
        )

    def detect_emerging_trends(self, data: pd.DataFrame, time_window: int = 24) -> Dict:
        """
        Detect and analyze emerging trends in recent data
        
        Args:
            data: DataFrame with text and temporal data
            time_window: Analysis window in hours
            
        Returns:
            Dictionary with trends, sentiment shifts, and anomalies
        """
        try:
            # Get recent data subset based on time window
            recent_data = self.filter_recent_data(data, time_window)
            
            # Return empty result if no data available
            if recent_data.empty:
                return {"trends": [], "keywords": [], "sentiment_shift": None}
            
            # Execute three-phase trend analysis
            trends = self.extract_trending_topics(recent_data)  # Phase 1: Topic extraction
            sentiment = self.analyze_sentiment_trends(recent_data)  # Phase 2: Sentiment analysis
            anomalies = self.detect_anomalies(recent_data)  # Phase 3: Anomaly detection
            
            # Combine all analysis results
            return {
                "trends": trends,
                "sentiment_analysis": sentiment,
                "anomalies": anomalies,
                "data_points": len(recent_data)
            }
            
        except Exception as e:
            self.logger.error(f"Trend detection failed: {e}")
            return {"trends": [], "keywords": [], "sentiment_shift": None}
    
    def extract_trending_topics(self, data: pd.DataFrame) -> List[Dict]:
        """
        Extract trending topics using TF-IDF vectorization and K-means clustering
        
        Args:
            data: DataFrame containing text data to analyze
            
        Returns:
            List of dictionaries containing cluster information and top terms
        """
        texts = data['text'].fillna('').tolist()
        
        if not texts:
            return []
        
        # TF-IDF vectorization
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # K-means clustering
        n_clusters = min(5, len(texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Extract top terms for each cluster
        feature_names = self.vectorizer.get_feature_names_out()
        trends = []
        
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-5:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            cluster_data = data[clusters == i]
            avg_sentiment = cluster_data['confidence'].mean() if 'confidence' in cluster_data.columns else 0
            
            trends.append({
                'cluster_id': i,
                'keywords': top_terms,
                'sentiment_score': avg_sentiment,
                'data_points': len(cluster_data)
            })
        
        return trends
    
    def analyze_sentiment_trends(self, data: pd.DataFrame) -> Dict[str, Union[str, Dict[str, float]]]:
        """Analyze sentiment trends over time"""
        if 'sentiment' not in data.columns:
            return {"overall_sentiment": "neutral", "trend": "stable"}
        
        sentiment_counts = data['sentiment'].value_counts()
        total_count = len(data)
        
        sentiment_distribution = {
            'positive': sentiment_counts.get('positive', 0) / total_count,
            'negative': sentiment_counts.get('negative', 0) / total_count,
            'neutral': sentiment_counts.get('neutral', 0) / total_count
        }
        
        # Determine overall trend
        if sentiment_distribution['positive'] > 0.6:
            overall_sentiment = "positive"
        elif sentiment_distribution['negative'] > 0.6:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "distribution": sentiment_distribution,
            "trend": self.determine_trend_direction(data)
        }
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Union[str, float]]]:
        """Detect anomalies in sentiment or engagement patterns"""
        anomalies = []
        
        # Check for sentiment anomalies
        if 'confidence' in data.columns:
            confidence_scores = data['confidence'].dropna()
            if len(confidence_scores) > 0:
                mean_confidence = confidence_scores.mean()
                std_confidence = confidence_scores.std()
                
                high_confidence_threshold = mean_confidence + 2 * std_confidence
                anomalous_items = data[data['confidence'] > high_confidence_threshold]
                
                for _, item in anomalous_items.iterrows():
                    anomalies.append({
                        'type': 'high_confidence_sentiment',
                        'confidence': item['confidence'],
                        'text': item.get('text', '')[:100] + '...',
                        'sentiment': item.get('sentiment', 'unknown')
                    })
        
        return anomalies
    
    def filter_recent_data(self, data: pd.DataFrame, hours: int) -> pd.DataFrame:
        """Filter data to recent time window"""
        if 'created_at' not in data.columns:
            return data
        
        cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
        return data[pd.to_datetime(data['created_at']) > cutoff_time]
    
    def determine_trend_direction(self, data: pd.DataFrame) -> str:
        """Determine if sentiment is improving or declining"""
        if len(data) < 10:
            return "insufficient_data"
        
        # Simple trend analysis based on recent vs earlier sentiment
        mid_point = len(data) // 2
        earlier_sentiment = data.iloc[:mid_point]['sentiment'].value_counts()
        recent_sentiment = data.iloc[mid_point:]['sentiment'].value_counts()
        
        earlier_positive_ratio = earlier_sentiment.get('positive', 0) / max(1, earlier_sentiment.sum())
        recent_positive_ratio = recent_sentiment.get('positive', 0) / max(1, recent_sentiment.sum())
        
        if recent_positive_ratio > earlier_positive_ratio + 0.1:
            return "improving"
        elif recent_positive_ratio < earlier_positive_ratio - 0.1:
            return "declining"
        else:
            return "stable"
