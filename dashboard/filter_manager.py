"""
dashboard/filter_manager.py - Data filtering and selection logic

This module provides:
- Filter application and validation
- Dynamic filter discovery
- Filter chain management
- Filter optimization
"""

# Core data processing imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

# Configure Python path for imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class FilterManager:
    """
    Manages data filtering operations and filter state
    
    Handles:
    - Filter chain application
    - Filter validation
    - Available filter discovery
    - Filter optimization
    """
    
    def __init__(self):
        """Initialize filter manager with logging and filter definitions"""
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Define supported filter types and options
        self.available_filters = {
            'time_range': ['1d', '7d', '30d', '90d', 'custom'],
            'sentiment_types': ['positive', 'negative', 'neutral', 'all'],
            'data_sources': ['twitter', 'news', 'reddit', 'forums', 'all'],
            'confidence_levels': ['high', 'medium', 'low', 'all']
        }
    
    def apply_filters(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply multiple filters to the dataset"""
        try:
            if data.empty:
                return data
            
            filtered_data = data.copy()
            
            # Apply time range filter
            if filters.get('time_range') or filters.get('start_date') or filters.get('end_date'):
                filtered_data = self._apply_time_filter(filtered_data, filters)
            
            # Apply sentiment filter
            if filters.get('sentiment_filter') and filters['sentiment_filter'] != 'all':
                filtered_data = self._apply_sentiment_filter(filtered_data, filters)
            
            # Apply source filter
            if filters.get('data_sources') and 'all' not in filters['data_sources']:
                filtered_data = self._apply_source_filter(filtered_data, filters)
            
            # Apply competitor filter
            if filters.get('competitors'):
                filtered_data = self._apply_competitor_filter(filtered_data, filters)
            
            # Apply confidence filter
            if filters.get('confidence_levels') and 'all' not in filters['confidence_levels']:
                filtered_data = self._apply_confidence_filter(filtered_data, filters)
            
            # Apply sector filter
            if filters.get('sectors'):
                filtered_data = self._apply_sector_filter(filtered_data, filters)
            
            self.logger.info(f"Filtering reduced data from {len(data)} to {len(filtered_data)} rows")
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {e}")
            return data
    
    def _apply_time_filter(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply time-based filtering"""
        try:
            if 'created_at' not in data.columns:
                return data
            
            # Convert to datetime
            data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
            data = data.dropna(subset=['created_at'])
            
            # Determine time range
            end_time = datetime.now()
            
            if filters.get('start_date') and filters.get('end_date'):
                start_time = pd.to_datetime(filters['start_date'])
                end_time = pd.to_datetime(filters['end_date'])
            else:
                time_range = filters.get('time_range', '7d')
                if time_range == '1d':
                    start_time = end_time - timedelta(days=1)
                elif time_range == '7d':
                    start_time = end_time - timedelta(days=7)
                elif time_range == '30d':
                    start_time = end_time - timedelta(days=30)
                elif time_range == '90d':
                    start_time = end_time - timedelta(days=90)
                else:
                    start_time = end_time - timedelta(days=7)  # Default
            
            # Apply filter
            mask = (data['created_at'] >= start_time) & (data['created_at'] <= end_time)
            return data[mask]
            
        except Exception as e:
            self.logger.error(f"Error applying time filter: {e}")
            return data
    
    def _apply_sentiment_filter(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply sentiment-based filtering"""
        try:
            sentiment_col = None
            
            # Find sentiment column (could be different names)
            for col in ['sentiment_sentiment', 'sentiment', 'analyzed_sentiment']:
                if col in data.columns:
                    sentiment_col = col
                    break
            
            if not sentiment_col:
                return data
            
            sentiment_filter = filters['sentiment_filter']
            if sentiment_filter == 'all':
                return data
            
            return data[data[sentiment_col] == sentiment_filter]
            
        except Exception as e:
            self.logger.error(f"Error applying sentiment filter: {e}")
            return data
    
    def _apply_source_filter(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply data source filtering"""
        try:
            if 'source' not in data.columns:
                return data
            
            sources = filters['data_sources']
            if not sources or 'all' in sources:
                return data
            
            return data[data['source'].isin(sources)]
            
        except Exception as e:
            self.logger.error(f"Error applying source filter: {e}")
            return data
    
    def _apply_competitor_filter(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply competitor-based filtering"""
        try:
            competitors = filters['competitors']
            if not competitors:
                return data
            
            # This would need to be adapted based on how you identify competitor mentions
            # For now, we'll filter based on text content containing competitor names
            if 'text' in data.columns:
                mask = data['text'].str.contains('|'.join(competitors), case=False, na=False)
                return data[mask]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error applying competitor filter: {e}")
            return data
    
    def _apply_confidence_filter(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply confidence level filtering"""
        try:
            confidence_col = None
            
            # Find confidence column
            for col in ['sentiment_confidence', 'confidence', 'analyzed_confidence']:
                if col in data.columns:
                    confidence_col = col
                    break
            
            if not confidence_col:
                return data
            
            confidence_levels = filters['confidence_levels']
            if 'all' in confidence_levels:
                return data
            
            # Define confidence thresholds
            mask = pd.Series([False] * len(data))
            
            if 'high' in confidence_levels:
                mask |= data[confidence_col] >= 0.8
            if 'medium' in confidence_levels:
                mask |= (data[confidence_col] >= 0.5) & (data[confidence_col] < 0.8)
            if 'low' in confidence_levels:
                mask |= data[confidence_col] < 0.5
            
            return data[mask]
            
        except Exception as e:
            self.logger.error(f"Error applying confidence filter: {e}")
            return data
    
    def _apply_sector_filter(self, data: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply sector-based filtering"""
        try:
            sectors = filters['sectors']
            if not sectors:
                return data
            
            # This would be adapted based on how sectors are identified in your data
            # Could be based on keywords, categories, or separate sector columns
            if 'text' in data.columns:
                sector_keywords = {
                    'technology': ['tech', 'AI', 'software', 'digital', 'innovation'],
                    'finance': ['financial', 'banking', 'investment', 'trading', 'market'],
                    'healthcare': ['health', 'medical', 'pharmaceutical', 'biotech'],
                    'retail': ['retail', 'consumer', 'shopping', 'e-commerce']
                }
                
                mask = pd.Series([False] * len(data))
                
                for sector in sectors:
                    if sector.lower() in sector_keywords:
                        keywords = sector_keywords[sector.lower()]
                        sector_mask = data['text'].str.contains('|'.join(keywords), case=False, na=False)
                        mask |= sector_mask
                
                return data[mask]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error applying sector filter: {e}")
            return data
    
    def get_available_filters(self, data: pd.DataFrame) -> Dict:
        """Get available filter options based on current data"""
        try:
            available = {}
            
            # Time range options (always available)
            available['time_ranges'] = self.available_filters['time_range']
            
            # Sentiment options
            if any(col in data.columns for col in ['sentiment_sentiment', 'sentiment']):
                sentiment_col = 'sentiment_sentiment' if 'sentiment_sentiment' in data.columns else 'sentiment'
                available['sentiments'] = ['all'] + data[sentiment_col].dropna().unique().tolist()
            
            # Source options
            if 'source' in data.columns:
                available['sources'] = ['all'] + data['source'].dropna().unique().tolist()
            
            # Confidence options (always available if confidence column exists)
            if any(col in data.columns for col in ['sentiment_confidence', 'confidence']):
                available['confidence_levels'] = self.available_filters['confidence_levels']
            
            # Date range
            if 'created_at' in data.columns:
                dates = pd.to_datetime(data['created_at'], errors='coerce').dropna()
                if len(dates) > 0:
                    available['date_range'] = {
                        'min_date': dates.min().date().isoformat(),
                        'max_date': dates.max().date().isoformat()
                    }
            
            return available
            
        except Exception as e:
            self.logger.error(f"Error getting available filters: {e}")
            return {}
    
    def validate_filters(self, filters: Dict) -> Dict:
        """Validate and sanitize filter inputs"""
        validated = {}
        
        try:
            # Time range validation
            if filters.get('time_range') in self.available_filters['time_range']:
                validated['time_range'] = filters['time_range']
            
            # Date validation
            for date_field in ['start_date', 'end_date']:
                if filters.get(date_field):
                    try:
                        pd.to_datetime(filters[date_field])
                        validated[date_field] = filters[date_field]
                    except:
                        self.logger.warning(f"Invalid date format for {date_field}: {filters[date_field]}")
            
            # Sentiment filter validation
            if filters.get('sentiment_filter') in self.available_filters['sentiment_types']:
                validated['sentiment_filter'] = filters['sentiment_filter']
            
            # List-based filter validation
            for filter_name in ['data_sources', 'competitors', 'sectors', 'confidence_levels']:
                if filters.get(filter_name):
                    if isinstance(filters[filter_name], list):
                        validated[filter_name] = filters[filter_name]
                    elif isinstance(filters[filter_name], str):
                        validated[filter_name] = [filters[filter_name]]
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Error validating filters: {e}")
            return {}
    
    def get_filter_summary(self, filters: Dict, data_count: int) -> str:
        """Generate human-readable filter summary"""
        try:
            summary_parts = []
            
            if filters.get('time_range'):
                summary_parts.append(f"Time: {filters['time_range']}")
            
            if filters.get('sentiment_filter') and filters['sentiment_filter'] != 'all':
                summary_parts.append(f"Sentiment: {filters['sentiment_filter']}")
            
            if filters.get('data_sources') and 'all' not in filters['data_sources']:
                sources = ', '.join(filters['data_sources'])
                summary_parts.append(f"Sources: {sources}")
            
            if filters.get('competitors'):
                competitors = ', '.join(filters['competitors'][:3])  # Show first 3
                if len(filters['competitors']) > 3:
                    competitors += f" (+{len(filters['competitors'])-3} more)"
                summary_parts.append(f"Competitors: {competitors}")
            
            summary = " | ".join(summary_parts) if summary_parts else "No filters applied"
            return f"{summary} ({data_count:,} items)"
            
        except Exception as e:
            self.logger.error(f"Error creating filter summary: {e}")
            return f"Filters applied ({data_count:,} items)"

# Test the filter manager
def test_filter_manager():
    """Test the filter manager functionality"""
    print("Testing Filter Manager...")
    
    try:
        filter_manager = FilterManager()
        print("✅ Filter manager initialized")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'created_at': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'source': ['twitter'] * 50 + ['news'] * 50,
            'sentiment_sentiment': ['positive'] * 40 + ['negative'] * 30 + ['neutral'] * 30,
            'sentiment_confidence': np.random.uniform(0.5, 1.0, 100),
            'text': ['Sample text about technology'] * 100
        })
        
        # Test filters
        test_filters = {
            'time_range': '7d',
            'sentiment_filter': 'positive',
            'data_sources': ['twitter']
        }
        
        filtered_data = filter_manager.apply_filters(sample_data, test_filters)
        print(f"✅ Applied filters: {len(sample_data)} -> {len(filtered_data)} rows")
        
        # Test available filters
        available = filter_manager.get_available_filters(sample_data)
        print(f"✅ Available filter options: {len(available)} types")
        
        print("Filter manager ready!")
        
    except Exception as e:
        print(f"❌ Filter manager test failed: {e}")

if __name__ == "__main__":
    test_filter_manager()
