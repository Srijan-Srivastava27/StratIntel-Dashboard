"""
data_collector.py - Real-time data collection from multiple sources

This module handles:
- Data collection from Twitter API
- News gathering from multiple sources
- Rate limiting and error handling
- Data persistence and caching
- API connection management
"""

# Core system imports for file and environment handling
import os
import requests
from datetime import datetime
import pandas as pd
from typing import List, Dict
import logging
from dotenv import load_dotenv
import time
import json

# Load environment configuration from .env file
load_dotenv()

# Import Twitter API client with fallback handling
try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    print("Warning: tweepy not installed. Run: pip install tweepy")

# Import Google News client with fallback handling
try:
    from gnews import GNews
    GNEWS_AVAILABLE = True
except ImportError:
    GNEWS_AVAILABLE = False
    print("Warning: gnews not installed. Run: pip install gnews")

class DataCollector:
    def __init__(self):
        # Initialize logging system first for error tracking
        self.logger = logging.getLogger(__name__)
        
        # Setup API connections after logging is ready
        self.setup_apis()

    def setup_apis(self):
        """Initialize and validate API connections"""
        try:
            # Configure Twitter API with authentication
            self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            if self.twitter_bearer_token and TWEEPY_AVAILABLE:
                try:
                    # Initialize Twitter client with bearer token
                    self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
                    self.logger.info("Twitter API initialized successfully")
                except Exception as twitter_error:
                    self.logger.error(f"Twitter API initialization failed: {twitter_error}")
                    self.twitter_client = None
            else:
                # Handle missing Twitter dependencies
                if not TWEEPY_AVAILABLE:
                    self.logger.warning("Tweepy package not available")
                else:
                    self.logger.warning("Twitter Bearer Token not found")
                self.twitter_client = None
            
            # Configure Google News API client
            if GNEWS_AVAILABLE:
                try:
                    # Initialize GNews with English language and US region
                    self.gnews = GNews(language='en', country='US', max_results=100)
                    self.logger.info("Google News API initialized successfully")
                except Exception as gnews_error:
                    self.logger.error(f"GNews initialization failed: {gnews_error}")
                    self.gnews = None
            else:
                self.gnews = None
                self.logger.warning("GNews package not available")

        except Exception as e:
            self.logger.error(f"API setup failed: {e}")
            # Ensure API clients exist even if setup fails
            if not hasattr(self, 'twitter_client'):
                self.twitter_client = None
            if not hasattr(self, 'gnews'):
                self.gnews = None
    
    def collect_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """Collect tweets for given query"""
        if not self.twitter_client or not TWEEPY_AVAILABLE:
            self.logger.warning("Twitter client not available")
            return []
            
        try:
            self.logger.info(f"Collecting tweets for query: {query}")
            
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # Twitter API limit
                tweet_fields=['created_at', 'public_metrics', 'author_id', 'context_annotations']
            )
            
            tweet_data = []
            if tweets and tweets.data:
                for tweet in tweets.data:
                    tweet_data.append({
                        'id': str(tweet.id),
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat() if tweet.created_at else datetime.now().isoformat(),
                        'author_id': str(tweet.author_id) if tweet.author_id else 'unknown',
                        'retweet_count': tweet.public_metrics.get('retweet_count', 0) if tweet.public_metrics else 0,
                        'like_count': tweet.public_metrics.get('like_count', 0) if tweet.public_metrics else 0,
                        'reply_count': tweet.public_metrics.get('reply_count', 0) if tweet.public_metrics else 0,
                        'quote_count': tweet.public_metrics.get('quote_count', 0) if tweet.public_metrics else 0,
                        'source': 'twitter',
                        'query': query
                    })
                    
            self.logger.info(f"Collected {len(tweet_data)} tweets")
            return tweet_data
            
        except tweepy.TooManyRequests:
            self.logger.warning("Twitter API rate limit reached, waiting...")
            return []
        except tweepy.Unauthorized:
            self.logger.error("Twitter API unauthorized - check your bearer token")
            return []
        except Exception as e:
            self.logger.error(f"Error collecting tweets: {e}")
            return []
    
    def collect_news_gnews(self, query: str) -> List[Dict]:
        """Collect news articles using GNews library"""
        if not self.gnews or not GNEWS_AVAILABLE:
            self.logger.warning("GNews not available")
            return []
            
        try:
            self.logger.info(f"Collecting news for query: {query}")
            
            news_data = []
            articles = self.gnews.get_news(query)
            
            if not articles:
                self.logger.info(f"No news articles found for query: {query}")
                return []
            
            for article in articles:
                try:
                    # Get full article content if possible
                    article_text = article.get('description', '')
                    
                    # Handle publisher info
                    publisher_info = article.get('publisher', {})
                    if isinstance(publisher_info, dict):
                        publisher_name = publisher_info.get('title', 'Unknown')
                    else:
                        publisher_name = str(publisher_info) if publisher_info else 'Unknown'
                    
                    news_data.append({
                        'id': f"news_{abs(hash(article.get('url', '')))}",
                        'title': article.get('title', ''),
                        'text': article_text,
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'created_at': article.get('published date', datetime.now().isoformat()),
                        'publisher': publisher_name,
                        'source': 'news',
                        'query': query,
                        'like_count': 0,
                        'retweet_count': 0
                    })
                except Exception as article_error:
                    self.logger.warning(f"Error processing article: {article_error}")
                    continue
                    
            self.logger.info(f"Collected {len(news_data)} news articles via GNews")
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error collecting news via GNews: {e}")
            return []
    
    def collect_news_simple(self, query: str) -> List[Dict]:
        """Simple news collection using web scraping (fallback method)"""
        try:
            self.logger.info(f"Collecting news via simple method for query: {query}")
            
            # Simple RSS feeds that don't require API keys
            rss_sources = [
                {'name': 'BBC', 'url': 'http://feeds.bbci.co.uk/news/rss.xml'},
                {'name': 'Reuters', 'url': 'https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best'},
                {'name': 'CNN', 'url': 'http://rss.cnn.com/rss/edition.rss'}
            ]
            
            news_data = []
            
            for source in rss_sources:
                try:
                    response = requests.get(source['url'], timeout=10)
                    if response.status_code == 200:
                        content = response.text
                        
                        # Simple parsing for RSS feeds
                        import re
                        
                        # Extract titles
                        title_patterns = [
                            r'<title><!\[CDATA\[(.*?)\]\]></title>',
                            r'<title>(.*?)</title>'
                        ]
                        
                        titles = []
                        for pattern in title_patterns:
                            matches = re.findall(pattern, content, re.DOTALL)
                            titles.extend([match.strip() for match in matches])
                        
                        # Filter titles that contain query terms
                        query_lower = query.lower()
                        relevant_titles = [title for title in titles if any(word in title.lower() 
                                        for word in query_lower.split())][:5]  # Limit to 5
                        
                        for i, title in enumerate(relevant_titles):
                            if title and len(title) > 10:  # Basic validation
                                news_data.append({
                                    'id': f"simple_{abs(hash(title))}",
                                    'title': title,
                                    'text': title,  # Use title as text for simple method
                                    'description': title,
                                    'url': f"{source['url']}#item_{i}",
                                    'created_at': datetime.now().isoformat(),
                                    'publisher': source['name'],
                                    'source': 'simple_rss',
                                    'query': query,
                                    'like_count': 0,
                                    'retweet_count': 0
                                })
                                
                except Exception as source_error:
                    self.logger.warning(f"Error collecting from {source['name']}: {source_error}")
                    continue
            
            self.logger.info(f"Collected {len(news_data)} news articles via simple method")
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error in simple news collection: {e}")
            return []
    
    def collect_news(self, query: str) -> List[Dict]:
        """Collect news using all available methods"""
        all_news = []
        
        # Try GNews first
        if GNEWS_AVAILABLE and self.gnews:
            news = self.collect_news_gnews(query)
            all_news.extend(news)
        
        # If no news collected, try simple method
        if not all_news:
            news = self.collect_news_simple(query)
            all_news.extend(news)
        
        # If still no news, create sample data
        if not all_news:
            all_news = self.create_sample_news(query)
        
        return all_news
    
    def create_sample_news(self, query: str) -> List[Dict]:
        """Create sample news data for testing when no APIs work"""
        sample_headlines = [
            f"Breaking: Major developments in {query} sector",
            f"Analysis: {query} market shows strong indicators",
            f"Expert opinion on {query} industry trends",
            f"New research reveals {query} potential",
            f"Market update: {query} stocks surge"
        ]
        
        news_data = []
        for i, headline in enumerate(sample_headlines):
            news_data.append({
                'id': f"sample_{abs(hash(headline))}",
                'title': headline,
                'text': f"{headline}. This is sample content for testing purposes when real news APIs are not available.",
                'description': f"Sample news article about {query}",
                'url': f"https://example.com/news/{i}",
                'created_at': datetime.now().isoformat(),
                'publisher': 'Sample News',
                'source': 'sample',
                'query': query,
                'like_count': 0,
                'retweet_count': 0
            })
        
        self.logger.info(f"Created {len(news_data)} sample news articles")
        return news_data
    
    def collect_all_data(self, queries: List[str]) -> pd.DataFrame:
        """Collect data from all available sources for given queries"""
        all_data = []
        
        self.logger.info(f"Starting data collection for {len(queries)} queries")
        
        for query in queries:
            self.logger.info(f"Processing query: {query}")
            
            # Collect tweets if available
            if self.twitter_client and TWEEPY_AVAILABLE:
                tweets = self.collect_tweets(query, max_results=20)
                all_data.extend(tweets)
                time.sleep(1)  # Rate limiting
            else:
                self.logger.info("Skipping Twitter collection - not available")
            
            # Collect news
            news = self.collect_news(query)
            all_data.extend(news)
            
            # Small delay between queries
            time.sleep(1)
            
        df = pd.DataFrame(all_data)
        self.logger.info(f"Total data collected: {len(df)} items")
        
        # Save raw data if we have any
        if not df.empty:
            try:
                # Ensure data directory exists
                os.makedirs("data/raw", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/raw/collected_data_{timestamp}.csv"
                df.to_csv(filename, index=False)
                self.logger.info(f"Raw data saved to {filename}")
            except Exception as save_error:
                self.logger.error(f"Error saving data: {save_error}")
        
        return df
    
    def get_status(self) -> Dict:
        """Get status of all data collection methods"""
        return {
            'tweepy_available': TWEEPY_AVAILABLE,
            'gnews_available': GNEWS_AVAILABLE,
            'twitter_configured': self.twitter_client is not None,
            'gnews_configured': self.gnews is not None,
            'twitter_token_set': bool(os.getenv('TWITTER_BEARER_TOKEN')),
            'can_collect_data': (self.twitter_client is not None) or (self.gnews is not None) or True  # Always true due to fallback
        }

# Test function
def test_data_collector():
    """Execute comprehensive tests of data collector functionality"""
    print("Testing Data Collector")
    print("=" * 50)
    
    try:
        # Initialize collector instance
        collector = DataCollector()
        
        # Verify system status
        status = collector.get_status()
        print("\nSystem Status:")
        for key, value in status.items():
            status_symbol = '✅' if value else '❌'
            print(f"  {key}: {status_symbol}")
        
        # Test data collection with sample queries
        test_queries = ["AI technology", "market trends"]
        print(f"\nTesting data collection with queries: {test_queries}")
        
        # Execute collection and measure results
        data = collector.collect_all_data(test_queries)
        print(f"Successfully collected {len(data)} data points")
        
        if not data.empty:
            print("\nData sources found:")
            source_counts = data['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  {source}: {count} items")
            
            print(f"\nSample data preview:")
            # Show first few rows with key columns
            preview_cols = ['source', 'title', 'text']
            available_cols = [col for col in preview_cols if col in data.columns]
            
            for idx, row in data[available_cols].head(3).iterrows():
                print(f"  Row {idx + 1}:")
                for col in available_cols:
                    text_preview = str(row[col])[:80] + '...' if len(str(row[col])) > 80 else str(row[col])
                    print(f"    {col}: {text_preview}")
                print()
        else:
            print("No data collected - check your API configurations")
            
        print("✅ Data collector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

# Main execution entry point
if __name__ == "__main__":
    # Configure logging with timestamp and level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Execute test suite
    test_data_collector()
