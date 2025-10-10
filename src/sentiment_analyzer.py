"""
sentiment_analyzer.py - LLM-based sentiment analysis using OpenAI GPT

This module provides:
- Sentiment analysis using OpenAI's GPT models
- Fallback keyword-based analysis
- Batch processing capabilities
- DataFrame integration
"""

# System imports
import os                  # For environment variables and file operations
import json               # For parsing OpenAI responses
import time              # For rate limiting and timestamps
import logging           # For error and operation logging
from typing import Dict, List, Optional  # Type hints
import pandas as pd      # For DataFrame operations
from dotenv import load_dotenv  # Environment variable management

# Import custom OpenAI retry helper
from src.utils.openai_helpers import call_openai_with_retry  # Handles API retries

# Initialize OpenAI with error handling
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI package not installed. Run: pip install openai")

# Load configuration from .env file
load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_client()
        
        # Fallback sentiment analysis using simple keyword matching
        self.positive_keywords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'positive', 'growth', 'increase', 'profit', 'success', 'win',
            'bullish', 'optimistic', 'upward', 'rising', 'boom', 'strong'
        ]
        
        self.negative_keywords = [
            'bad', 'terrible', 'awful', 'horrible', 'negative', 'decline',
            'decrease', 'loss', 'fail', 'crash', 'bearish', 'pessimistic',
            'downward', 'falling', 'recession', 'weak', 'drop', 'plunge'
        ]
        
    def setup_client(self):
        """Initialize OpenAI client with comprehensive error handling"""
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI package not available")
            self.client = None
            return
            
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            self.logger.warning("OpenAI API key not found or using placeholder value")
            self.client = None
            return
            
        try:
            # Initialize OpenAI client (works with openai>=1.0.0)
            self.client = openai.OpenAI(api_key=api_key)
            
            # Test the connection with a simple request
            self._test_connection()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _test_connection(self):
        """Test OpenAI API connection"""
        try:
            # Make a minimal test request
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            self.logger.info("OpenAI API connection successful")
        except openai.AuthenticationError:
            self.logger.error("OpenAI API authentication failed - check your API key")
            self.client = None
        except openai.RateLimitError:
            self.logger.warning("OpenAI API rate limit reached during connection test")
            # Don't disable client for rate limits
        except Exception as e:
            self.logger.error(f"OpenAI API connection test failed: {e}")
            self.client = None
    
    def analyze_sentiment(self, text: str, context: str = "general") -> Dict:
        """Analyze sentiment of given text using OpenAI GPT or fallback method"""
        if not text or len(text.strip()) == 0:
            return self._get_fallback_result("Empty text provided")
        
        # Clean text
        cleaned_text = str(text).strip()[:2000]  # Limit text length
        
        # Try OpenAI analysis first
        if self.client:
            result = self._analyze_with_openai(cleaned_text, context)
            if not result.get('error', False):
                return result
        
        # Fallback to simple keyword-based analysis
        self.logger.info("Using fallback sentiment analysis")
        return self._analyze_with_keywords(cleaned_text)
    
    def _analyze_with_openai(self, text: str, context: str) -> Dict:
        """Analyze sentiment using OpenAI GPT"""
        try:
            prompt = self._create_analysis_prompt(text, context)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert financial sentiment analyst. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.1,
                timeout=30
            )
            
            # Extract and parse response
            response_text = response.choices[0].message.content.strip()
            result = self._parse_openai_response(response_text)
            
            # Add metadata
            result['original_text'] = text[:200] + '...' if len(text) > 200 else text
            result['analysis_method'] = 'openai'
            result['analysis_timestamp'] = time.time()
            
            return result
            
        except openai.AuthenticationError:
            self.logger.error("OpenAI authentication failed")
            return self._get_fallback_result("Authentication failed")
            
        except openai.RateLimitError:
            self.logger.warning("OpenAI rate limit reached, waiting...")
            time.sleep(1)  # Wait 1 minute
            return self._get_fallback_result("Rate limit reached")
            
        except openai.APITimeoutError:
            self.logger.warning("OpenAI API timeout")
            return self._get_fallback_result("API timeout")
            
        except openai.APIConnectionError:
            self.logger.error("OpenAI API connection error")
            return self._get_fallback_result("Connection error")
            
        except Exception as e:
            self.logger.error(f"OpenAI analysis error: {e}")
            return self._get_fallback_result(f"OpenAI error: {str(e)}")
    
    def _create_analysis_prompt(self, text: str, context: str) -> str:
        """Create the analysis prompt for OpenAI"""
        return f"""
Analyze the sentiment of this text for {context} and market intelligence:

Text: "{text}"

Respond with ONLY this JSON format:
{{
    "sentiment": "positive",
    "confidence": 0.85,
    "emotions": ["optimism", "excitement"],
    "market_relevance": "high",
    "reasoning": "Brief explanation",
    "keywords": ["key1", "key2"]
}}

Rules:
- sentiment: "positive", "negative", or "neutral"  
- confidence: number between 0.0 and 1.0
- emotions: list of 1-3 emotions
- market_relevance: "high", "medium", or "low"
- reasoning: max 100 characters
- keywords: 2-5 important terms
"""
    
    def _parse_openai_response(self, response_text: str) -> Dict:
        """Parse OpenAI response and handle JSON issues"""
        try:
            # Clean response text
            clean_text = response_text.strip()
            
            # Remove markdown code blocks if present
            #if "```
            #   clean_text = clean_text.split("```json").split("```
            #elif "```" in clean_text:
            #    clean_text = clean_text.split("```
            
            # Try parsing JSON
            try:
                result = json.loads(clean_text)
            except json.JSONDecodeError:
                # Try fixing common issues
                clean_text = clean_text.replace("'", '"')  # Replace single quotes
                clean_text = clean_text.replace('True', 'true').replace('False', 'false')
                result = json.loads(clean_text)
            
            # Validate and clean the result
            return self._validate_openai_result(result)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return self._get_fallback_result("JSON parsing failed")
        except Exception as e:
            self.logger.error(f"Response parsing error: {e}")
            return self._get_fallback_result(f"Parsing error: {str(e)}")
    
    def _validate_openai_result(self, result: Dict) -> Dict:
        """Validate and standardize OpenAI analysis result"""
        validated = {}
        
        # Validate sentiment
        sentiment = result.get('sentiment', '').lower()
        if sentiment in ['positive', 'negative', 'neutral']:
            validated['sentiment'] = sentiment
        else:
            validated['sentiment'] = 'neutral'
        
        # Validate confidence
        try:
            confidence = float(result.get('confidence', 0))
            validated['confidence'] = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            validated['confidence'] = 0.5
        
        # Validate emotions
        emotions = result.get('emotions', [])
        if isinstance(emotions, list):
            validated['emotions'] = [str(e) for e in emotions[:3]]  # Max 3 emotions
        else:
            validated['emotions'] = []
        
        # Validate market relevance
        market_relevance = result.get('market_relevance', '').lower()
        if market_relevance in ['high', 'medium', 'low']:
            validated['market_relevance'] = market_relevance
        else:
            validated['market_relevance'] = 'low'
        
        # Validate reasoning
        reasoning = result.get('reasoning', '')
        validated['reasoning'] = str(reasoning)[:200] if reasoning else 'No reasoning provided'
        
        # Validate keywords
        keywords = result.get('keywords', [])
        if isinstance(keywords, list):
            validated['keywords'] = [str(k) for k in keywords[:5]]  # Max 5 keywords
        else:
            validated['keywords'] = []
        
        validated['error'] = False
        return validated
    
    def _analyze_with_keywords(self, text: str) -> Dict:
        """Fallback sentiment analysis using keyword matching"""
        text_lower = text.lower()
        
        # Count positive and negative keywords
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.8, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.8, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.3
        
        # Extract found keywords
        found_keywords = [word for word in self.positive_keywords + self.negative_keywords 
                        if word in text_lower]
        
        # Determine market relevance based on business keywords
        business_keywords = ['market', 'stock', 'business', 'company', 'financial', 'economic', 'industry']
        market_relevance = 'high' if any(word in text_lower for word in business_keywords) else 'low'
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'emotions': [sentiment] if sentiment != 'neutral' else ['neutral'],
            'market_relevance': market_relevance,
            'reasoning': f'Keyword-based analysis: {positive_count} positive, {negative_count} negative keywords',
            'keywords': found_keywords[:5],
            'original_text': text[:200] + '...' if len(text) > 200 else text,
            'analysis_method': 'keywords',
            'analysis_timestamp': time.time(),
            'error': False
        }
    
    def _get_fallback_result(self, reason: str) -> Dict:
        """Return fallback result when analysis fails"""
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'emotions': [],
            'market_relevance': 'low',
            'reasoning': f'Analysis failed: {reason}',
            'keywords': [],
            'analysis_method': 'fallback',
            'analysis_timestamp': time.time(),
            'error': True
        }
    
    def batch_analyze(self, texts: List[str], context: str = "general") -> List[Dict]:
        """Analyze sentiment for multiple texts with proper rate limiting"""
        if not texts:
            return []
        
        results = []
        total_texts = len(texts)
        
        self.logger.info(f"Starting batch analysis for {total_texts} texts")
        
        for i, text in enumerate(texts):
            try:
                # Progress logging
                if i > 0 and i % 10 == 0:
                    self.logger.info(f"Processed {i}/{total_texts} texts")
                
                result = self.analyze_sentiment(text, context)
                results.append(result)
                
                # Rate limiting - only if using OpenAI
                if self.client and not result.get('error', False):
                    time.sleep(0.5)  # 500ms between OpenAI requests
                    
            except Exception as e:
                self.logger.error(f"Error processing text {i}: {e}")
                results.append(self._get_fallback_result(f"Processing error: {str(e)}"))
        
        self.logger.info(f"Completed batch analysis: {len(results)} results")
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Analyze sentiment for texts in a DataFrame"""
        if df.empty:
            self.logger.warning("DataFrame is empty")
            return df
        
        if text_column not in df.columns:
            self.logger.error(f"Column '{text_column}' not found in DataFrame")
            return df
        
        self.logger.info(f"Analyzing sentiment for {len(df)} items in DataFrame")
        
        # Extract texts and handle missing values
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Analyze sentiment
        results = self.batch_analyze(texts)
        
        # Create results DataFrame
        sentiment_df = pd.DataFrame(results)
        
        # Merge with original DataFrame
        result_df = df.reset_index(drop=True)
        for col in sentiment_df.columns:
            result_df[f'sentiment_{col}'] = sentiment_df[col]
        
        return result_df
    
    def get_sentiment_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics from sentiment analysis results"""
        if not results:
            return {'error': 'No results to summarize'}
        
        # Extract values safely
        sentiments = [r.get('sentiment', 'neutral') for r in results]
        confidences = [r.get('confidence', 0) for r in results if isinstance(r.get('confidence'), (int, float))]
        market_relevances = [r.get('market_relevance', 'low') for r in results]
        errors = [r for r in results if r.get('error', False)]
        
        # Calculate distributions
        sentiment_dist = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        relevance_dist = {
            'high': market_relevances.count('high'),
            'medium': market_relevances.count('medium'),
            'low': market_relevances.count('low')
        }
        
        # Calculate averages
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        summary = {
            'total_analyzed': len(results),
            'sentiment_distribution': sentiment_dist,
            'market_relevance_distribution': relevance_dist,
            'average_confidence': round(avg_confidence, 3),
            'confidence_range': {
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            },
            'error_count': len(errors),
            'success_rate': round((len(results) - len(errors)) / len(results), 3) if results else 0,
            'analysis_methods': {
                'openai': len([r for r in results if r.get('analysis_method') == 'openai']),
                'keywords': len([r for r in results if r.get('analysis_method') == 'keywords']),
                'fallback': len([r for r in results if r.get('analysis_method') == 'fallback'])
            }
        }
        
        return summary
    
    def is_available(self) -> bool:
        """Check if sentiment analyzer is properly configured"""
        return self.client is not None or OPENAI_AVAILABLE

# Test function with comprehensive error handling
def test_sentiment_analyzer():
    """Test the sentiment analyzer with various scenarios"""
    print("Testing Sentiment Analyzer")
    print("=" * 50)
    
    analyzer = SentimentAnalyzer()
    
    # Check availability
    print(f"OpenAI Available: {OPENAI_AVAILABLE}")
    print(f"Analyzer Available: {analyzer.is_available()}")
    print(f"OpenAI Client Status: {'Connected' if analyzer.client else 'Not connected'}")
    
    # Test texts with various scenarios
    test_texts = [
        "The stock market is showing incredible growth this quarter with record profits!",
        "I'm very concerned about the economic downturn affecting our business operations.",
        "The company reported standard quarterly results with no significant changes.",
        "Amazing breakthrough in AI technology will revolutionize the entire industry!",
        "",  # Empty text
        "Short text",
        "A" * 3000,  # Very long text
    ]
    
    print(f"\nTesting with {len(test_texts)} different text samples...")
    print("-" * 50)
    
    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            result = analyzer.analyze_sentiment(text)
            
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Market Relevance: {result['market_relevance']}")
            print(f"  Method: {result.get('analysis_method', 'unknown')}")
            print(f"  Keywords: {result['keywords'][:3]}")  # Show first 3 keywords
            print(f"  Error: {result.get('error', False)}")
        
        # Test batch analysis
        print(f"\n{'='*50}")
        print("Testing batch analysis...")
        
        batch_results = analyzer.batch_analyze(test_texts[:3])
        summary = analyzer.get_sentiment_summary(batch_results)
        
        print(f"Batch Results Summary:")
        print(f"  Total analyzed: {summary['total_analyzed']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Average confidence: {summary['average_confidence']:.2f}")
        print(f"  Sentiment distribution: {summary['sentiment_distribution']}")
        print(f"  Analysis methods used: {summary['analysis_methods']}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_sentiment_analyzer()