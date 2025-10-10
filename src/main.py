"""
Strategic Intelligence Platform - Main Orchestrator

This module provides:
- Platform initialization and coordination
- Component lifecycle management
- Continuous and batch processing modes
- Error handling and logging
"""

# Core system imports
import asyncio
import logging
import os
from dotenv import load_dotenv

# Import platform components
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from trend_detector import TrendDetector
from alert_system import AlertSystem
from data_pipeline import DataPipeline

# Load environment variables from .env file
load_dotenv()

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,  # Set minimum logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler('strategic_intelligence.log'),  # Persistent log file
        logging.StreamHandler()  # Console output
    ]
)

# Create module-level logger
logger = logging.getLogger(__name__)

class StrategicIntelligencePlatform:
    """Main platform orchestrator class"""
    
    def __init__(self):
        """Initialize platform components"""
        # Create component instances
        self.data_collector = DataCollector()           # Data collection system
        self.sentiment_analyzer = SentimentAnalyzer()   # NLP analysis engine
        self.trend_detector = TrendDetector()          # Trend detection system
        self.alert_system = AlertSystem()              # Alert management
        self.data_pipeline = DataPipeline(            # Data processing pipeline
            self.data_collector, 
            self.sentiment_analyzer
        )
        
        # Define core monitoring topics
        self.monitoring_queries = [
            "AI technology trends",      # Track AI developments
            "competitor analysis",       # Monitor competition
            "market sentiment",          # Overall market mood
            "emerging technologies",     # New tech trends
            "industry disruption"        # Disruptive changes
        ]
    
    async def run_continuous_monitoring(self):
        """Execute platform in continuous monitoring mode"""
        logger.info("Starting Strategic Intelligence Platform...")
        
        try:
            # Start real-time data processing
            await self.data_pipeline.process_real_time_data(self.monitoring_queries)
            
        except KeyboardInterrupt:
            # Handle graceful shutdown
            logger.info("Shutting down platform...")
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Platform error: {e}")
    
    def run_batch_analysis(self):
        """Execute one-time batch analysis of data"""
        logger.info("Running batch analysis...")
        
        try:
            # Collect data from all sources
            raw_data = self.data_collector.collect_all_data(self.monitoring_queries)
            
            # Check for empty dataset
            if raw_data.empty:
                logger.warning("No data collected")
                return
            
            # Perform sentiment analysis
            texts = raw_data['text'].fillna('').tolist()
            sentiment_results = self.sentiment_analyzer.batch_analyze(texts[:10])  # Limited sample
            
            # Execute trend detection
            trend_results = self.trend_detector.detect_emerging_trends(raw_data)
            
            # Generate and process alerts
            self.alert_system.process_alerts(sentiment_results)
            
            # Log analysis results
            logger.info(f"Batch analysis complete. Processed {len(sentiment_results)} items")
            logger.info(f"Detected {len(trend_results.get('trends', []))} trends")
            
        except Exception as e:
            logger.error(f"Batch analysis error: {e}")

def main():
    """Main entry point for platform execution"""
    # Initialize platform instance
    platform = StrategicIntelligencePlatform()
    
    # Determine execution mode from environment
    mode = os.getenv('RUN_MODE', 'batch')  # Default to batch mode
    
    # Execute platform in appropriate mode
    if mode == 'continuous':
        asyncio.run(platform.run_continuous_monitoring())
    else:
        platform.run_batch_analysis()

# Script entry point
if __name__ == "__main__":
    main()
