"""
Predictive Models Module - ML-based trend prediction system

Provides:
- Feature engineering for time series data
- Model training and validation
- Trend prediction and confidence scoring
- Model persistence and loading
"""

# Core data processing and ML imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

class TrendPredictor:
    """Machine learning model for trend prediction"""
    
    def __init__(self):
        # Initialize random forest with 100 trees for robust predictions
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # Feature scaling for normalized input
        self.scaler = StandardScaler()
        # Configure logging
        self.logger = logging.getLogger(__name__)
        # Track model training status
        self.is_trained = False
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw data into ML-ready features
        
        Args:
            data: Raw DataFrame with temporal and textual data
            
        Returns:
            DataFrame of engineered features
        """
        features = pd.DataFrame()
        
        # Extract temporal features from timestamps
        if 'created_at' in data.columns:
            data['created_at'] = pd.to_datetime(data['created_at'])
            features['hour'] = data['created_at'].dt.hour           # Hour of day (0-23)
            features['day_of_week'] = data['created_at'].dt.dayofweek  # Day of week (0-6)
            features['month'] = data['created_at'].dt.month         # Month (1-12)
        
        # Convert sentiment categories to binary features
        if 'sentiment' in data.columns:
            features['sentiment_positive'] = (data['sentiment'] == 'positive').astype(int)
            features['sentiment_negative'] = (data['sentiment'] == 'negative').astype(int)
        
        # Add confidence scores if available
        if 'confidence' in data.columns:
            features['confidence'] = data['confidence'].fillna(0)
        
        # Include engagement metrics
        if 'like_count' in data.columns:
            features['like_count'] = data['like_count'].fillna(0)
        if 'retweet_count' in data.columns:
            features['retweet_count'] = data['retweet_count'].fillna(0)
        
        # Add text-based features
        if 'text' in data.columns:
            features['text_length'] = data['text'].str.len().fillna(0)
        
        # Ensure no missing values
        return features.fillna(0)
    
    def train_model(self, historical_data: pd.DataFrame, target_column: str):
        """Train the predictive model"""
        try:
            features = self.prepare_features(historical_data)
            
            if features.empty or target_column not in historical_data.columns:
                raise ValueError("Insufficient data for training")
            
            X = features
            y = historical_data[target_column].fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            predictions = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.logger.info(f"Model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            return {"mse": mse, "r2": r2}
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            return None
    
    def predict_trends(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Predict future trends based on current data"""
        if not self.is_trained:
            self.logger.warning("Model not trained yet")
            return pd.DataFrame()
        
        try:
            features = self.prepare_features(current_data)
            
            if features.empty:
                return pd.DataFrame()
            
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            
            result = current_data.copy()
            result['predicted_trend'] = predictions
            result['prediction_confidence'] = np.abs(predictions)  # Simple confidence measure
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return pd.DataFrame()
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            joblib.dump(self.model, 'data/models/trend_predictor_model.pkl')
            joblib.dump(self.scaler, 'data/models/trend_predictor_scaler.pkl')
            self.logger.info("Model saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            self.model = joblib.load('data/models/trend_predictor_model.pkl')
            self.scaler = joblib.load('data/models/trend_predictor_scaler.pkl')
            self.is_trained = True
            self.logger.info("Model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
