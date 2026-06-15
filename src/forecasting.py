import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

class PortDemandForecaster:
    def __init__(self, seed=42):
        self.model = GradientBoostingRegressor(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=seed
        )
        self.feature_names = []
        self.is_trained = False

    def create_features(self, df):
        """
        Engineers temporal, lag, rolling, and operational features for ML modeling.
        """
        df_feats = df.copy()
        
        # Lag features (Target: Total_Demand_kW)
        df_feats['Lag_1h'] = df_feats['Total_Demand_kW'].shift(1)
        df_feats['Lag_2h'] = df_feats['Total_Demand_kW'].shift(2)
        df_feats['Lag_24h'] = df_feats['Total_Demand_kW'].shift(24)
        
        # Rolling features (using shift(1) to avoid data leakage)
        df_feats['Rolling_Mean_3h'] = df_feats['Total_Demand_kW'].shift(1).rolling(window=3).mean()
        df_feats['Rolling_Std_3h'] = df_feats['Total_Demand_kW'].shift(1).rolling(window=3).std()
        df_feats['Rolling_Mean_24h'] = df_feats['Total_Demand_kW'].shift(1).rolling(window=24).mean()
        
        # Environmental/Renewable features (assumed forecastable or known)
        df_feats['Solar_Forecast_kW'] = df_feats['Solar_Gen_kW']
        df_feats['Wind_Forecast_kW'] = df_feats['Wind_Gen_kW']
        
        # Drop rows with NaN (from lag/rolling calculations)
        df_feats = df_feats.dropna()
        
        return df_feats

    def train(self, train_df):
        """
        Prepares training data and fits the Gradient Boosting Regressor.
        """
        train_processed = self.create_features(train_df)
        
        self.feature_names = [
            'HourOfDay', 'DayOfWeek', 
            'Lag_1h', 'Lag_2h', 'Lag_24h', 
            'Rolling_Mean_3h', 'Rolling_Std_3h', 'Rolling_Mean_24h',
            'Vessel_Count', 'Cargo_Load_Tons',
            'Solar_Forecast_kW', 'Wind_Forecast_kW'
        ]
        
        X_train = train_processed[self.feature_names]
        y_train = train_processed['Total_Demand_kW']
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return train_processed

    def predict(self, test_df, train_df):
        """
        Generates predictions for the test dataset. To construct lag features at the start of
        the test period, it prepends the end of the training data.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting.")
            
        # Combine last 48 hours of training with test data to compute lag features correctly
        prepended_df = pd.concat([train_df.tail(48), test_df], ignore_index=True)
        prepended_processed = self.create_features(prepended_df)
        
        # Extract the rows corresponding to the original test dataset
        # The first 48 rows of prepended_df are from train_df. Due to lagging, the first 24 rows
        # of prepended_processed might still be dropped or be part of training data.
        # Since we concatenated train_df.tail(48) and test_df:
        # test_df starts at index 48 in prepended_df.
        # In prepended_processed, we filter for hours that match test_df's original hours.
        test_hours = test_df['Hour'].values
        test_processed = prepended_processed[prepended_processed['Hour'].isin(test_hours)].copy()
        
        X_test = test_processed[self.feature_names]
        
        # Generate predictions
        test_processed['Forecast_Demand_kW'] = self.model.predict(X_test)
        
        # Calculate metrics
        y_actual = test_processed['Total_Demand_kW'].values
        y_pred = test_processed['Forecast_Demand_kW'].values
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / np.maximum(1.0, y_actual))) * 100
        
        metrics = {
            'MAE_kW': mae,
            'RMSE_kW': rmse,
            'MAPE_percent': mape
        }
        
        # Calculate feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return test_processed, metrics, feature_importance_df
