import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import requests
import warnings
warnings.filterwarnings('ignore')

class DutchGPPredictor:
    def __init__(self, model_dir, data_path, weather_api_key):
        self.model_dir = model_dir
        self.data_path = data_path
        self.weather_api_key = weather_api_key
        self.model = None
        self.scaler = None
        self.features = None
        self.historical_data = None
        
    def fetch_weather_forecast(self, lat=52.3888, lon=4.5409):
        """Fetch weather forecast for Circuit Zandvoort"""
        print("🌤️ Fetching weather forecast for Dutch GP...")
        url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={self.weather_api_key}&units=metric"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Find forecast closest to August 31, 2025, 15:00 (Dutch GP typical start time)
            forecast = next((item for item in data['list'] if '2025-08-31 15:00' in item['dt_txt']), data['list'][0])
            
            return {
                'temperature': forecast['main']['temp'],
                'humidity': forecast['main']['humidity'],
                'rain_probability': forecast.get('pop', 0),
                'weather_condition': 1 if forecast['weather'][0]['main'] in ['Rain', 'Drizzle', 'Thunderstorm'] else 0,
                'wind_speed': forecast['wind']['speed']
            }
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return {
                'temperature': 18.0,
                'humidity': 70.0,
                'rain_probability': 0.3,
                'weather_condition': 0,
                'wind_speed': 4.0
            }
    
    def load_model_and_data(self):
        """Load the enhanced model, scaler, and historical data"""
        try:
            self.model = joblib.load(f'{self.model_dir}/enhanced_xgb_model.pkl')
            self.scaler = joblib.load(f'{self.model_dir}/feature_scaler.pkl')
            with open(f'{self.model_dir}/features.txt', 'r') as f:
                self.features = [line.strip() for line in f.readlines()]
            self.historical_data = pd.read_csv(self.data_path)
            self.historical_data = self.historical_data[self.historical_data['season'] >= 2021].copy()
            print(f'✅ Model, scaler, features, and data (2021-2025) loaded successfully')
            return True
        except Exception as e:
            print(f'❌ Error loading model/data: {e}')
            return False
    
    def create_advanced_features(self, df):
        """Create advanced features with weather and qualifying delta"""
        print("🔧 Creating advanced features...")
        
        df = df.sort_values(['season', 'round', 'position']).reset_index(drop=True)
        
        # Helper function to convert time strings (MM:SS.s) to seconds
        def time_to_seconds(time_str):
            try:
                if pd.isna(time_str) or not isinstance(time_str, str):
                    return np.nan
                parts = time_str.strip().split(':')
                if len(parts) == 2:
                    minutes, seconds = parts
                    return float(minutes) * 60 + float(seconds)
                return float(time_str)
            except (ValueError, TypeError):
                return np.nan
        
        # Convert fastest_lap_time to seconds
        df['fastest_lap_time_seconds'] = df['fastest_lap_time'].apply(time_to_seconds)
        
        # Existing features
        for window in [3, 5, 10]:
            df[f'driver_avg_pos_last_{window}'] = df.groupby('driver_id')['position'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            df[f'driver_points_last_{window}'] = df.groupby('driver_id')['points'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
        
        for window in [5, 10]:
            df[f'constructor_avg_pos_last_{window}'] = df.groupby('constructor_id')['position'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
            df[f'constructor_points_last_{window}'] = df.groupby('constructor_id')['points'].transform(
                lambda x: x.rolling(window, min_periods=1).mean().shift(1)
            )
        
        df['driver_circuit_avg_pos'] = df.groupby(['driver_id', 'circuit_id'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['constructor_circuit_avg_pos'] = df.groupby(['constructor_id', 'circuit_id'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['races_completed_in_season'] = df.groupby(['season', 'driver_id']).cumcount()
        df['season_points_cumulative'] = df.groupby(['season', 'driver_id'])['points'].cumsum().shift(1)
        df['season_avg_pos'] = df.groupby(['season', 'driver_id'])['position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        df['grid_to_finish_improvement'] = df.groupby('driver_id').apply(
            lambda x: (x['grid'] - x['position']).rolling(5, min_periods=1).mean().shift(1)
        ).reset_index(level=0, drop=True)
        
        df['avg_grid_pos_last_5'] = df.groupby('driver_id')['grid'].transform(
            lambda x: x.rolling(5, min_periods=1).mean().shift(1)
        )
        
        df['position_std_last_5'] = df.groupby('driver_id')['position'].transform(
            lambda x: x.rolling(5, min_periods=1).std().shift(1)
        )
        
        df['dnf_rate_last_10'] = df.groupby('driver_id')['status'].transform(
            lambda x: (x != 'Finished').rolling(10, min_periods=1).mean().shift(1)
        )
        
        df['fastest_lap_points_last_5'] = df.groupby('driver_id').apply(
            lambda x: (x['fastest_lap_rank'] == 1).rolling(5, min_periods=1).sum().shift(1)
        ).reset_index(level=0, drop=True)
        
        df['teammate_avg_pos'] = df.groupby(['season', 'constructor_id', 'round'])['position'].transform(
            lambda x: x.mean() if len(x) > 1 else x.iloc[0]
        )
        
        df['race_pace_variance'] = df.groupby(['season', 'round'])['time_millis'].transform('std')
        
        # New features (only include if supported by training)
        df['qualifying_delta'] = df.groupby(['season', 'round'])['fastest_lap_time_seconds'].transform(
            lambda x: (x - x.min()) / x.min() if x.notnull().all() else x.fillna(x.mean())
        )
        
        df['avg_pit_stops_last_5'] = df.groupby('driver_id')['laps'].transform(
            lambda x: (x < x.max()).rolling(5, min_periods=1).mean().shift(1)
        )
        
        df['wet_race_performance'] = df.groupby('driver_id').apply(
            lambda x: x[x['status'].str.contains('Rain|Wet', case=False, na=False)]['position'].mean()
        ).reindex(df.index, fill_value=10.5)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col.endswith('_pos') or col.endswith('position'):
                df[col] = df[col].fillna(10.5)
            elif col.endswith('_points'):
                df[col] = df[col].fillna(0)
            elif col.endswith('_std') or col.endswith('_delta'):
                df[col] = df[col].fillna(5)
            elif col.endswith('_rate'):
                df[col] = df[col].fillna(0.1)
            else:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def prepare_race_data(self):
        """Prepare 2025 Dutch GP data based on 2021-2025 performance"""
        print("🏁 Preparing 2025 Dutch GP data...")
        
        # Filter 2025 season data
        season_2025_data = self.historical_data[self.historical_data['season'] == 2025].copy()
        
        # Get historical Zandvoort performance (2021-2024)
        zandvoort_data = self.historical_data[
            (self.historical_data['circuit_id'] == 'zandvoort') & 
            (self.historical_data['season'].between(2021, 2024))
        ].copy()
        
        if season_2025_data.empty:
            print("❌ No 2025 data found in CSV. Using Zandvoort historical data.")
            driver_performance = zandvoort_data.groupby('driver_id').agg({
                'grid': 'mean',
                'position': 'mean',
                'points': 'sum',
                'constructor_id': 'last',
                'fastest_lap_rank': 'mean'
            }).reset_index()
        else:
            # Aggregate 2025 season performance
            driver_performance = season_2025_data.groupby('driver_id').agg({
                'grid': 'mean',
                'position': 'mean',
                'points': 'sum',
                'constructor_id': 'last',
                'fastest_lap_rank': 'mean'
            }).reset_index()
            
            # Incorporate Zandvoort-specific performance (2021-2024)
            zandvoort_performance = zandvoort_data.groupby('driver_id').agg({
                'grid': 'mean',
                'position': 'mean',
                'points': 'sum'
            }).reset_index().rename(columns={
                'grid': 'zandvoort_grid_avg',
                'position': 'zandvoort_position_avg',
                'points': 'zandvoort_points_sum'
            })
            
            driver_performance = driver_performance.merge(zandvoort_performance, on='driver_id', how='left')
            driver_performance['zandvoort_grid_avg'] = driver_performance['zandvoort_grid_avg'].fillna(10.5)
            driver_performance['zandvoort_position_avg'] = driver_performance['zandvoort_position_avg'].fillna(10.5)
            driver_performance['zandvoort_points_sum'] = driver_performance['zandvoort_points_sum'].fillna(0)
            
            # Combine season and Zandvoort performance for grid ranking
            driver_performance['combined_score'] = (
                0.7 * (driver_performance['points'] / (driver_performance['position'] + 1)) +
                0.3 * (driver_performance['zandvoort_points_sum'] / (driver_performance['zandvoort_position_avg'] + 1))
            )
        
        # Sort drivers by performance to estimate Dutch GP grid
        driver_performance = driver_performance.sort_values('combined_score' if 'combined_score' in driver_performance.columns else 'points', ascending=False)
        
        # Select top 20 drivers
        drivers = driver_performance['driver_id'].head(20).tolist()
        if len(drivers) < 20:
            print(f"Warning: Only {len(drivers)} drivers found. Filling with defaults.")
            default_drivers = [
                'max_verstappen', 'norris', 'piastri', 'leclerc', 'hamilton', 'russell', 
                'sainz', 'alonso', 'gasly', 'stroll', 'ocon', 'tsunoda', 'lawson',
                'albon', 'colapinto', 'hulkenberg', 'bortoleto', 'bearman', 'antonelli', 'hadjar'
            ]
            drivers.extend([d for d in default_drivers if d not in drivers][:20 - len(drivers)])
        
        # Create race data for Dutch GP
        race_data = {
            'driver_id': drivers,
            'constructor_id': driver_performance.set_index('driver_id').loc[drivers, 'constructor_id'].tolist(),
            'circuit_id': ['zandvoort'] * len(drivers),
            'grid': list(range(1, len(drivers) + 1)),
            'season': [2025] * len(drivers),
            'round': [15] * len(drivers),  # Dutch GP is typically round 15
            'fastest_lap_rank': driver_performance.set_index('driver_id').loc[drivers, 'fastest_lap_rank'].fillna(0).tolist(),
            'position': driver_performance.set_index('driver_id').loc[drivers, 'position'].round().astype(int).clip(1, 20).tolist(),
            'points': driver_performance.set_index('driver_id').loc[drivers, 'position'].map({
                1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
            }).fillna(0).tolist(),
            'status': ['Finished'] * len(drivers),
            'time_millis': [4950000] * len(drivers),  # Typical Zandvoort race time
            'laps': [72] * len(drivers)  # Zandvoort lap count
        }
        race_df = pd.DataFrame(race_data)
        
        # Add weather features
        weather = self.fetch_weather_forecast()
        race_df['temperature'] = weather['temperature']
        race_df['humidity'] = weather['humidity']
        race_df['rain_probability'] = weather['rain_probability']
        race_df['weather_condition'] = weather['weather_condition']
        race_df['wind_speed'] = weather['wind_speed']
        
        # Encode categorical variables
        driver_mapping = {driver: i for i, driver in enumerate(self.historical_data['driver_id'].unique())}
        constructor_mapping = {constructor: i for i, constructor in enumerate(self.historical_data['constructor_id'].unique())}
        circuit_mapping = {circuit: i for i, circuit in enumerate(self.historical_data['circuit_id'].unique())}
        
        race_df['driver_id_encoded'] = race_df['driver_id'].map(driver_mapping).fillna(-1)
        race_df['constructor_id_encoded'] = race_df['constructor_id'].map(constructor_mapping).fillna(-1)
        race_df['circuit_id_encoded'] = race_df['circuit_id'].map(circuit_mapping).fillna(-1)
        
        return race_df
    
    def extract_recent_performance(self, race_df):
        """Extract recent performance data with weather integration"""
        print("📊 Extracting recent performance data...")
        
        recent_data = self.historical_data[self.historical_data['season'] >= 2021].copy()
        recent_data['position'] = recent_data['position'].apply(
            lambda x: min(max(int(x), 1), 20) if pd.notnull(x) else 20
        )
        
        # Combine with race data
        combined_df = pd.concat([recent_data, race_df], ignore_index=True)
        combined_df = self.create_advanced_features(combined_df)
        
        # Add weather features to 2025 data
        weather = self.fetch_weather_forecast()
        combined_df.loc[combined_df['season'] == 2025, 'temperature'] = weather['temperature']
        combined_df.loc[combined_df['season'] == 2025, 'humidity'] = weather['humidity']
        combined_df.loc[combined_df['season'] == 2025, 'rain_probability'] = weather['rain_probability']
        combined_df.loc[combined_df['season'] == 2025, 'weather_condition'] = weather['weather_condition']
        combined_df.loc[combined_df['season'] == 2025, 'wind_speed'] = weather['wind_speed']
        
        return combined_df.tail(len(race_df)).copy()
    
    def predict_dutch_gp(self):
        """Predict the 2025 Dutch GP results"""
        print("🔮 Predicting 2025 Dutch GP results...")
        
        race_df = self.prepare_race_data()
        race_features = self.extract_recent_performance(race_df)
        
        # Use only features from training
        X_race = race_features[[f for f in self.features if f in race_features.columns]]
        
        # Handle missing features
        for feature in self.features:
            if feature not in X_race.columns:
                X_race[feature] = 0
        
        # Ensure correct column order
        X_race = X_race[self.features]
        
        # Scale features
        X_race_scaled = self.scaler.transform(X_race)
        
        # Create ensemble model
        xgb = self.model
        lgbm = LGBMClassifier(
            objective='multiclass',
            num_class=20,
            random_state=42,
            n_jobs=-1,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=300,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        ensemble = VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm)], voting='soft')
        
        # Fit ensemble on recent data (2024-2025)
        recent_data = self.historical_data[self.historical_data['season'] >= 2024].copy()
        recent_data = self.create_advanced_features(recent_data)
        X_recent = recent_data[[f for f in self.features if f in recent_data.columns]]
        for feature in self.features:
            if feature not in X_recent.columns:
                X_recent[feature] = 0
        X_recent = X_recent[self.features]
        X_recent_scaled = self.scaler.transform(X_recent)
        y_recent = recent_data['position'].astype(int) - 1
        ensemble.fit(X_recent_scaled, y_recent)
        
        predictions = ensemble.predict(X_race_scaled) + 1
        probabilities = ensemble.predict_proba(X_race_scaled)
        confidence_scores = np.max(probabilities, axis=1)
        
        results_df = race_df.copy()
        results_df['predicted_position'] = predictions
        results_df['confidence'] = confidence_scores
        results_df['win_probability'] = probabilities[:, 0]
        results_df['podium_probability'] = probabilities[:, :3].sum(axis=1)
        
        return results_df.sort_values('predicted_position')
    
    def display_predictions(self, results_df):
        """Display prediction results"""
        print("\n" + "="*80)
        print("🇳🇱 2025 DUTCH GRAND PRIX - RACE PREDICTIONS")
        print("🏁 Circuit Zandvoort, Netherlands")
        print("="*80)
        print(f"{'Pos':<4} {'Driver':<20} {'Team':<15} {'Grid':<5} {'Win %':<8} {'Podium %':<10} {'Confidence':<10}")
        print("-"*80)
        
        for _, row in results_df.iterrows():
            print(f"{int(row['predicted_position']):<4} {row['driver_id']:<20} {row['constructor_id']:<15} "
                  f"{int(row['grid']):<5} {row['win_probability']*100:<8.1f} {row['podium_probability']*100:<10.1f} "
                  f"{row['confidence']*100:<10.1f}")
        
        print("\n🥇 PODIUM PREDICTIONS:")
        for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
            print(f"{i}. {row['driver_id']} ({row['constructor_id']}) - {row['win_probability']*100:.1f}% win chance")
    
    def create_visualizations(self, results_df):
        """Create enhanced visualizations"""
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Grid vs Predicted Position
        ax1.scatter(results_df['grid'], results_df['predicted_position'], s=100, alpha=0.7, c=results_df['confidence'], cmap='viridis')
        ax1.plot([1, 20], [1, 20], 'r--', alpha=0.5, label='Perfect correlation')
        ax1.set_xlabel('Grid Position')
        ax1.set_ylabel('Predicted Position')
        ax1.set_title('Grid vs Predicted Position (Dutch GP)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Win Probabilities
        top_10 = results_df.head(10)
        bars = ax2.bar(range(len(top_10)), top_10['win_probability']*100, color='orange', alpha=0.8)  # Dutch orange
        ax2.set_xlabel('Driver')
        ax2.set_ylabel('Win Probability (%)')
        ax2.set_title('Top 10 Win Probabilities - Dutch GP')
        ax2.set_xticks(range(len(top_10)))
        ax2.set_xticklabels(top_10['driver_id'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Podium Probabilities
        ax3.bar(range(len(top_10)), top_10['podium_probability']*100, color='darkorange', alpha=0.8)
        ax3.set_xlabel('Driver')
        ax3.set_ylabel('Podium Probability (%)')
        ax3.set_title('Top 10 Podium Probabilities - Dutch GP')
        ax3.set_xticks(range(len(top_10)))
        ax3.set_xticklabels(top_10['driver_id'], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Constructor Points
        constructor_points = results_df.groupby('constructor_id').agg({
            'predicted_position': lambda x: sum(26 - pos for pos in x if pos <= 10),
        }).sort_values('predicted_position', ascending=False)
        ax4.bar(constructor_points.index, constructor_points['predicted_position'], color='steelblue', alpha=0.8)
        ax4.set_xlabel('Constructor')
        ax4.set_ylabel('Predicted Points')
        ax4.set_title('Constructor Points Prediction - Dutch GP')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dutch_gp_2025_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Visualizations saved as 'dutch_gp_2025_predictions.png'")
    
    def save_predictions(self, results_df):
        """Save predictions to CSV"""
        output_df = results_df[['driver_id', 'constructor_id', 'grid', 'predicted_position', 
                               'win_probability', 'podium_probability', 'confidence']].copy()
        output_df.to_csv('dutch_gp_2025_predictions.csv', index=False)
        print("✅ Predictions saved to 'dutch_gp_2025_predictions.csv'")

# Main execution
if __name__ == "__main__":
    predictor = DutchGPPredictor(
        model_dir='E:/Shivam/F1/f1-ai-predictor/models',
        data_path='E:/Shivam/F1/f1-ai-predictor/data/f1_race_data.csv',
        weather_api_key='87e5ba753df3d9e4f6fc96f167f6a7a6'
    )
    
    if predictor.load_model_and_data():
        results = predictor.predict_dutch_gp()
        predictor.display_predictions(results)
        predictor.create_visualizations(results)
        predictor.save_predictions(results)
        print("\n🎉 Dutch GP 2025 predictions complete!")
    else:
        print("❌ Failed to load model or data.")