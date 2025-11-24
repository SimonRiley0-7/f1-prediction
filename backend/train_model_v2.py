"""
Enhanced F1 Race Winner Model Training
Improved model with better hyperparameters and feature engineering
"""
import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Paths
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'f1_race_data_prepared.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def create_advanced_features(df):
    """Create additional features from existing data"""
    print("⚙️  Creating advanced features...")
    
    # Sort by time
    df = df.sort_values(['season', 'round', 'driver_id']).copy()
    
    # 1. Grid to finish differential (historical average)
    df['grid_to_finish_avg'] = df.groupby('driver_id').apply(
        lambda x: (x['grid'] - x['position']).rolling(5, min_periods=1).mean().shift(1)
    ).reset_index(level=0, drop=True).fillna(0)
    
    # 2. Recent position trend
    df['recent_position_avg'] = df.groupby('driver_id')['position'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    ).fillna(10)
    
    # 3. Constructor consistency
    df['constructor_position_std'] = df.groupby('constructor_id')['position'].transform(
        lambda x: x.rolling(5, min_periods=1).std().shift(1)
    ).fillna(5)
    
    # 4. Qualifying vs grid delta (penalties indicator)
    df['quali_grid_delta'] = abs(df['grid'] - df['quali_pos'])
    
    # 5. Position improvement potential (based on grid)
    df['improvement_potential'] = 20 - df['grid']
    
    # 6. Form momentum (improving or declining)
    df['form_momentum'] = df.groupby('driver_id')['driver_form'].transform(
        lambda x: x.diff().shift(1)
    ).fillna(0)
    
    return df

def train_enhanced_model():
    """Train enhanced race winner prediction model"""
    print("\n" + "="*70)
    print("F1 ENHANCED RACE WINNER MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Loaded {len(df):,} race records")
    print(f"   Seasons: {df['season'].min()} - {df['season'].max()}\n")
    
    # Create advanced features
    df = create_advanced_features(df)
    
    # Enhanced feature set
    feature_cols = [
        # Core features
        'driver_encoded',
        'constructor_encoded', 
        'circuit_encoded',
        'grid',
        'quali_pos',
        
        # Form features
        'driver_form',
        'constructor_form',
        
        # Advanced features
        'grid_penalty',
        'track_experience',
        'grid_to_finish_avg',
        'recent_position_avg',
        'constructor_position_std',
        'quali_grid_delta',
        'improvement_potential',
        'form_momentum'
    ]
    
    # Check available features
    available_features = [f for f in feature_cols if f in df.columns]
    print(f"📊 Using {len(available_features)} features:")
    for f in available_features:
        print(f"   - {f}")
    print()
    
    # Prepare data
    df_clean = df[df['position'].notna()].copy()
    df_clean = df_clean[df_clean['position'] <= 20]
    
    print(f"📈 Training data: {len(df_clean):,} records\n")
    
    X = df_clean[available_features]
    y = df_clean['position'].astype(int)
    
    # Time-based split
    train_mask = df_clean['season'] < 2024
    test_mask = df_clean['season'] >= 2024
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"🔹 Training set: {len(X_train):,} records (pre-2024)")
    print(f"🔹 Test set: {len(X_test):,} records (2024+)\n")
    
    # Convert to 0-indexed classes
    y_train_cls = y_train - 1
    y_test_cls = y_test - 1
    
    # Train enhanced XGBoost model
    print("🤖 Training Enhanced XGBoost Model...")
    print("   Hyperparameters optimized for race prediction\n")
    
    xgb_model = XGBClassifier(
        n_estimators=300,  # More trees
        learning_rate=0.05,  # Slower learning
        max_depth=8,  # Deeper trees
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=20,
        n_jobs=-1,
        random_state=42,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0   # L2 regularization
    )
    
    xgb_model.fit(X_train, y_train_cls)
    print("✅ Model trained!\n")
    
    # Evaluate
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_pos = y_pred + 1
    
    # Overall accuracy
    acc = accuracy_score(y_test_cls, y_pred)
    print(f"📊 Exact Position Accuracy: {acc:.2%}\n")
    
    # Top-3 accuracy (within 3 positions)
    within_3 = np.abs(y_pred_pos - y_test) <= 3
    top3_acc = within_3.mean()
    print(f"🎯 Within 3 Positions Accuracy: {top3_acc:.2%}\n")
    
    # Winner prediction
    test_df = df_clean[test_mask].copy()
    test_df['predicted_position'] = y_pred_pos
    
    winners = test_df[test_df['position'] == 1]
    if len(winners) > 0:
        correct_winners = winners[winners['predicted_position'] == 1]
        winner_acc = len(correct_winners) / len(winners)
        print(f"🏆 Winner Prediction Accuracy: {winner_acc:.2%}")
        print(f"   ({len(correct_winners)}/{len(winners)} races)\n")
        
        # Winners predicted in top 3
        winners_in_top3 = winners[winners['predicted_position'] <= 3]
        winner_top3_acc = len(winners_in_top3) / len(winners)
        print(f"🥇 Winners Predicted in Top 3: {winner_top3_acc:.2%}")
        print(f"   ({len(winners_in_top3)}/{len(winners)} races)\n")
    
    # Podium accuracy
    podium = test_df[test_df['position'] <= 3]
    if len(podium) > 0:
        podium_in_top3 = podium[podium['predicted_position'] <= 3]
        podium_acc = len(podium_in_top3) / len(podium)
        print(f"🥉 Podium Finishers in Top 3: {podium_acc:.2%}")
        print(f"   ({len(podium_in_top3)}/{len(podium)} podium finishes)\n")
    
    # Feature importance
    print("="*70)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*70 + "\n")
    
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'race_winner_model_v2.pkl')
    joblib.dump(xgb_model, model_path)
    
    print(f"\n✅ Enhanced model saved to: {model_path}")
    
    # Save feature list
    feature_list_path = os.path.join(MODEL_DIR, 'model_features_v2.txt')
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(available_features))
    print(f"✅ Feature list saved to: {feature_list_path}")
    
    print("\n" + "="*70)
    print("✅ ENHANCED TRAINING COMPLETE!")
    print("="*70)
    print(f"\n💡 Improvements over v1:")
    print(f"   - {len(available_features)} features (vs 9 in v1)")
    print(f"   - Optimized hyperparameters")
    print(f"   - Better regularization")
    print(f"   - Advanced race pace features")
    print("\n" + "="*70 + "\n")
    
    return xgb_model

if __name__ == "__main__":
    train_enhanced_model()
