"""
Italian Grand Prix 2025 (Monza) - Race Prediction
Using V2 Enhanced Model
"""
import os
import pandas as pd
import joblib

# Load V2 model and encoders
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
model_v2 = joblib.load(os.path.join(MODEL_DIR, 'race_winner_model_v2.pkl'))
driver_encoder = joblib.load(os.path.join(MODEL_DIR, 'driver_encoder.pkl'))
constructor_encoder = joblib.load(os.path.join(MODEL_DIR, 'constructor_encoder.pkl'))
circuit_encoder = joblib.load(os.path.join(MODEL_DIR, 'circuit_encoder.pkl'))

# Monza GP 2025 - Expected Grid (example based on typical Monza performance)
# Monza is a high-speed, low-downforce circuit favoring powerful engines
MONZA_GP_GRID = [
    # McLaren typically strong at Monza (low drag)
    {'driver_id': 'norris', 'constructor_id': 'mclaren', 'quali_pos': 1, 'grid': 1, 'driver_form': 18.0, 'constructor_form': 20.0, 'track_exp': 7, 'grid_to_finish_avg': -0.5, 'recent_position_avg': 2.5, 'constructor_position_std': 3.0, 'form_momentum': 2.0},
    {'driver_id': 'piastri', 'constructor_id': 'mclaren', 'quali_pos': 2, 'grid': 2, 'driver_form': 15.0, 'constructor_form': 20.0, 'track_exp': 2, 'grid_to_finish_avg': 1.0, 'recent_position_avg': 2.0, 'constructor_position_std': 3.0, 'form_momentum': 3.0},
    
    # Ferrari strong at home race (Tifosi power!)
    {'driver_id': 'leclerc', 'constructor_id': 'ferrari', 'quali_pos': 3, 'grid': 3, 'driver_form': 12.0, 'constructor_form': 15.0, 'track_exp': 7, 'grid_to_finish_avg': 0.8, 'recent_position_avg': 3.5, 'constructor_position_std': 4.0, 'form_momentum': 1.0},
    {'driver_id': 'sainz', 'constructor_id': 'ferrari', 'quali_pos': 4, 'grid': 4, 'driver_form': 10.0, 'constructor_form': 15.0, 'track_exp': 7, 'grid_to_finish_avg': 0.3, 'recent_position_avg': 5.0, 'constructor_position_std': 4.0, 'form_momentum': 0.5},
    
    # Red Bull (strong engine but not optimal for low downforce)
    {'driver_id': 'max_verstappen', 'constructor_id': 'red_bull', 'quali_pos': 5, 'grid': 5, 'driver_form': 25.0, 'constructor_form': 25.0, 'track_exp': 9, 'grid_to_finish_avg': 1.5, 'recent_position_avg': 1.5, 'constructor_position_std': 2.0, 'form_momentum': 1.0},
    {'driver_id': 'perez', 'constructor_id': 'red_bull', 'quali_pos': 6, 'grid': 6, 'driver_form': 8.0, 'constructor_form': 25.0, 'track_exp': 9, 'grid_to_finish_avg': 0.0, 'recent_position_avg': 6.0, 'constructor_position_std': 2.0, 'form_momentum': -0.5},
    
    # Mercedes
    {'driver_id': 'hamilton', 'constructor_id': 'mercedes', 'quali_pos': 7, 'grid': 7, 'driver_form': 12.0, 'constructor_form': 12.0, 'track_exp': 17, 'grid_to_finish_avg': 0.5, 'recent_position_avg': 5.0, 'constructor_position_std': 3.5, 'form_momentum': 0.5},
    {'driver_id': 'russell', 'constructor_id': 'mercedes', 'quali_pos': 8, 'grid': 8, 'driver_form': 10.0, 'constructor_form': 12.0, 'track_exp': 4, 'grid_to_finish_avg': 0.2, 'recent_position_avg': 6.0, 'constructor_position_std': 3.5, 'form_momentum': 0.0},
    
    # Aston Martin
    {'driver_id': 'alonso', 'constructor_id': 'aston_martin', 'quali_pos': 9, 'grid': 9, 'driver_form': 4.0, 'constructor_form': 5.0, 'track_exp': 21, 'grid_to_finish_avg': 0.0, 'recent_position_avg': 9.0, 'constructor_position_std': 5.0, 'form_momentum': -0.5},
    {'driver_id': 'stroll', 'constructor_id': 'aston_martin', 'quali_pos': 10, 'grid': 10, 'driver_form': 2.0, 'constructor_form': 5.0, 'track_exp': 7, 'grid_to_finish_avg': -0.5, 'recent_position_avg': 11.0, 'constructor_position_std': 5.0, 'form_momentum': -1.0},
]

def predict_monza_gp():
    """Predict Italian Grand Prix at Monza"""
    print("\n" + "="*70)
    print("🇮🇹 ITALIAN GRAND PRIX 2025 - MONZA")
    print("="*70 + "\n")
    
    print("🏁 CIRCUIT INFO:")
    print("  Name: Autodromo Nazionale Monza")
    print("  Type: High-speed, low-downforce circuit")
    print("  Characteristics:")
    print("    - Fastest circuit on calendar")
    print("    - Long straights favor powerful engines")
    print("    - Low drag setup crucial")
    print("    - Slipstream effect very strong")
    print("    - Overtaking: High (DRS zones)")
    print("    - Ferrari home race (Tifosi pressure!)")
    print()
    
    predictions = []
    
    for entry in MONZA_GP_GRID:
        driver_id = entry['driver_id']
        constructor_id = entry['constructor_id']
        
        try:
            driver_enc = driver_encoder.transform([driver_id])[0]
            constructor_enc = constructor_encoder.transform([constructor_id])[0]
            circuit_enc = circuit_encoder.transform(['monza'])[0] if 'monza' in circuit_encoder.classes_ else 0
        except:
            print(f"⚠️  {driver_id} not in training data, skipping")
            continue
        
        # Create feature vector
        features = pd.DataFrame([{
            'driver_encoded': driver_enc,
            'constructor_encoded': constructor_enc,
            'circuit_encoded': circuit_enc,
            'grid': entry['grid'],
            'quali_pos': entry['quali_pos'],
            'driver_form': entry['driver_form'],
            'constructor_form': entry['constructor_form'],
            'grid_penalty': 0,
            'track_experience': entry['track_exp'],
            'grid_to_finish_avg': entry['grid_to_finish_avg'],
            'recent_position_avg': entry['recent_position_avg'],
            'constructor_position_std': entry['constructor_position_std'],
            'quali_grid_delta': 0,
            'improvement_potential': 20 - entry['grid'],
            'form_momentum': entry['form_momentum']
        }])
        
        # Predict
        pred_class = model_v2.predict(features)[0]
        pred_position = pred_class + 1
        
        probabilities = model_v2.predict_proba(features)[0]
        win_prob = probabilities[0] * 100
        podium_prob = sum(probabilities[:3]) * 100
        
        predictions.append({
            'driver': driver_id,
            'constructor': constructor_id,
            'grid': entry['grid'],
            'predicted_position': pred_position,
            'win_probability': win_prob,
            'podium_probability': podium_prob,
            'track_experience': entry['track_exp']
        })
    
    # Sort by predicted position
    predictions.sort(key=lambda x: x['predicted_position'])
    
    # Display predictions
    print("="*70)
    print("🤖 V2 MODEL PREDICTIONS:")
    print("="*70 + "\n")
    
    print(f"{'Pos':<5} {'Driver':<20} {'Team':<12} {'Grid':<6} {'Win %':<8} {'Podium %':<10} {'Exp':<5}")
    print("-" * 75)
    
    for i, pred in enumerate(predictions, 1):
        driver_name = pred['driver'].replace('_', ' ').title()
        team = pred['constructor'].replace('_', ' ').title()
        
        # Highlight Ferrari drivers at home race
        emoji = "🏠" if pred['constructor'] == 'ferrari' else ""
        
        print(f"P{i:<4} {driver_name:<20} {team:<12} P{pred['grid']:<5} "
              f"{pred['win_probability']:>5.1f}%  {pred['podium_probability']:>6.1f}%   "
              f"{pred['track_experience']:>3} {emoji}")
    
    # Podium predictions
    print("\n" + "="*70)
    print("🏆 PREDICTED PODIUM:")
    print("="*70 + "\n")
    
    for i in range(min(3, len(predictions))):
        pred = predictions[i]
        driver_name = pred['driver'].replace('_', ' ').title()
        team = pred['constructor'].replace('_', ' ').title()
        medal = ["🥇", "🥈", "🥉"][i]
        
        print(f"{medal} P{i+1}: {driver_name} ({team})")
        print(f"     Starting Grid: P{pred['grid']}")
        print(f"     Win Probability: {pred['win_probability']:.1f}%")
        print(f"     Podium Probability: {pred['podium_probability']:.1f}%")
        print(f"     Monza Experience: {pred['track_experience']} races")
        print()
    
    # Key insights
    print("="*70)
    print("💡 KEY INSIGHTS:")
    print("="*70 + "\n")
    
    # McLaren analysis
    mclaren_drivers = [p for p in predictions if p['constructor'] == 'mclaren']
    if mclaren_drivers:
        print("🟠 McLaren:")
        print("   - Low-drag philosophy suits Monza perfectly")
        print("   - Strong slipstream effect in qualifying")
        print("   - Expect competitive race pace")
        print()
    
    # Ferrari home race
    ferrari_drivers = [p for p in predictions if p['constructor'] == 'ferrari']
    if ferrari_drivers:
        print("🔴 Ferrari (Home Race):")
        print("   - Tifosi pressure and motivation")
        print("   - Strong historical performance at Monza")
        avg_ferrari_pos = sum(p['predicted_position'] for p in ferrari_drivers) / len(ferrari_drivers)
        print(f"   - Average predicted position: P{avg_ferrari_pos:.1f}")
        print()
    
    # Verstappen analysis
    verstappen = next((p for p in predictions if p['driver'] == 'max_verstappen'), None)
    if verstappen:
        print("🔵 Verstappen:")
        print(f"   - Predicted: P{verstappen['predicted_position']} (from P{verstappen['grid']})")
        print("   - Red Bull may struggle with low downforce setup")
        print("   - But never count out the champion!")
        print()
    
    # Race characteristics
    print("🏁 MONZA RACE FACTORS:")
    print("   - Slipstream crucial for overtaking")
    print("   - Tire management less critical (low degradation)")
    print("   - Safety car probability: Medium")
    print("   - Weather: Typically dry in September")
    print("   - First chicane often chaotic at start!")
    
    print("\n" + "="*70 + "\n")
    
    # Save predictions
    output_file = os.path.join(os.path.dirname(__file__), '..', 'monza_gp_2025_predictions.csv')
    pd.DataFrame(predictions).to_csv(output_file, index=False)
    print(f"💾 Predictions saved to: {output_file}\n")
    
    return predictions

if __name__ == "__main__":
    predict_monza_gp()
