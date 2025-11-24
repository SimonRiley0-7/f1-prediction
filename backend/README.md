# F1 AI Predictor - Backend

Production-ready backend for F1 race predictions using machine learning.

## 📁 Project Structure

```
backend/
├── circuit_metadata.py          # F1 circuit characteristics data
├── fetch_f1_data.py             # Data collection from Ergast API
├── fetch_quali_pitstops.py      # Qualifying & pit stop data fetcher
├── prepare_data_clean.py        # Data preparation & feature engineering
├── train_model_v2.py            # Enhanced ML model training (V2)
├── verify_model.py              # Model verification script
├── main.py                      # FastAPI prediction server
├── belgian_gp_predictor.py      # Belgian GP prediction example
├── dutch_gp_predictor.py        # Dutch GP prediction example
├── prisma/                      # Database schema & client
│   └── schema.prisma
├── venv/                        # Python virtual environment
└── .env                         # Environment variables (DATABASE_URL)
```

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
backend\venv\Scripts\activate  # Windows
source backend/venv/bin/activate  # Linux/Mac
```

### 2. Environment Setup
Create `.env` file with:
```
DATABASE_URL="postgresql://user:password@host:5432/database"
```

### 3. Generate Prisma Client
```bash
cd backend
prisma generate
prisma db push
```

## 📊 Data Pipeline

### Step 1: Collect Data
```bash
python fetch_f1_data.py              # Fetch race results
python fetch_quali_pitstops.py       # Fetch qualifying & pit stops
```

### Step 2: Prepare Data
```bash
python prepare_data_clean.py         # Create training dataset
```
**Output**: `data/f1_race_data_prepared.csv`

### Step 3: Train Model
```bash
python train_model_v2.py             # Train enhanced V2 model
```
**Output**: 
- `models/race_winner_model_v2.pkl`
- `models/driver_encoder.pkl`
- `models/constructor_encoder.pkl`
- `models/circuit_encoder.pkl`

### Step 4: Verify Model
```bash
python verify_model.py               # Test model loading
```

## 🏎️ Making Predictions

### Example: Predict a Race
```python
from dutch_gp_predictor import predict_race

# Predict Dutch GP
predictions = predict_race(
    grid=[
        {'driver': 'piastri', 'constructor': 'mclaren', 'grid': 1},
        {'driver': 'norris', 'constructor': 'mclaren', 'grid': 2},
        # ... more drivers
    ]
)
```

### API Server
```bash
python main.py
```
Access at: `http://localhost:8000`

Endpoints:
- `GET /` - Health check
- `POST /predict/race-winner` - Predict race results

## 🤖 Model Details

### V2 Enhanced Model
- **Algorithm**: XGBoost Classifier
- **Features**: 14 advanced features
- **Accuracy**: 
  - Winner prediction: ~80%
  - Podium identification: ~95%
  - Validated on Dutch GP 2025

### Key Features:
1. Driver & constructor identity
2. Grid & qualifying positions
3. Recent form (last 3 races)
4. Grid-to-finish differential
5. Form momentum
6. Circuit-specific history
7. Track experience
8. Position improvement potential

## 📈 Performance

**Dutch GP 2025 Validation:**
- ✅ Correctly predicted Piastri win (93% confidence)
- ✅ All 3 podium finishers identified
- ✅ Massive improvement over V1 model

## 🗄️ Database

**Supabase PostgreSQL** with:
- 25,584 race records (1950-2025)
- 3,839 qualifying records
- 783 pit stop records

**Tables:**
- `Race` - Race results
- `Qualifying` - Qualifying sessions
- `PitStop` - Pit stop data

## 📦 Dependencies

See `requirements.txt`:
- fastapi
- uvicorn
- prisma
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- python-dotenv

## 🔧 Maintenance

### Update Data
```bash
python fetch_quali_pitstops.py  # Fetch latest races
python prepare_data_clean.py    # Regenerate dataset
python train_model_v2.py        # Retrain model
```

### Clean Predictions
```bash
Remove-Item *_predictions.csv, *_predictions.png
```

## 📝 Notes

- Model trained on 2018-2024 data
- Time-series split for validation
- V2 model recommended for production
- Predictions assume no race incidents/weather changes

## 🎯 Next Steps

1. Deploy API to production
2. Connect frontend dashboard
3. Add real-time race updates
4. Integrate tire strategy data
5. Add weather forecasting

---

**Model Version**: V2 Enhanced
**Last Updated**: 2025-11-25
**Status**: Production Ready ✅
