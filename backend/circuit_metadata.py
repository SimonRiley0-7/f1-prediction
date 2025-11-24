
# Circuit Metadata - Domain Knowledge for F1 Predictor
# Scores are 1-5 (1 = Very Low/Easy, 5 = Very High/Hard)

CIRCUIT_METADATA = {
    "bahrain": {
        "name": "Bahrain International Circuit",
        "type": "Race",
        "downforce": 3,
        "overtaking": 4,
        "tire_stress": 5,
        "asphalt_abrasion": 5,
        "track_evolution": 3
    },
    "jeddah": {
        "name": "Jeddah Corniche Circuit",
        "type": "Street",
        "downforce": 2,
        "overtaking": 3,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 4
    },
    "albert_park": {
        "name": "Albert Park Grand Prix Circuit",
        "type": "Street",
        "downforce": 4,
        "overtaking": 2,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 5
    },
    "suzuka": {
        "name": "Suzuka Circuit",
        "type": "Race",
        "downforce": 5,
        "overtaking": 2,
        "tire_stress": 5,
        "asphalt_abrasion": 4,
        "track_evolution": 3
    },
    "shanghai": {
        "name": "Shanghai International Circuit",
        "type": "Race",
        "downforce": 4,
        "overtaking": 4,
        "tire_stress": 4,
        "asphalt_abrasion": 3,
        "track_evolution": 3
    },
    "miami": {
        "name": "Miami International Autodrome",
        "type": "Street",
        "downforce": 3,
        "overtaking": 3,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 5
    },
    "imola": {
        "name": "Autodromo Enzo e Dino Ferrari",
        "type": "Race",
        "downforce": 4,
        "overtaking": 1,
        "tire_stress": 3,
        "asphalt_abrasion": 3,
        "track_evolution": 2
    },
    "monaco": {
        "name": "Circuit de Monaco",
        "type": "Street",
        "downforce": 5,
        "overtaking": 1,
        "tire_stress": 1,
        "asphalt_abrasion": 1,
        "track_evolution": 5
    },
    "villeneuve": {
        "name": "Circuit Gilles Villeneuve",
        "type": "Hybrid",
        "downforce": 2,
        "overtaking": 3,
        "tire_stress": 2,
        "asphalt_abrasion": 2,
        "track_evolution": 4
    },
    "catalunya": {
        "name": "Circuit de Barcelona-Catalunya",
        "type": "Race",
        "downforce": 5,
        "overtaking": 2,
        "tire_stress": 4,
        "asphalt_abrasion": 4,
        "track_evolution": 2
    },
    "red_bull_ring": {
        "name": "Red Bull Ring",
        "type": "Race",
        "downforce": 3,
        "overtaking": 4,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 3
    },
    "silverstone": {
        "name": "Silverstone Circuit",
        "type": "Race",
        "downforce": 5,
        "overtaking": 3,
        "tire_stress": 5,
        "asphalt_abrasion": 4,
        "track_evolution": 2
    },
    "hungaroring": {
        "name": "Hungaroring",
        "type": "Race",
        "downforce": 5,
        "overtaking": 1,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 4
    },
    "spa": {
        "name": "Circuit de Spa-Francorchamps",
        "type": "Race",
        "downforce": 2,
        "overtaking": 5,
        "tire_stress": 5,
        "asphalt_abrasion": 3,
        "track_evolution": 2
    },
    "zandvoort": {
        "name": "Circuit Zandvoort",
        "type": "Race",
        "downforce": 5,
        "overtaking": 1,
        "tire_stress": 4,
        "asphalt_abrasion": 4,
        "track_evolution": 3
    },
    "monza": {
        "name": "Autodromo Nazionale Monza",
        "type": "Race",
        "downforce": 1,
        "overtaking": 4,
        "tire_stress": 2,
        "asphalt_abrasion": 2,
        "track_evolution": 3
    },
    "baku": {
        "name": "Baku City Circuit",
        "type": "Street",
        "downforce": 1,
        "overtaking": 4,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 5
    },
    "singapore": {
        "name": "Marina Bay Street Circuit",
        "type": "Street",
        "downforce": 5,
        "overtaking": 1,
        "tire_stress": 4,
        "asphalt_abrasion": 3,
        "track_evolution": 5
    },
    "austin": {
        "name": "Circuit of the Americas",
        "type": "Race",
        "downforce": 4,
        "overtaking": 3,
        "tire_stress": 4,
        "asphalt_abrasion": 3,
        "track_evolution": 3
    },
    "mexico": {
        "name": "Autodromo Hermanos Rodriguez",
        "type": "Race",
        "downforce": 5,
        "overtaking": 3,
        "tire_stress": 2,
        "asphalt_abrasion": 2,
        "track_evolution": 4
    },
    "interlagos": {
        "name": "Autodromo Jose Carlos Pace",
        "type": "Race",
        "downforce": 4,
        "overtaking": 4,
        "tire_stress": 3,
        "asphalt_abrasion": 3,
        "track_evolution": 3
    },
    "las_vegas": {
        "name": "Las Vegas Strip Circuit",
        "type": "Street",
        "downforce": 1,
        "overtaking": 4,
        "tire_stress": 2,
        "asphalt_abrasion": 1,
        "track_evolution": 5
    },
    "losail": {
        "name": "Losail International Circuit",
        "type": "Race",
        "downforce": 4,
        "overtaking": 3,
        "tire_stress": 5,
        "asphalt_abrasion": 4,
        "track_evolution": 2
    },
    "yas_marina": {
        "name": "Yas Marina Circuit",
        "type": "Race",
        "downforce": 3,
        "overtaking": 2,
        "tire_stress": 3,
        "asphalt_abrasion": 2,
        "track_evolution": 3
    }
}

def get_circuit_features(circuit_id):
    """Returns a dictionary of circuit features or defaults if not found."""
    defaults = {
        "downforce": 3,
        "overtaking": 3,
        "tire_stress": 3,
        "asphalt_abrasion": 3,
        "track_evolution": 3,
        "is_street": 0
    }
    
    if circuit_id in CIRCUIT_METADATA:
        data = CIRCUIT_METADATA[circuit_id]
        return {
            "downforce": data["downforce"],
            "overtaking": data["overtaking"],
            "tire_stress": data["tire_stress"],
            "asphalt_abrasion": data["asphalt_abrasion"],
            "track_evolution": data["track_evolution"],
            "is_street": 1 if data["type"] == "Street" else 0
        }
    return defaults
