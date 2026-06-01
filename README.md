# HealthFusion

Cardiovascular risk assessment Streamlit app powered by a GradientBoostingClassifier
combining clinical, lifestyle and environmental features.

## Files

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `healthfusion_model.pkl` — trained GradientBoostingClassifier (you provide)
- `healthfusion_scaler.pkl` — fitted scaler used during training (you provide)
- `feature_columns.pkl` — list of feature names in the exact training order (you provide)

Place the three `.pkl` files next to `app.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app**.
3. Select the repo, branch and `app.py` as the entry point.
4. Click **Deploy**.

## Derived features

- `BMI = weight / (height/100)^2`
- `Health_Stress_Score = cholesterol + gluc + ap_hi/100`
- `Heat_Risk = Temperature + Humidity`
- `Pollution_Risk = PM2.5 + PM10 + NO2`

## Notes on feature alignment

The app loads `feature_columns.pkl` and reorders inputs to match the training
schema. Common cardio-dataset names (`age`, `gender`, `ap_hi`, `ap_lo`,
`cholesterol`, `gluc`, `smoke`, `alco`, `active`) and environmental names
(`Temperature`, `Humidity`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`,
`Proximity_to_Industrial_Areas`, `Population_Density`, `Air_Quality`,
`BMI`, `Health_Stress_Score`, `Heat_Risk`, `Pollution_Risk`) are mapped
automatically (case-insensitive). If your column names differ, edit the
`raw_features` dict in `app.py`.