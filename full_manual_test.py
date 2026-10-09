import sys
import joblib
import pandas as pd
import numpy as np

# 1. Load trained specialist models
try:
    specialists = joblib.load('model2_specialists.joblib')
    print("Loaded specialist models successfully.")
except Exception as e:
    print(f"Error loading model2_specialists.joblib: {e}")
    sys.exit(1)

# Check SHAP availability
try:
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_SHAP_EXPLAINERS = {}

OPTIMUM_NPK_BENCHMARKS = {
    'RICE':         {'N': 100, 'P': 50, 'K': 50, 'water_sensitive': True,  'daily_water_mm': 50.0},
    'WHEAT':        {'N': 120, 'P': 60, 'K': 40, 'water_sensitive': True,  'daily_water_mm': 30.0},
    'MAIZE':        {'N': 120, 'P': 60, 'K': 40, 'water_sensitive': True,  'daily_water_mm': 35.0},
    'CHICKPEA':     {'N': 20,  'P': 40, 'K': 20, 'water_sensitive': False, 'daily_water_mm': 20.0},
    'SOYABEAN':     {'N': 30,  'P': 60, 'K': 40, 'water_sensitive': False, 'daily_water_mm': 25.0},
    'GROUNDNUT':    {'N': 25,  'P': 50, 'K': 75, 'water_sensitive': False, 'daily_water_mm': 25.0},
    'SUGARCANE':    {'N': 250, 'P': 100,'K': 120,'water_sensitive': True,  'daily_water_mm': 45.0},
    'SUNFLOWER':    {'N': 60,  'P': 40, 'K': 30, 'water_sensitive': False, 'daily_water_mm': 25.0},
    'BARLEY':       {'N': 60,  'P': 30, 'K': 20, 'water_sensitive': True,  'daily_water_mm': 25.0},
    'LINSEED':      {'N': 40,  'P': 20, 'K': 20, 'water_sensitive': False, 'daily_water_mm': 20.0},
    'MINOR PULSES': {'N': 20,  'P': 40, 'K': 20, 'water_sensitive': False, 'daily_water_mm': 20.0},
    'OILSEEDS':     {'N': 50,  'P': 30, 'K': 30, 'water_sensitive': False, 'daily_water_mm': 25.0}
}

STATE_DEFAULT_SOIL_MAP = {
    'MADHYA PRADESH': 'VERTIC SOILS',
    'MAHARASHTRA': 'VERTIC SOILS',
    'GUJARAT': 'VERTIC SOILS',
    'PUNJAB': 'INCEPTISOLS',
    'HARYANA': 'INCEPTISOLS',
    'UTTAR PRADESH': 'INCEPTISOLS',
    'BIHAR': 'INCEPTISOLS',
    'WEST BENGAL': 'INCEPTISOLS',
    'KARNATAKA': 'ALFISOLS',
    'ANDHRA PRADESH': 'ALFISOLS',
    'TELANGANA': 'ALFISOLS',
    'TAMIL NADU': 'ALFISOLS',
    'RAJASTHAN': 'ARIDISOLS'
}

def resolve_soil_type(state: str, provided_soil: str = None) -> str:
    if provided_soil and provided_soil.strip().upper() not in ['', 'NONE', 'UNKNOWN', "DON'T KNOW"]:
        return provided_soil.strip().upper()
    return STATE_DEFAULT_SOIL_MAP.get(state.strip().upper(), 'INCEPTISOLS')

def prepare_feature_row(crop: str, state: str, soil: str, irrigated: bool,
                        n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float = 850.0):
    crop_upper = crop.strip().upper()
    fam_name = 'cereals'
    for f_name, data in specialists.items():
        if crop_upper in data['crops']:
            fam_name = f_name
            break
            
    engine = specialists[fam_name]
    feature_names = engine['feature_names']
    
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    total_npk = n_kg + p_kg + k_kg
    safe_npk = total_npk if total_npk > 0 else 1.0
    irr_flag = 1.0 if irrigated else 0.0
    
    row['Crop_Irrigated_Area_ha'] = 1.0 if irrigated else 0.0
    row['Is_Irrigated'] = irr_flag
    row['Total_District_Irrigated_1000_ha'] = 50.0
    row['Fertilizer_N_kg_per_ha'] = n_kg
    row['Fertilizer_P_kg_per_ha'] = p_kg
    row['Fertilizer_K_kg_per_ha'] = k_kg
    row['Fertilizer_NPK_Total_kg_per_ha'] = total_npk
    row['N_Ratio'] = n_kg / safe_npk
    row['P_Ratio'] = p_kg / safe_npk
    row['K_Ratio'] = k_kg / safe_npk
    row['Dominant_Soil_Percent'] = 70.0
    
    if 'Rainfall_Kharif_Monsoon' in row.columns:
        row['Rainfall_Kharif_Monsoon'] = rainfall_mm
    if 'Moisture_Index' in row.columns:
        row['Moisture_Index'] = rainfall_mm * (1.0 + 0.5 * irr_flag)
        
    resolved_soil = resolve_soil_type(state, soil)
    for prefix, name in [('Crop_', crop_upper), ('State Name_', state), ('Dominant_Soil_Type_', resolved_soil)]:
        col = f"{prefix}{name}"
        if col in row.columns:
            row[col] = 1.0
            
    return fam_name, row

def predict_raw_yield(crop: str, state: str, soil: str, irrigated: bool,
                      n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float = 850.0):
    fam_name, row = prepare_feature_row(crop, state, soil, irrigated, n_kg, p_kg, k_kg, rainfall_mm)
    model = specialists[fam_name]['model']
    pred_kg_ha = max(0.0, float(model.predict(row)[0]))
    return pred_kg_ha / 247.105

def compute_smart_irrigation_prescription(crop: str, acres: float, soil_moisture_pct: float,
                                         forecast_rain_mm_48h: float, pump_hp: float = 5.0):
    crop_upper = crop.strip().upper()
    benchmarks = OPTIMUM_NPK_BENCHMARKS.get(crop_upper, {'daily_water_mm': 30.0, 'water_sensitive': True})
    target_depth_mm = benchmarks['daily_water_mm']
    
    current_water_mm = (soil_moisture_pct / 60.0) * target_depth_mm
    raw_deficit_mm = max(0.0, target_depth_mm - current_water_mm)
    effective_rain_mm = forecast_rain_mm_48h * 0.85
    net_irrigation_depth_mm = max(0.0, raw_deficit_mm - effective_rain_mm)
    
    total_water_liters = net_irrigation_depth_mm * 4047.0 * acres
    liters_per_acre = net_irrigation_depth_mm * 4047.0
    rain_liters_expected = effective_rain_mm * 4047.0 * acres
    
    pump_flow_lph = pump_hp * 5000.0
    pump_hours_needed = (total_water_liters / pump_flow_lph) if total_water_liters > 0 else 0.0
    
    advice_lines = []
    if forecast_rain_mm_48h >= raw_deficit_mm and raw_deficit_mm > 0:
        decision = "HOLD IRRIGATION (DO NOT TURN ON PUMP)"
        advice_lines.append(f"Weather forecast predicts {forecast_rain_mm_48h:.1f} mm rain (~{rain_liters_expected:,.0f} Liters across plot) in next 48 hrs.")
        advice_lines.append("Natural rainfall will satisfy crop root water needs.")
        advice_lines.append("Pumping water now will waste electricity and cause waterlogging.")
        trigger = False
    elif forecast_rain_mm_48h > 0 and net_irrigation_depth_mm > 0:
        decision = "PARTIAL SUPPLEMENTAL IRRIGATION"
        advice_lines.append(f"Weather API shows light rain of {forecast_rain_mm_48h:.1f} mm (~{rain_liters_expected:,.0f} Liters) coming.")
        advice_lines.append(f"Rain is insufficient. Provide {total_water_liters:,.0f} Liters ({liters_per_acre:,.0f} L/Acre).")
        advice_lines.append(f"Run {pump_hp:.0f} HP pump for ~{pump_hours_needed:.1f} hours ({int(pump_hours_needed * 60)} minutes).")
        trigger = True
    elif net_irrigation_depth_mm > 0:
        decision = "CRITICAL IRRIGATION REQUIRED"
        advice_lines.append(f"No rain in 48-hr forecast. Soil moisture is low ({soil_moisture_pct:.1f}%).")
        advice_lines.append(f"Apply {total_water_liters:,.0f} Liters across {acres:.1f} Acres ({liters_per_acre:,.0f} L/Acre).")
        advice_lines.append(f"Run {pump_hp:.0f} HP pump for {pump_hours_needed:.1f} hours (~{pump_hours_needed/acres:.1f} hrs/acre).")
        trigger = True
    else:
        decision = "OPTIMAL MOISTURE - NO IRRIGATION NEEDED"
        advice_lines.append(f"Soil moisture is in optimal zone ({soil_moisture_pct:.1f}%).")
        advice_lines.append("Re-check moisture readings in 48 hours.")
        trigger = False
        
    return {
        "decision": decision,
        "water_deficit_depth_mm": round(net_irrigation_depth_mm, 1),
        "total_water_needed_liters": round(total_water_liters, 0),
        "liters_per_acre": round(liters_per_acre, 0),
        "recommended_pump_hours": round(pump_hours_needed, 1),
        "advice_steps": advice_lines,
        "pump_trigger": trigger
    }

def get_shap_feature_attributions(crop: str, state: str, soil: str, irrigated: bool,
                                  n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float = 850.0,
                                  save_waterfall_path: str = None):
    if not SHAP_AVAILABLE:
        return {"status": "shap_not_installed"}

    fam_name, row = prepare_feature_row(crop, state, soil, irrigated, n_kg, p_kg, k_kg, rainfall_mm)
    model = specialists[fam_name]['model']

    if fam_name not in _SHAP_EXPLAINERS:
        try:
            _SHAP_EXPLAINERS[fam_name] = shap.TreeExplainer(model)
        except Exception:
            _SHAP_EXPLAINERS[fam_name] = shap.Explainer(model, row)

    explainer = _SHAP_EXPLAINERS[fam_name]
    shap_res = explainer(row)

    base_val_q = float(shap_res.base_values[0]) / 247.105
    shap_vals_q = shap_res.values[0] / 247.105

    friendly_labels = {
        'Fertilizer_N_kg_per_ha': 'Nitrogen (Urea)',
        'Fertilizer_P_kg_per_ha': 'Phosphorus (DAP)',
        'Fertilizer_K_kg_per_ha': 'Potash (MOP)',
        'Is_Irrigated': 'Irrigation Access',
        'Rainfall_Kharif_Monsoon': 'Seasonal Rainfall',
        'Moisture_Index': 'Root Moisture Level',
        'N_Ratio': 'Nitrogen Ratio',
        'P_Ratio': 'Phosphorus Ratio',
        'K_Ratio': 'Potash Ratio'
    }

    feature_impacts = []
    for col, val in zip(row.columns, shap_vals_q):
        val_q = round(float(val), 2)
        if abs(val_q) >= 0.02 and not col.startswith('State Name_'):
            name = friendly_labels.get(col, col.replace('Crop_', '').replace('Dominant_Soil_Type_', 'Soil: '))
            feature_impacts.append({
                "feature": name,
                "impact_q_acre": val_q,
                "direction": "boost" if val_q > 0 else "penalty"
            })

    boosters = sorted([f for f in feature_impacts if f["impact_q_acre"] > 0], key=lambda x: x["impact_q_acre"], reverse=True)
    limiters = sorted([f for f in feature_impacts if f["impact_q_acre"] < 0], key=lambda x: x["impact_q_acre"])

    if save_waterfall_path:
        try:
            plt.figure(figsize=(9, 4.5))
            shap_res.values[0] = shap_vals_q
            shap_res.base_values[0] = base_val_q
            shap.plots.waterfall(shap_res[0], show=False, max_display=6)
            plt.tight_layout()
            plt.savefig(save_waterfall_path, dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save waterfall chart: {e}")

    return {
        "status": "success",
        "regional_base_yield_q_acre": round(base_val_q, 2),
        "top_boosters": boosters[:3],
        "top_bottlenecks": limiters[:3]
    }

def run_pipeline_test():
    print("\n" + "=" * 72)
    print("      FASAL-SATHI: END-TO-END ML & EXPLAINABLE IRRIGATION HARNESS")
    print("=" * 72)

    crop = (input("1. Crop [WHEAT, RICE, MAIZE, CHICKPEA, SUGARCANE] (default: WHEAT): ").strip().upper() or "WHEAT")
    state = (input("2. State [e.g., Madhya Pradesh, Punjab, Maharashtra] (default: Madhya Pradesh): ").strip() or "Madhya Pradesh")
    soil_input = input("3. Soil Type [leave blank to auto-infer from state]: ").strip()
    soil = resolve_soil_type(state, soil_input)

    try:
        acres = float(input("4. Land Area in Acres (default: 3.0): ") or 3.0)
    except ValueError:
        acres = 3.0

    print("\n--- Current Fertilizer Application (kg/ha) ---")
    try:
        n = float(input("• Nitrogen / Urea (default: 150): ") or 150.0)
        p = float(input("• Phosphorus / DAP (default: 20): ") or 20.0)
        k = float(input("• Potash / MOP (default: 10): ") or 10.0)
    except ValueError:
        n, p, k = 150.0, 20.0, 10.0

    print("\n--- Live Telemetry & Forecast ---")
    try:
        moisture = float(input("• Live Soil Moisture Sensor % (0 - 100) (default: 28.0): ") or 28.0)
    except ValueError:
        moisture = 28.0

    try:
        rain_48h = float(input("• 48-Hour Forecast Rain in mm [e.g., 0 or 25] (default: 0.0): ") or 0.0)
    except ValueError:
        rain_48h = 0.0

    try:
        pump_hp = float(input("• Tube-well Pump HP (default: 5.0): ") or 5.0)
    except ValueError:
        pump_hp = 5.0

    is_irrigated = moisture >= 45.0
    effective_rainfall = (moisture / 100.0) * 1000.0
    benchmarks = OPTIMUM_NPK_BENCHMARKS.get(crop, {'N': 80, 'P': 40, 'K': 30, 'water_sensitive': True})

    baseline_yield = predict_raw_yield(crop, state, soil, is_irrigated, n, p, k, effective_rainfall)
    max_yield = predict_raw_yield(crop, state, soil, True, benchmarks['N'], benchmarks['P'], benchmarks['K'], effective_rainfall)
    yield_gap = max(0.0, max_yield - baseline_yield)

    irr_plan = compute_smart_irrigation_prescription(crop, acres, moisture, rain_48h, pump_hp)

    print("\n[INFO] Calculating SHAP game-theoretic feature attributions...")
    shap_data = get_shap_feature_attributions(
        crop=crop, state=state, soil=soil, irrigated=is_irrigated,
        n_kg=n, p_kg=p, k_kg=k, rainfall_mm=effective_rainfall,
        save_waterfall_path="manual_test_waterfall.png"
    )

    print("\n" + "=" * 72)
    print("                     EVALUATION REPORT")
    print("=" * 72)
    print(f"Plot Metadata: {crop} | {acres} Acres | {state} | Soil Order: {soil}")
    print(f"Field Sensors: Moisture = {moisture:.1f}% | 48h Rain Forecast = {rain_48h:.1f} mm")
    print("-" * 72)
    print(f"Predicted Baseline Yield : {baseline_yield:.2f} Q/Acre  ({baseline_yield * acres:.1f} Q Total)")
    print(f"Maximum Achievable Yield : {max_yield:.2f} Q/Acre  ({max_yield * acres:.1f} Q Total)")
    print(f"Untapped Yield Gap       : +{yield_gap:.2f} Q/Acre (+{(yield_gap / max(0.1, baseline_yield)) * 100:.1f}%)")
    print(f"Potential Extra Revenue  : ₹{yield_gap * acres * 2300:,.0f} (at MSP ₹2,300/Q)")
    print("-" * 72)

    if shap_data.get("status") == "success":
        print("\n[EXPLAINABLE AI - SHAP ATTRIBUTIONS]")
        print(f"• Regional Baseline Benchmark: {shap_data['regional_base_yield_q_acre']} Q/Acre")
        if shap_data.get("top_boosters"):
            print("• Top Yield Booster (+):")
            for item in shap_data["top_boosters"][:2]:
                print(f"    ▲ {item['feature']}: +{item['impact_q_acre']} Q/Acre")
        if shap_data.get("top_bottlenecks"):
            print("• Top Limiting Factor (-):")
            for item in shap_data["top_bottlenecks"][:2]:
                print(f"    ▼ {item['feature']}: {item['impact_q_acre']} Q/Acre")
        print("• Chart Exported: manual_test_waterfall.png")

    print("\n[SMART IRRIGATION & PUMP TELEMETRY]")
    print(f"• Decision Status : >>> {irr_plan['decision']} <<<")
    for step in irr_plan['advice_steps']:
        print(f"  - {step}")
    if irr_plan['total_water_needed_liters'] > 0:
        print(f"• Water Required  : {irr_plan['total_water_needed_liters']:,.0f} Liters ({irr_plan['liters_per_acre']:,.0f} L/Acre)")
        print(f"• Pump Runtime    : {irr_plan['recommended_pump_hours']:.1f} Hours on {pump_hp:.0f} HP pump")
        print(f"• IoT Pump Trigger: {'[TURN RELAY ON]' if irr_plan['pump_trigger'] else '[KEEP RELAY OFF]'}")

    print("\n[FERTILIZER BALANCING]")
    print(f"• Recommended Target for {crop}: N={benchmarks['N']}, P={benchmarks['P']}, K={benchmarks['K']} kg/ha")
    if n > benchmarks['N'] + 20:
        print(f"  - REDUCE Urea: Excess by {n - benchmarks['N']:.0f} kg/ha (wasted expense).")
    elif n < benchmarks['N'] - 20:
        print(f"  - INCREASE Urea: Deficit of {benchmarks['N'] - n:.0f} kg/ha.")
    if p < benchmarks['P'] - 10:
        print(f"  - ADD DAP: Deficit of {benchmarks['P'] - p:.0f} kg/ha for root development.")
    if k < benchmarks['K'] - 10:
        print(f"  - ADD Potash: Deficit of {benchmarks['K'] - k:.0f} kg/ha for grain filling.")

    print("=" * 72 + "\n")

if __name__ == "__main__":
    run_pipeline_test()