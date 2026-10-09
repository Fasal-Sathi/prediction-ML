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

# Agronomic Benchmark Targets (ICAR standards in kg/ha)
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

def predict_raw_yield(crop: str, state: str, soil: str, irrigated: bool,
                      n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float = 850.0):
    crop_upper = crop.strip().upper()
    
    fam_name = 'cereals'
    for f_name, data in specialists.items():
        if crop_upper in data['crops']:
            fam_name = f_name
            break
            
    engine = specialists[fam_name]
    model = engine['model']
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
        
    for prefix, name in [('Crop_', crop_upper), ('State Name_', state), ('Dominant_Soil_Type_', soil)]:
        col = f"{prefix}{name}"
        if col in row.columns:
            row[col] = 1.0
            
    pred_kg_ha = max(0.0, float(model.predict(row)[0]))
    return pred_kg_ha / 247.105  # Convert kg/ha to Quintals/Acre

def compute_smart_irrigation_prescription(crop: str, acres: float, soil_moisture_pct: float,
                                         forecast_rain_mm_48h: float, pump_hp: float = 5.0):
    """
    Translates soil sensor readings and weather forecasts into concrete Liters & Pump Run Hours.
    1 mm water over 1 acre = 4,047 Liters.
    5 HP Agricultural Pump Discharge = ~25,000 Liters/hour.
    """
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
        status = "HOLD"
        advice_lines.append(f"Weather forecast predicts {forecast_rain_mm_48h:.1f} mm rain (~{rain_liters_expected:,.0f} Liters across plot) in next 48 hrs.")
        advice_lines.append("Natural rainfall will satisfy crop root water needs.")
        advice_lines.append("Pumping water now will waste electricity and cause waterlogging.")
        trigger = False
    elif forecast_rain_mm_48h > 0 and net_irrigation_depth_mm > 0:
        decision = "PARTIAL SUPPLEMENTAL IRRIGATION"
        status = "PARTIAL"
        advice_lines.append(f"Weather API shows light rain of {forecast_rain_mm_48h:.1f} mm (~{rain_liters_expected:,.0f} Liters) coming.")
        advice_lines.append(f"Rain is insufficient. Provide {total_water_liters:,.0f} Liters ({liters_per_acre:,.0f} L/Acre).")
        advice_lines.append(f"Run {pump_hp:.0f} HP pump for ~{pump_hours_needed:.1f} hours ({int(pump_hours_needed * 60)} minutes).")
        trigger = True
    elif net_irrigation_depth_mm > 0:
        decision = "CRITICAL IRRIGATION REQUIRED"
        status = "CRITICAL"
        advice_lines.append(f"No rain in 48-hr forecast. Soil moisture is low ({soil_moisture_pct:.1f}%).")
        advice_lines.append(f"Apply {total_water_liters:,.0f} Liters across {acres:.1f} Acres ({liters_per_acre:,.0f} L/Acre).")
        advice_lines.append(f"Run {pump_hp:.0f} HP pump for {pump_hours_needed:.1f} hours (~{pump_hours_needed/acres:.1f} hrs/acre).")
        trigger = True
    else:
        decision = "OPTIMAL MOISTURE - NO IRRIGATION NEEDED"
        status = "OPTIMAL"
        advice_lines.append(f"Soil moisture is in optimal zone ({soil_moisture_pct:.1f}%).")
        advice_lines.append("Re-check moisture readings in 48 hours.")
        trigger = False
        
    return {
        "decision": decision,
        "status": status,
        "water_deficit_depth_mm": round(net_irrigation_depth_mm, 1),
        "total_water_needed_liters": round(total_water_liters, 0),
        "liters_per_acre": round(liters_per_acre, 0),
        "recommended_pump_hours": round(pump_hours_needed, 1),
        "advice_steps": advice_lines,
        "pump_trigger": trigger
    }

def generate_prescriptive_report(crop: str, state: str, soil: str, acres: float,
                                 current_irrigated: bool, current_n: float, current_p: float, current_k: float,
                                 soil_moisture_pct: float = 35.0, forecast_rain_mm_48h: float = 0.0,
                                 rainfall_mm: float = 850.0, pump_hp: float = 5.0):
    crop_upper = crop.strip().upper()
    benchmarks = OPTIMUM_NPK_BENCHMARKS.get(crop_upper, {'N': 80, 'P': 40, 'K': 30, 'water_sensitive': True})
    
    baseline_yield = predict_raw_yield(crop, state, soil, current_irrigated, current_n, current_p, current_k, rainfall_mm)
    yield_if_irrigated = predict_raw_yield(crop, state, soil, True, current_n, current_p, current_k, rainfall_mm)
    irrigation_gain = max(0.0, yield_if_irrigated - baseline_yield) if not current_irrigated else 0.0
    
    opt_n, opt_p, opt_k = benchmarks['N'], benchmarks['P'], benchmarks['K']
    yield_if_fert_opt = predict_raw_yield(crop, state, soil, current_irrigated, opt_n, opt_p, opt_k, rainfall_mm)
    fert_gain = max(0.0, yield_if_fert_opt - baseline_yield)
    
    max_potential_yield = predict_raw_yield(crop, state, soil, True, opt_n, opt_p, opt_k, rainfall_mm)
    total_possible_gain = max(0.0, max_potential_yield - baseline_yield)
    
    n_diff = current_n - opt_n
    p_diff = current_p - opt_p
    k_diff = current_k - opt_k
    
    fert_action = []
    if n_diff > 25:
        fert_action.append(f"REDUCE Nitrogen (Urea) by {abs(n_diff):.0f} kg/ha. You are over-applying nitrogen with zero yield return.")
    elif n_diff < -20:
        fert_action.append(f"INCREASE Nitrogen by {abs(n_diff):.0f} kg/ha to support crop vegetative growth.")
    if p_diff < -15:
        fert_action.append(f"ADD Phosphorus (DAP) by {abs(p_diff):.0f} kg/ha for healthy root initiation.")
    if k_diff < -15:
        fert_action.append(f"ADD Potash (MOP) by {abs(k_diff):.0f} kg/ha to fill grains and prevent lodging.")
    if not fert_action:
        fert_action.append("Current N-P-K nutrient application is balanced.")
        
    # Get physical water requirements
    irr_plan = compute_smart_irrigation_prescription(crop, acres, soil_moisture_pct, forecast_rain_mm_48h, pump_hp)
    
    print("\n" + "="*68)
    print("      FARMER ACTIONABLE PRESCRIPTION & OPTIMIZATION REPORT")
    print("="*68)
    print(f"Crop: {crop_upper} | Plot: {acres} Acres | State: {state} | Soil: {soil}")
    print(f"Sensor Moisture: {soil_moisture_pct:.1f}% | 48h Weather Rain Forecast: {forecast_rain_mm_48h:.1f} mm")
    print(f"Applied N-P-K: {current_n:.0f}-{current_p:.0f}-{current_k:.0f} kg/ha")
    print("-"*68)
    print(f" CURRENT PREDICTED YIELD:   {baseline_yield:.2f} Quintals/Acre  ({baseline_yield * acres:.1f} Q Total)")
    print(f" MAXIMUM ACHIEVABLE YIELD: {max_potential_yield:.2f} Quintals/Acre  ({max_potential_yield * acres:.1f} Q Total)")
    print(f" UNTAPPED YIELD GAP:       +{total_possible_gain:.2f} Quintals/Acre (+{(total_possible_gain / max(0.1, baseline_yield)) * 100:.1f}%)")
    print("-"*68)
    
    print("\n 1. FERTILIZER PRESCRIPTION:")
    print(f"    Target N-P-K for {crop_upper}: {opt_n} - {opt_p} - {opt_k} kg/ha")
    for act in fert_action:
        print(f"    • {act}")
    if fert_gain > 0:
        print(f"    Expected Gain from N-P-K Correction: +{fert_gain:.2f} Quintals/Acre")

    print("\n 2. SMART IRRIGATION PRESCRIPTION (WEATHER & SENSOR INTEGRATED):")
    print(f"    DECISION: >>> {irr_plan['decision']} <<<")
    for line in irr_plan['advice_steps']:
        print(f"    • {line}")
    if irr_plan['total_water_needed_liters'] > 0:
        print(f"    • Total Water Required: {irr_plan['total_water_needed_liters']:,.0f} Liters ({irr_plan['liters_per_acre']:,.0f} L/Acre)")
        print(f"    • Tube-well Runtime: {irr_plan['recommended_pump_hours']:.1f} Hours on a {pump_hp:.0f} HP pump")

    print("\n 3. EXPECTED FINANCIAL BENEFIT:")
    extra_q = total_possible_gain * acres
    extra_rev = extra_q * 2300
    print(f"    Estimated Net Grain Gain: ~{extra_q:.1f} Quintals across your {acres} Acres.")
    print(f"    Estimated Revenue Increase: ~₹{extra_rev:,.0f} (at baseline MSP ₹2,300/Q).")
    print("="*68 + "\n")

def process_live_request(data: dict) -> dict:
    """
    Clean JSON API handler for Backend & ESP32.
    """
    crop = data.get('crop', 'WHEAT').upper()
    state = data.get('state', 'Madhya Pradesh')
    soil = data.get('soil', 'VERTIC SOILS')
    acres = float(data.get('acres', 1.0))
    pump_hp = float(data.get('pump_hp', 5.0))
    
    # Sensor telemetry + Weather API values
    soil_moisture = float(data.get('soil_moisture_pct', 50.0))
    forecast_rain_48h = float(data.get('forecast_rain_mm_48h', 0.0))
    
    # N-P-K from user
    current_n = float(data.get('applied_n_kg_ha', 100.0))
    current_p = float(data.get('applied_p_kg_ha', 40.0))
    current_k = float(data.get('applied_k_kg_ha', 20.0))
    
    is_irrigated = soil_moisture >= 45.0
    effective_rainfall = (soil_moisture / 100.0) * 1000.0
    
    benchmarks = OPTIMUM_NPK_BENCHMARKS.get(crop, {'N': 80, 'P': 40, 'K': 30, 'water_sensitive': True})
    
    baseline_yield = predict_raw_yield(crop, state, soil, is_irrigated, current_n, current_p, current_k, effective_rainfall)
    max_yield = predict_raw_yield(crop, state, soil, True, benchmarks['N'], benchmarks['P'], benchmarks['K'], effective_rainfall)
    yield_gap = max(0.0, max_yield - baseline_yield)
    
    irr_plan = compute_smart_irrigation_prescription(crop, acres, soil_moisture, forecast_rain_48h, pump_hp)
    
    actions = []
    if current_n > benchmarks['N'] + 20:
        actions.append(f"Reduce Nitrogen by {current_n - benchmarks['N']:.0f} kg/ha")
    elif current_n < benchmarks['N'] - 20:
        actions.append(f"Add Nitrogen by {benchmarks['N'] - current_n:.0f} kg/ha")
        
    if current_p < benchmarks['P'] - 10:
        actions.append(f"Add DAP by {benchmarks['P'] - current_p:.0f} kg/ha")
    if current_k < benchmarks['K'] - 10:
        actions.append(f"Add Potash by {benchmarks['K'] - current_k:.0f} kg/ha")
        
    return {
        "status": "success",
        "crop": crop,
        "plot_acres": acres,
        "predictions": {
            "current_yield_q_acre": round(baseline_yield, 2),
            "max_yield_q_acre": round(max_yield, 2),
            "yield_gap_q_acre": round(yield_gap, 2),
            "total_harvest_quintals": round(baseline_yield * acres, 1),
            "potential_extra_revenue_inr": round(yield_gap * acres * 2300, 0)
        },
        "fertilizer_prescriptions": actions,
        "irrigation_prescription": irr_plan,
        "pump_trigger": irr_plan["pump_trigger"]
    }

if __name__ == '__main__':
    # Test run: Wheat with low soil moisture and rain forecast
    generate_prescriptive_report(
        crop='WHEAT',
        state='Madhya Pradesh',
        soil='VERTIC SOILS',
        acres=3.0,
        current_irrigated=False,
        current_n=150.0,
        current_p=20.0,
        current_k=10.0,
        soil_moisture_pct=28.0,
        forecast_rain_mm_48h=0.0,
        rainfall_mm=650.0,
        pump_hp=5.0
    )