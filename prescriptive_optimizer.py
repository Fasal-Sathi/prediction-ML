import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "model2_specialists.joblib"
specialists = None
if os.path.exists(MODEL_PATH):
    specialists = joblib.load(MODEL_PATH)
else:
    print(f"Warning: {MODEL_PATH} not found. Ensure models are trained.")

try:
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

_SHAP_EXPLAINERS = {}

ECONOMICS = {
    'COSTS': {
        'UREA_PER_KG': 6.0,
        'DAP_PER_KG': 27.0,
        'MOP_PER_KG': 34.0,
        'ELECTRICITY_PER_KWH': 7.0,
        'PUMP_KW_PER_HP': 0.746,
        'SEED_PER_KG': {
            'WHEAT': 45.0, 'RICE': 50.0, 'MAIZE': 120.0, 'BARLEY': 40.0,
            'CHICKPEA': 90.0, 'MINOR PULSES': 85.0,
            'SOYABEAN': 80.0, 'GROUNDNUT': 95.0, 'SUNFLOWER': 220.0,
            'LINSEED': 70.0, 'OILSEEDS': 90.0,
            'SUGARCANE': 3.5
        }
    },
    'MSP': {
        'WHEAT': 2300.0, 'RICE': 2300.0, 'MAIZE': 2100.0, 'BARLEY': 1850.0,
        'CHICKPEA': 5440.0, 'MINOR PULSES': 6600.0,
        'SOYABEAN': 4892.0, 'GROUNDNUT': 6783.0, 'SUNFLOWER': 6760.0,
        'LINSEED': 5200.0, 'OILSEEDS': 5650.0,
        'SUGARCANE': 350.0
    }
}

AGRONOMIC_TARGETS = {
    'WHEAT':        {'N_target': 120.0, 'P_target': 60.0, 'K_target': 40.0, 'moisture_optimum_pct': 55.0, 'water_depth_mm': 30.0, 'seed_rate_kg_per_acre': 40.0, 'seed_method': 'Line sowing at 20 cm row spacing with 4-5 cm depth'},
    'RICE':         {'N_target': 100.0, 'P_target': 50.0, 'K_target': 50.0, 'moisture_optimum_pct': 70.0, 'water_depth_mm': 50.0, 'seed_rate_kg_per_acre': 15.0, 'seed_method': 'Nursery bed sowing for transplanting or 8-10 kg for DSR'},
    'MAIZE':        {'N_target': 120.0, 'P_target': 60.0, 'K_target': 40.0, 'moisture_optimum_pct': 50.0, 'water_depth_mm': 35.0, 'seed_rate_kg_per_acre': 8.0,  'seed_method': 'Ridge and furrow planting at 60x20 cm spacing'},
    'BARLEY':       {'N_target': 60.0,  'P_target': 30.0, 'K_target': 20.0, 'moisture_optimum_pct': 45.0, 'water_depth_mm': 25.0, 'seed_rate_kg_per_acre': 35.0, 'seed_method': 'Drill sowing at 22.5 cm row spacing'},
    'CHICKPEA':     {'N_target': 20.0,  'P_target': 40.0, 'K_target': 20.0, 'moisture_optimum_pct': 40.0, 'water_depth_mm': 20.0, 'seed_rate_kg_per_acre': 30.0, 'seed_method': 'Seed drill at 30 cm spacing treated with Rhizobium'},
    'MINOR PULSES': {'N_target': 20.0,  'P_target': 40.0, 'K_target': 20.0, 'moisture_optimum_pct': 40.0, 'water_depth_mm': 20.0, 'seed_rate_kg_per_acre': 10.0, 'seed_method': 'Broadcasting or line sowing at 30 cm spacing'},
    'SOYABEAN':     {'N_target': 30.0,  'P_target': 60.0, 'K_target': 40.0, 'moisture_optimum_pct': 50.0, 'water_depth_mm': 25.0, 'seed_rate_kg_per_acre': 30.0, 'seed_method': 'Broad-bed furrow sowing at 45 cm row spacing'},
    'GROUNDNUT':    {'N_target': 25.0,  'P_target': 50.0, 'K_target': 75.0, 'moisture_optimum_pct': 50.0, 'water_depth_mm': 25.0, 'seed_rate_kg_per_acre': 45.0, 'seed_method': 'Line planting with treated kernels at 30x10 cm spacing'},
    'SUNFLOWER':    {'N_target': 60.0,  'P_target': 40.0, 'K_target': 30.0, 'moisture_optimum_pct': 45.0, 'water_depth_mm': 25.0, 'seed_rate_kg_per_acre': 3.0,  'seed_method': 'Dibbling seeds at 60x30 cm spacing'},
    'LINSEED':      {'N_target': 40.0,  'P_target': 20.0, 'K_target': 20.0, 'moisture_optimum_pct': 40.0, 'water_depth_mm': 20.0, 'seed_rate_kg_per_acre': 12.0, 'seed_method': 'Line sowing at 25-30 cm row spacing'},
    'OILSEEDS':     {'N_target': 50.0,  'P_target': 30.0, 'K_target': 30.0, 'moisture_optimum_pct': 45.0, 'water_depth_mm': 25.0, 'seed_rate_kg_per_acre': 5.0,  'seed_method': 'Line sowing at 30-45 cm spacing'},
    'SUGARCANE':    {'N_target': 250.0, 'P_target': 100.0,'K_target': 120.0,'moisture_optimum_pct': 70.0, 'water_depth_mm': 50.0, 'seed_rate_kg_per_acre': 2500.0,'seed_method': 'Two/three-budded sett furrow planting at 90-120 cm spacing'}
}

CROP_ALIASES = {
    'W': 'WHEAT', 'WHEAT': 'WHEAT',
    'R': 'RICE', 'RICE': 'RICE', 'PADDY': 'RICE',
    'M': 'MAIZE', 'MAIZE': 'MAIZE', 'CORN': 'MAIZE',
    'B': 'BARLEY', 'BARLEY': 'BARLEY',
    'C': 'CHICKPEA', 'CHICKPEA': 'CHICKPEA', 'GRAM': 'CHICKPEA', 'CHANA': 'CHICKPEA',
    'P': 'MINOR PULSES', 'PULSES': 'MINOR PULSES', 'MINOR PULSES': 'MINOR PULSES',
    'SOYA': 'SOYABEAN', 'SOYABEAN': 'SOYABEAN', 'SOYBEAN': 'SOYABEAN',
    'G': 'GROUNDNUT', 'GROUNDNUT': 'GROUNDNUT', 'PEANUT': 'GROUNDNUT',
    'SF': 'SUNFLOWER', 'SUNFLOWER': 'SUNFLOWER',
    'L': 'LINSEED', 'LINSEED': 'LINSEED', 'ALSI': 'LINSEED',
    'O': 'OILSEEDS', 'OILSEEDS': 'OILSEEDS', 'MUSTARD': 'OILSEEDS',
    'S': 'SUGARCANE', 'SUGARCANE': 'SUGARCANE', 'GANA': 'SUGARCANE'
}

STATE_ALIASES = {
    'MP': 'Madhya Pradesh', 'MADHYA PRADESH': 'Madhya Pradesh',
    'PB': 'Punjab', 'PUNJAB': 'Punjab',
    'HR': 'Haryana', 'HARYANA': 'Haryana',
    'MH': 'Maharashtra', 'MAHARASHTRA': 'Maharashtra',
    'UP': 'Uttar Pradesh', 'UTTAR PRADESH': 'Uttar Pradesh',
    'RJ': 'Rajasthan', 'RAJASTHAN': 'Rajasthan',
    'GJ': 'Gujarat', 'GUJARAT': 'Gujarat',
    'BR': 'Bihar', 'BIHAR': 'Bihar',
    'WB': 'West Bengal', 'WEST BENGAL': 'West Bengal',
    'KA': 'Karnataka', 'KARNATAKA': 'Karnataka',
    'AP': 'Andhra Pradesh', 'ANDHRA PRADESH': 'Andhra Pradesh',
    'TS': 'Telangana', 'TELANGANA': 'Telangana',
    'TN': 'Tamil Nadu', 'TAMIL NADU': 'Tamil Nadu',
    'OD': 'Odisha', 'ODISHA': 'Odisha', 'ORISSA': 'Odisha'
}

STATE_DEFAULT_SOIL = {
    'MADHYA PRADESH': 'VERTIC SOILS',
    'MAHARASHTRA': 'VERTIC SOILS',
    'GUJARAT': 'VERTIC SOILS',
    'PUNJAB': 'INCEPTISOLS',
    'HARYANA': 'INCEPTISOLS',
    'UTTAR PRADESH': 'INCEPTISOLS',
    'BIHAR': 'INCEPTISOLS',
    'WEST BENGAL': 'INCEPTISOLS',
    'RAJASTHAN': 'ARIDISOLS',
    'KARNATAKA': 'ALFISOLS',
    'ANDHRA PRADESH': 'ALFISOLS',
    'TELANGANA': 'ALFISOLS',
    'TAMIL NADU': 'ALFISOLS',
    'ODISHA': 'ALFISOLS'
}

def normalize_crop_name(user_input: str) -> str:
    cleaned = (user_input or "").strip().upper()
    return CROP_ALIASES.get(cleaned, cleaned or "WHEAT")

def normalize_state_name(user_input: str) -> str:
    cleaned = (user_input or "").strip().upper()
    return STATE_ALIASES.get(cleaned, (user_input or "Madhya Pradesh").strip().title())

def resolve_soil_type(state: str, soil: str = None) -> str:
    if soil and soil.strip().upper() not in ['', 'NONE', 'UNKNOWN', "DON'T KNOW"]:
        return soil.strip().upper()
    return STATE_DEFAULT_SOIL.get(state.strip().upper(), 'INCEPTISOLS')

def prepare_feature_row(crop: str, state: str, soil: str, irrigated: bool,
                        n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float = 850.0):
    crop_upper = normalize_crop_name(crop)
    fam_name = 'cereals'
    for f_name, data in specialists.items():
        if crop_upper in data['crops']:
            fam_name = f_name
            break
            
    engine = specialists[fam_name]
    feature_names = engine['feature_names']
    
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)
    total_npk = max(1.0, n_kg + p_kg + k_kg)
    irr_flag = 1.0 if irrigated else 0.0
    
    row['Crop_Irrigated_Area_ha'] = irr_flag
    row['Is_Irrigated'] = irr_flag
    row['Total_District_Irrigated_1000_ha'] = 50.0
    row['Fertilizer_N_kg_per_ha'] = n_kg
    row['Fertilizer_P_kg_per_ha'] = p_kg
    row['Fertilizer_K_kg_per_ha'] = k_kg
    row['Fertilizer_NPK_Total_kg_per_ha'] = total_npk
    row['N_Ratio'] = n_kg / total_npk
    row['P_Ratio'] = p_kg / total_npk
    row['K_Ratio'] = k_kg / total_npk
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
                      n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float = 850.0) -> float:
    fam_name, row = prepare_feature_row(crop, state, soil, irrigated, n_kg, p_kg, k_kg, rainfall_mm)
    model = specialists[fam_name]['model']
    pred_kg_ha = max(0.0, float(model.predict(row)[0]))
    return pred_kg_ha / 247.105

def compute_shap_explainability(crop: str, state: str, soil: str, irrigated: bool,
                                n_kg: float, p_kg: float, k_kg: float, rainfall_mm: float):
    if not SHAP_AVAILABLE:
        return {"active": False, "reason": "SHAP library not installed"}
    
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
    
    attributions = []
    for col, val in zip(row.columns, shap_vals_q):
        val_q = round(float(val), 2)
        if abs(val_q) >= 0.05 and not col.startswith('State Name_'):
            attributions.append({
                "feature": col.replace('Fertilizer_', '').replace('Crop_', '').replace('Dominant_Soil_Type_', 'Soil: '),
                "impact_q_acre": val_q,
                "direction": "BOOST (+)" if val_q > 0 else "BOTTLENECK (-)"
            })
            
    boosters = sorted([a for a in attributions if a['impact_q_acre'] > 0], key=lambda x: x['impact_q_acre'], reverse=True)
    limiters = sorted([a for a in attributions if a['impact_q_acre'] < 0], key=lambda x: x['impact_q_acre'])
    
    return {
        "active": True,
        "regional_baseline_q_acre": round(base_val_q, 2),
        "top_booster": boosters[0] if boosters else None,
        "top_bottleneck": limiters[0] if limiters else None
    }

def generate_decision_audit(crop: str, state: str, soil: str, acres: float,
                            applied_n: float, applied_p: float, applied_k: float,
                            soil_moisture_pct: float, forecast_rain_mm_48h: float,
                            farmer_seed_kg_total: float, pump_hp: float = 5.0):
    crop_upper = normalize_crop_name(crop)
    norm_state = normalize_state_name(state)
    agro = AGRONOMIC_TARGETS.get(crop_upper, AGRONOMIC_TARGETS['WHEAT'])
    msp = ECONOMICS['MSP'].get(crop_upper, 2300.0)
    resolved_soil = resolve_soil_type(norm_state, soil)
    
    is_irrigated = soil_moisture_pct >= 45.0
    effective_rainfall = (soil_moisture_pct / 100.0) * 1000.0

    base_yield = predict_raw_yield(crop_upper, norm_state, resolved_soil, is_irrigated, applied_n, applied_p, applied_k, effective_rainfall)
    base_harvest_q = base_yield * acres

    target_n = agro['N_target']
    delta_n = target_n - applied_n
    step_yield_n = predict_raw_yield(crop_upper, norm_state, resolved_soil, is_irrigated, target_n, applied_p, applied_k, effective_rainfall)
    dy_n = max(0.0, step_yield_n - base_yield)
    cost_n = (delta_n * acres) * ECONOMICS['COSTS']['UREA_PER_KG']
    action_n = "DECREASE" if delta_n < 0 else ("INCREASE" if delta_n > 0 else "MAINTAIN")
    impact_n = f"CASH SAVING of ₹{abs(cost_n):,.0f}" if delta_n < 0 else ("PROFIT GAIN" if delta_n > 0 else "OPTIMAL")
    why_n = "Excess Nitrogen causes vegetative lodging without grain fill." if delta_n < 0 else "Provides chlorophyll for photosynthesis."

    target_p = agro['P_target']
    delta_p = target_p - applied_p
    step_yield_p = predict_raw_yield(crop_upper, norm_state, resolved_soil, is_irrigated, applied_n, target_p, applied_k, effective_rainfall)
    dy_p = max(0.0, step_yield_p - base_yield)
    cost_p = (delta_p * acres) * ECONOMICS['COSTS']['DAP_PER_KG']
    action_p = "INCREASE" if delta_p > 0 else ("DECREASE" if delta_p < 0 else "MAINTAIN")
    impact_p = "HIGH RETURN" if delta_p > 0 else (f"CASH SAVING of ₹{abs(cost_p):,.0f}" if delta_p < 0 else "OPTIMAL")
    why_p = "Stimulates crown root expansion to pull subsoil moisture." if delta_p > 0 else "Maintains phosphorus equilibrium."

    target_k = agro['K_target']
    delta_k = target_k - applied_k
    step_yield_k = predict_raw_yield(crop_upper, norm_state, resolved_soil, is_irrigated, applied_n, applied_p, target_k, effective_rainfall)
    dy_k = max(0.0, step_yield_k - base_yield)
    cost_k = (delta_k * acres) * ECONOMICS['COSTS']['MOP_PER_KG']
    action_k = "INCREASE" if delta_k > 0 else ("DECREASE" if delta_k < 0 else "MAINTAIN")
    impact_k = "HIGH RETURN" if delta_k > 0 else (f"CASH SAVING of ₹{abs(cost_k):,.0f}" if delta_k < 0 else "OPTIMAL")
    why_k = "Hardens stalk against lodging and ensures full grain weight." if delta_k > 0 else "Potash concentration is optimal."

    target_water_depth = agro['water_depth_mm']
    current_water_depth = (soil_moisture_pct / agro['moisture_optimum_pct']) * target_water_depth
    deficit_mm = max(0.0, target_water_depth - current_water_depth)
    rain_effective = forecast_rain_mm_48h * 0.85
    net_irr_mm = max(0.0, deficit_mm - rain_effective)
    
    total_water_liters = net_irr_mm * 4047.0 * acres
    pump_flow_lph = pump_hp * 5000.0
    pump_hours = (total_water_liters / pump_flow_lph) if total_water_liters > 0 else 0.0

    step_yield_irr = predict_raw_yield(crop_upper, norm_state, resolved_soil, True, applied_n, applied_p, applied_k, effective_rainfall)
    dy_irr = max(0.0, step_yield_irr - base_yield) if net_irr_mm > 0 else 0.0

    if forecast_rain_mm_48h >= deficit_mm and deficit_mm > 0:
        action_irr = "HOLD"
        impact_irr = "SAVE ELECTRICITY"
        why_irr = f"Incoming 48h rain ({forecast_rain_mm_48h:.1f} mm) satisfies water needs."
        pump_trigger = False
    elif net_irr_mm > 0:
        action_irr = "INCREASE"
        impact_irr = "CRITICAL MOISTURE RECOVERY"
        why_irr = "Water deficit stops root nutrient absorption; irrigation needed."
        pump_trigger = True
    else:
        action_irr = "MAINTAIN"
        impact_irr = "OPTIMAL"
        why_irr = "Moisture level in root zone is optimal."
        pump_trigger = False

    test_n = max(applied_n, target_n) if delta_n >= 0 else target_n
    test_p = max(applied_p, target_p)
    test_k = max(applied_k, target_k)
    raw_opt_yield = predict_raw_yield(crop_upper, norm_state, resolved_soil, True, test_n, test_p, test_k, effective_rainfall)
    
    if raw_opt_yield >= base_yield:
        optimal_yield = raw_opt_yield
        mode = "YIELD_EXPANSION"
    else:
        optimal_yield = base_yield
        mode = "COST_OPTIMIZATION"

    yield_gap_val = max(0.0, optimal_yield - base_yield)
    optimal_harvest_total = optimal_yield * acres
    harvest_gain_total = max(0.0, optimal_harvest_total - base_harvest_q)
    gross_recoverable = harvest_gain_total * msp
    pct_increase = (yield_gap_val / max(0.01, base_yield)) * 100.0

    impact_candidates = [
        ("Phosphorus (DAP)", dy_p),
        ("Potash (MOP)", dy_k),
        ("Nitrogen (Urea)", dy_n),
        ("Irrigation Water", dy_irr)
    ]
    dominant_factor_name, top_boost = max(impact_candidates, key=lambda x: x[1])
    if mode == "COST_OPTIMIZATION" or top_boost <= 0.01:
        dominant_factor_name = "Yield Preservation & Input Efficiency"
        top_boost = 0.0

    rec_seed_per_acre = agro.get('seed_rate_kg_per_acre', 40.0)
    target_seed_total_kg = rec_seed_per_acre * acres
    seed_cost_kg = ECONOMICS['COSTS']['SEED_PER_KG'].get(crop_upper, 45.0)
    seed_delta_kg = target_seed_total_kg - farmer_seed_kg_total
    
    if seed_delta_kg < -2.0:
        seed_action = "DECREASE"
        seed_diff = abs(seed_delta_kg)
        seed_financial = f"CASH SAVINGS of ₹{seed_diff * seed_cost_kg:,.0f}"
        seed_diagnosis = f"You are over-seeding by {seed_diff:.1f} kg. Excessive plant density causes high intra-crop competition for light and nutrients, resulting in weak, thin stalks."
    elif seed_delta_kg > 2.0:
        seed_action = "INCREASE"
        seed_diff = seed_delta_kg
        seed_financial = f"INVESTMENT of ₹{seed_diff * seed_cost_kg:,.0f}"
        seed_diagnosis = f"You are under-seeding by {seed_diff:.1f} kg. Low seedling population leaves gaps in your field canopy, permitting weed dominance and losing potential yield."
    else:
        seed_action = "MAINTAIN"
        seed_diff = 0.0
        seed_financial = "OPTIMAL"
        seed_diagnosis = "Your seed rate matches scientific plant population standards perfectly."

    shap_results = compute_shap_explainability(crop_upper, norm_state, resolved_soil, is_irrigated, applied_n, applied_p, applied_k, effective_rainfall)

    pros = []
    cons = []
    if applied_n >= target_n * 0.8:
        pros.append("Solid vegetative vigor base from initial Nitrogen application.")
    if applied_n > target_n + 20:
        cons.append(f"Excess Urea of {applied_n - target_n:.0f} kg/ha: increases lodging risk and wastes expenditure.")
    if applied_p < target_p - 10:
        cons.append(f"DAP deficiency of {target_p - applied_p:.0f} kg/ha: limits lateral root expansion.")
    else:
        pros.append("Adequate root branching support from Phosphorus.")
    if applied_k < target_k - 10:
        cons.append(f"Potash deficiency of {target_k - applied_k:.0f} kg/ha: causes weak grain filling and thin stems.")
    if soil_moisture_pct < 35.0 and forecast_rain_mm_48h == 0.0:
        cons.append(f"Low soil moisture ({soil_moisture_pct:.1f}%): limits fertilizer uptake.")

    factor_ledger = [
        {
            "factor_name": "Nitrogen (Urea)",
            "current_value": applied_n,
            "target_value": target_n,
            "action": action_n,
            "delta_units": round(abs(delta_n), 1),
            "unit": "kg/ha",
            "marginal_yield_delta_q_acre": round(dy_n, 2),
            "financial_impact": impact_n,
            "biological_why": why_n
        },
        {
            "factor_name": "Phosphorus (DAP)",
            "current_value": applied_p,
            "target_value": target_p,
            "action": action_p,
            "delta_units": round(abs(delta_p), 1),
            "unit": "kg/ha",
            "marginal_yield_delta_q_acre": round(dy_p, 2),
            "financial_impact": impact_p,
            "biological_why": why_p
        },
        {
            "factor_name": "Potash (MOP)",
            "current_value": applied_k,
            "target_value": target_k,
            "action": action_k,
            "delta_units": round(abs(delta_k), 1),
            "unit": "kg/ha",
            "marginal_yield_delta_q_acre": round(dy_k, 2),
            "financial_impact": impact_k,
            "biological_why": why_k
        },
        {
            "factor_name": "Irrigation Water",
            "current_value": soil_moisture_pct,
            "target_value": agro['moisture_optimum_pct'],
            "action": action_irr,
            "delta_units": round(total_water_liters, 0),
            "unit": "Liters",
            "pump_hours": round(pump_hours, 1),
            "marginal_yield_delta_q_acre": round(dy_irr, 2),
            "financial_impact": impact_irr,
            "biological_why": why_irr,
            "pump_relay_trigger": pump_trigger
        }
    ]

    return {
        "status": "success",
        "plot_summary": {
            "crop": crop_upper,
            "state": norm_state,
            "soil_type": resolved_soil,
            "acres": acres,
            "mode": mode,
            "base_yield_q_acre": round(base_yield, 2),
            "optimal_yield_q_acre": round(optimal_yield, 2),
            "yield_gap_q_acre": round(yield_gap_val, 2),
            "base_harvest_total_q": round(base_harvest_q, 1),
            "optimal_harvest_total_q": round(optimal_harvest_total, 1),
            "harvest_gain_total_q": round(harvest_gain_total, 1),
            "pct_increase": round(pct_increase, 1),
            "primary_driver": dominant_factor_name,
            "primary_driver_boost_q": round(top_boost, 2),
            "gross_grain_value_inr": round(gross_recoverable, 0)
        },
        "seed_audit": {
            "farmer_input_seed_kg": round(farmer_seed_kg_total, 1),
            "recommended_total_seed_kg": round(target_seed_total_kg, 1),
            "seed_rate_per_acre_kg": rec_seed_per_acre,
            "action": seed_action,
            "delta_units_kg": round(seed_diff, 1),
            "financial_verdict": seed_financial,
            "diagnosis": seed_diagnosis,
            "sowing_protocol": agro.get('seed_method')
        },
        "shap_explainability": shap_results,
        "scorecard": {
            "pros": pros,
            "cons": cons,
            "highest_wasted_effort": "Urea Overdose" if delta_n < -20 else "None",
            "highest_missed_return": "Irrigation Access" if net_irr_mm > 5.0 else "Phosphorus (DAP)"
        },
        "factor_breakdown": factor_ledger
    }
