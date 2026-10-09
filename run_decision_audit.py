from prescriptive_optimizer import generate_decision_audit, normalize_crop_name, normalize_state_name, AGRONOMIC_TARGETS

def main():
    print("\n" + "=" * 78)
    print("      FASAL-SATHI: INTERACTIVE AGRO-ECONOMIC DECISION ENGINE")
    print("=" * 78)
    print("SUPPORTED CROPS: Wheat, Rice, Maize, Barley, Chickpea, Minor Pulses,")
    print("                 Soyabean, Groundnut, Sunflower, Linseed, Oilseeds, Sugarcane")
    print("                 (Shortcuts work: w, r, m, b, c, soya, g, sf, l, o, s)")
    print("-" * 78)

    raw_crop = input("1. Enter Crop Name or Shortcut (default: WHEAT): ").strip()
    crop = normalize_crop_name(raw_crop)

    raw_state = input("2. Enter State Name or Code [e.g., MP, PB, MH, UP, HR, RJ, GJ, KA] (default: Madhya Pradesh): ").strip()
    state = normalize_state_name(raw_state)

    soil = input("3. Soil Type [leave blank for auto-detected]: ").strip()
    acres = float(input("4. Plot Size in Acres (default: 3.0): ") or 3.0)

    # Dynamic default seed rate lookup based on chosen crop
    rec_seed_rate = AGRONOMIC_TARGETS.get(crop, {}).get('seed_rate_kg_per_acre', 40.0)
    def_seed = rec_seed_rate * acres

    print("\n--- Farmer Seed Quantity Input ---")
    farmer_seed = float(input(f"• Total Seed Quantity YOU are using for {acres:.1f} Acres in kg (default: {def_seed:.1f}): ") or def_seed)

    print("\n--- Current Fertilizer Application (kg/ha) ---")
    n = float(input("• Nitrogen / Urea applied (default: 120): ") or 120.0)
    p = float(input("• Phosphorus / DAP applied (default: 50): ") or 50.0)
    k = float(input("• Potash / MOP applied (default: 40): ") or 40.0)

    print("\n--- Field Telemetry ---")
    moisture = float(input("• Soil Moisture % (default: 28.0): ") or 28.0)
    rain = float(input("• 48h Rain Forecast mm (default: 0.0): ") or 0.0)
    hp = float(input("• Pump HP (default: 5.0): ") or 5.0)

    res = generate_decision_audit(
        crop=crop, state=state, soil=soil, acres=acres,
        applied_n=n, applied_p=p, applied_k=k,
        soil_moisture_pct=moisture, forecast_rain_mm_48h=rain,
        farmer_seed_kg_total=farmer_seed, pump_hp=hp
    )

    summary = res["plot_summary"]
    score = res["scorecard"]
    seed = res["seed_audit"]
    shap_info = res["shap_explainability"]

    print("\n" + "=" * 78)
    print("                      1. FIELD SCORECARD & AUDIT")
    print("=" * 78)
    print(f"Plot Metadata: {summary['crop']} | {summary['acres']} Acres | {summary['state']} | Soil: {summary['soil_type']}")
    print("-" * 78)
    print("WHAT YOU DID RIGHT (PROS):")
    if score["pros"]:
        for item in score["pros"]:
            print(f"  [+] {item}")
    else:
        print("  [-] None detected.")

    print("\nWHAT YOU DID WRONG / MISSED (CONS):")
    if score["cons"]:
        for item in score["cons"]:
            print(f"  [-] {item}")
    else:
        print("  [+] No major errors detected.")

    print("\nEFFORT vs RETURN DIAGNOSTIC:")
    print(f"  • Most Wasted Effort  : {score['highest_wasted_effort']}")
    print(f"  • Highest Profit Gain : {score['highest_missed_return']}")

    print("\n" + "=" * 78)
    print("          2. FACTOR-BY-FACTOR MARGINAL OPTIMIZATION LEDGER")
    print("=" * 78)

    for factor in res["factor_breakdown"]:
        print(f"\n▶ [{factor['factor_name'].upper()}]")
        print(f"   • Recommended Action   : >>> {factor['action']} by {factor['delta_units']} {factor['unit']} <<<")
        print(f"   • Marginal Yield Delta : {factor['marginal_yield_delta_q_acre']:+.2f} Quintals/Acre")
        print(f"   • Financial Verdict    : {factor['financial_impact']}")
        if "pump_hours" in factor:
            print(f"   • Pump Operation       : Run pump for {factor['pump_hours']} Hours")
            print(f"   • IoT Relay Control    : {'[TURN ON RELAY]' if factor['pump_relay_trigger'] else '[KEEP RELAY OFF]'}")
        print(f"   • Physiological Reason : {factor['biological_why']}")

    print("\n" + "=" * 78)
    print("              3. SEED QUANTITY & FARMER SEED RATE AUDIT")
    print("=" * 78)
    print(f"• Crop Selected                : {summary['crop']}")
    print(f"• Farmer's Input Seed Quantity : {seed['farmer_input_seed_kg']} kg on {acres:.1f} Acres ({seed['farmer_input_seed_kg']/acres:.1f} kg/Acre)")
    print(f"• Recommended Standard Seed    : {seed['recommended_total_seed_kg']} kg on {acres:.1f} Acres ({seed['seed_rate_per_acre_kg']} kg/Acre)")
    print(f"• Recommended Action           : >>> {seed['action']} by {seed['delta_units_kg']} kg <<<")
    print(f"• Economic Impact              : {seed['financial_verdict']}")
    print(f"• Agronomic Diagnosis          : {seed['diagnosis']}")
    print(f"• Recommended Sowing Protocol  : {seed['sowing_protocol']}")

    print("\n" + "=" * 78)
    print("           4. SHAP EXPLAINABILITY ENGINE STATUS & VERIFICATION")
    print("=" * 78)
    if shap_info.get("active"):
        print("• SHAP Status                 : ACTIVE & RUNNING (Direct TreeExplainer Hook)")
        print(f"• Regional Historical Base    : {shap_info['regional_baseline_q_acre']} Quintals/Acre")
        if shap_info.get("top_booster"):
            print(f"• Highest Mathematical Boost  : ▲ {shap_info['top_booster']['feature']} (+{shap_info['top_booster']['impact_q_acre']} Q/Acre)")
        if shap_info.get("top_bottleneck"):
            print(f"• Single Biggest Bottleneck   : ▼ {shap_info['top_bottleneck']['feature']} ({shap_info['top_bottleneck']['impact_q_acre']} Q/Acre)")
    else:
        print(f"• SHAP Status                 : INACTIVE ({shap_info.get('reason')})")

    print("\n" + "=" * 78)
    print("                 5. YIELD PREDICTED & HARVEST IMPACT AUDIT")
    print("=" * 78)
    print(f"• Current Farmer Baseline Yield   : {summary['base_yield_q_acre']} Quintals/Acre  ({summary['base_harvest_total_q']} Q Total)")
    print(f"• AI/ML Guided Optimal Yield      : {summary['optimal_yield_q_acre']} Quintals/Acre  ({summary['optimal_harvest_total_q']} Q Total)")
    print(f"• Expected Harvest Increase       : +{summary['yield_gap_q_acre']} Quintals/Acre  (+{summary['harvest_gain_total_q']} Q across {acres:.1f} Acres)")
    print(f"• Percentage Yield Growth         : ▲ +{summary['pct_increase']}% Increase")
    print(f"• Primary Engine of Growth        : {summary['primary_driver']}")
    print("-" * 78)
    print("HOW THIS TRANSFORMATION WORKS:")
    if summary['mode'] == "COST_OPTIMIZATION":
        print(f"  1. High Baseline Retention: Your current setup produces an excellent {summary['base_harvest_total_q']} Quintals.")
        print(f"  2. Protection Strategy: The AI locks in your top harvest of {summary['optimal_harvest_total_q']} Quintals without reduction.")
        print(f"  3. Cost Optimization: Recommendations focus on cutting excess wasted fertilizer to save money without dropping yield.")
    else:
        print(f"  1. Current Output: Based on your current field setup, your plot yields {summary['base_harvest_total_q']} Quintals.")
        print(f"  2. Suggested AI Output: By adopting the recommendations, your plot reaches {summary['optimal_harvest_total_q']} Quintals.")
        print(f"  3. Total Increase: You gain +{summary['harvest_gain_total_q']} extra Quintals (+{summary['pct_increase']}% growth).")
        print(f"  4. Main Driver: The primary factor lifting your yield is {summary['primary_driver']}.")
        print(f"  5. Financial Value: That extra harvest is worth ~₹{summary['gross_grain_value_inr']:,.0f} at standard MSP.")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
