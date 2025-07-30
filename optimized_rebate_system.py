import streamlit as st
import pandas as pd
from pulp import *
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# --- Program Configuration ---
class RebateConfig:
    """Configuration class for all rebate programs"""
    
    # Program caps by income level
    HOMES_CAP = {"Low": 16000, "Moderate": 4000}
    HEAR_CAP = 14000
    WAP_CAP = 9000
    
    # Eligible measures for each program
    HOMES_ELIGIBLE = {
        "HEAT_PUMP", "HEAT_PUMP_WATER_HEATER", "INSULATION", "AIR_SEALING",
        "DUCT_INSULATION", "PIPE_INSULATION", "CONTROLS_SYSTEM", "LED",
        "WASHER_DRYER", "STOVE", "COOKTOP", "RANGE", "OVEN", "VENTILATION"
    }
    
    # HEAR item-specific caps
    HEAR_ITEM_CAPS = {
        "HEAT_PUMP": 8000,
        "HEAT_PUMP_WATER_HEATER": 1750,
        "HEAT_PUMP_DRYER": 840,
        "COOKTOP": 840,
        "RANGE": 840,
        "OVEN": 840,
        "STOVE": 840,
        "PANEL_UPGRADE": 4000,
        "WIRING": 2500,
        "INSULATION": 1600,
        "AIR_SEALING": 1600,
        "VENTILATION": 1600
    }
    
    HEAR_ELIGIBLE = set(HEAR_ITEM_CAPS.keys())
    
    WAP_ELIGIBLE = {
        "AIR_SEALING", "INSULATION", "DUCT_INSULATION", "PIPE_INSULATION", "CONTROLS_SYSTEM",
        "STORM_WINDOW", "STORM_DOOR", "REPLACEMENT_WINDOW", "REPLACEMENT_DOOR", "CAULKS_SEALANTS",
        "WEATHERSTRIPPING", "VAPOR_RETARDER", "ATTIC_VENTILATION", "CLOCK_THERMOSTAT",
        "HEAT_EXCHANGER", "BOILER_CONTROLS", "WATER_HEATER_MOD", "DESUPERHEATER", "BOILER_REPAIR",
        "BOILER_EFFICIENCY", "HEATING_TUNEUP", "VENT_DAMPER", "IGNITION_CONVERSION", "DUCTWORK",
        "FILTER_ALARM", "FURNACE_REPLACEMENT", "AIR_CONDITIONING", "SHADING", "REFRIGERATOR",
        "FURNACE", "BOILER", "THERMOSTAT", "SMART_THERMOSTAT", "ECM_FURNACE_MOTOR",
        "SMOKE_DETECTOR", "CARBON_MONOXIDE_DETECTOR", "MECHANICAL_VENTILATION", "RANGE_HOOD",
        "COMBUSTION_AIR_DUCT", "MINOR_ROOF_REPAIR", "INCIDENTAL_REPAIR", "BLOWER_DOOR_DIAGNOSTICS",
        "FAN_REPLACEMENT"
    }
    
    # Smart Saver rebates by replacement type
    SMART_SAVER_REBATES = {
        "HEAT_PUMP_15.2_SEER2": {"early": 700, "failure": 350},
        "HEAT_PUMP_16_SEER2": {"early": 800, "failure": 450},
        "HEAT_PUMP_17_SEER2": {"early": 900, "failure": 700},
        "CENTRAL_AC_15.2_SEER2": {"early": 300, "failure": 200},
        "CENTRAL_AC_16_SEER2": {"early": 400, "failure": 300},
        "CENTRAL_AC_17_SEER2": {"early": 500, "failure": 400},
        "MINI_SPLIT": {"early": 900, "failure": 700},
        "GEOTHERMAL": {"early": 900, "failure": 700},
        "SMART_THERMOSTAT": {"early": 125, "failure": 125},
        "DUCT_SEALING": {"early": 350, "failure": 350},
        "INSULATION_AIR_SEAL_BUNDLE": {"early": 700, "failure": 700},
        "HEAT_PUMP_WATER_HEATER": {"early": 500, "failure": 500},
        "POOL_PUMP": {"early": 900, "failure": 900},
        "CONVERT_TO_HEAT_PUMP": {"early": 1500, "failure": 1500},
        "CONVERT_TO_GEOTHERMAL": {"early": 2500, "failure": 2300},
        "CONVERT_TO_DUAL_FUEL": {"early": 2500, "failure": 2000},
        "CONVERT_TO_COLD_CLIMATE": {"early": 2400, "failure": 2100}
    }
    
    FEDERAL_PROGRAMS = ["HOMES", "HEAR", "WAP"]
    UTILITY_PROGRAMS = ["SMART_SAVER"]
    ALL_PROGRAMS = FEDERAL_PROGRAMS + UTILITY_PROGRAMS

class OptimizedRebateCalculator:
    """
    Optimized rebate calculator using Integer Linear Programming (ILP)
    
    This class uses mathematical optimization to find the combination of rebates
    that maximizes total household savings while respecting all program constraints.
    """
    
    def __init__(self, config: RebateConfig):
        self.config = config
        
    def calculate_optimal_rebates(self, household: Dict) -> Dict:
        """
        Calculate optimal rebate allocation using ILP optimization
        
        Args:
            household: Dictionary containing:
                - income_level: "Low" or "Moderate"
                - measures: List of measures with name, cost, savings, etc.
        
        Returns:
            Dictionary with optimization results and detailed breakdown
        """
        measures = household["measures"]
        income_level = household["income_level"]
        
        # Create the optimization problem
        prob = LpProblem("Rebate_Optimization", LpMaximize)
        
        # Decision variables: x[i,p] = rebate amount for measure i from program p
        x = {}
        for i, measure in enumerate(measures):
            for program in self.config.ALL_PROGRAMS:
                x[i, program] = LpVariable(
                    f"rebate_{i}_{program}", 
                    lowBound=0, 
                    cat='Continuous'
                )
        
        # Binary variables: y[i,p] = 1 if measure i uses program p, 0 otherwise
        y = {}
        for i, measure in enumerate(measures):
            for program in self.config.FEDERAL_PROGRAMS:
                y[i, program] = LpVariable(
                    f"uses_{i}_{program}", 
                    cat='Binary'
                )
        
        # OBJECTIVE: Maximize total rebates
        prob += lpSum([x[i, p] for i in range(len(measures)) for p in self.config.ALL_PROGRAMS])
        
        # CONSTRAINTS
        self._add_constraints(prob, x, y, measures, income_level)
        
        # Solve the optimization problem
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract and format results
        return self._extract_results(prob, x, y, measures, income_level)
    
    def _add_constraints(self, prob, x, y, measures, income_level):
        """Add all constraints to the optimization problem"""
        
        # 1. PROGRAM CAPACITY CONSTRAINTS
        homes_cap = self.config.HOMES_CAP[income_level]
        prob += lpSum([x[i, "HOMES"] for i in range(len(measures))]) <= homes_cap
        prob += lpSum([x[i, "HEAR"] for i in range(len(measures))]) <= self.config.HEAR_CAP
        prob += lpSum([x[i, "WAP"] for i in range(len(measures))]) <= self.config.WAP_CAP
        
        # 2. MEASURE-SPECIFIC CONSTRAINTS
        for i, measure in enumerate(measures):
            measure_name = measure["name"]
            measure_cost = measure["cost"]
            
            # Total rebate cannot exceed measure cost
            prob += lpSum([x[i, p] for p in self.config.ALL_PROGRAMS]) <= measure_cost
            
            # Each measure can only use ONE federal program
            prob += lpSum([y[i, p] for p in self.config.FEDERAL_PROGRAMS]) <= 1
            
            # Link binary variables to rebate amounts
            for program in self.config.FEDERAL_PROGRAMS:
                # If y[i,p] = 0, then x[i,p] = 0
                prob += x[i, program] <= measure_cost * y[i, program]
                
                # If measure is not eligible, force both to 0
                if not self._is_eligible(measure_name, program):
                    prob += x[i, program] == 0
                    prob += y[i, program] == 0
            
            # HEAR item-specific caps
            if measure_name in self.config.HEAR_ITEM_CAPS:
                prob += x[i, "HEAR"] <= self.config.HEAR_ITEM_CAPS[measure_name]
            
            # Smart Saver constraints
            if measure_name in self.config.SMART_SAVER_REBATES:
                replacement_type = measure.get("replacement_type", "early")
                smart_saver_cap = self.config.SMART_SAVER_REBATES[measure_name].get(replacement_type, 0)
                prob += x[i, "SMART_SAVER"] <= smart_saver_cap
            else:
                prob += x[i, "SMART_SAVER"] == 0
    
    def _is_eligible(self, measure_name: str, program: str) -> bool:
        """Check if a measure is eligible for a specific program"""
        if program == "HOMES":
            return measure_name in self.config.HOMES_ELIGIBLE
        elif program == "HEAR":
            return measure_name in self.config.HEAR_ELIGIBLE
        elif program == "WAP":
            return measure_name in self.config.WAP_ELIGIBLE
        return False
    
    def _extract_results(self, prob, x, y, measures, income_level) -> Dict:
        """Extract and format optimization results"""
        
        if prob.status != LpStatusOptimal:
            return {
                "status": "infeasible",
                "message": "No feasible solution found",
                "total_rebate": 0,
                "program_totals": {},
                "measure_details": []
            }
        
        # Extract rebate amounts
        rebate_matrix = {}
        for i, measure in enumerate(measures):
            rebate_matrix[i] = {}
            for program in self.config.ALL_PROGRAMS:
                rebate_matrix[i][program] = value(x[i, program]) if x[i, program].varValue else 0
        
        # Calculate program totals
        program_totals = {}
        for program in self.config.ALL_PROGRAMS:
            program_totals[program] = sum(rebate_matrix[i][program] for i in range(len(measures)))
        
        # Create detailed measure breakdown
        measure_details = []
        for i, measure in enumerate(measures):
            measure_rebates = rebate_matrix[i]
            total_measure_rebate = sum(measure_rebates.values())
            
            # Find primary federal program
            primary_federal = None
            for program in self.config.FEDERAL_PROGRAMS:
                if measure_rebates[program] > 0:
                    primary_federal = program
                    break
            
            measure_details.append({
                "name": measure["name"],
                "cost": measure["cost"],
                "rebates": measure_rebates,
                "total_rebate": total_measure_rebate,
                "net_cost": measure["cost"] - total_measure_rebate,
                "primary_federal": primary_federal,
                "has_smart_saver": measure_rebates["SMART_SAVER"] > 0,
                "rebate_percentage": (total_measure_rebate / measure["cost"]) * 100 if measure["cost"] > 0 else 0
            })
        
        return {
            "status": "optimal",
            "total_rebate": sum(program_totals.values()),
            "program_totals": program_totals,
            "measure_details": measure_details,
            "optimization_value": value(prob.objective),
            "income_level": income_level,
            "num_measures": len(measures),
            "capacity_utilization": self._calculate_capacity_utilization(program_totals, income_level)
        }
    
    def _calculate_capacity_utilization(self, program_totals, income_level) -> Dict:
        """Calculate how much of each program's capacity is being used"""
        homes_cap = self.config.HOMES_CAP[income_level]
        
        utilization = {
            "HOMES": (program_totals["HOMES"] / homes_cap) * 100 if homes_cap > 0 else 0,
            "HEAR": (program_totals["HEAR"] / self.config.HEAR_CAP) * 100 if self.config.HEAR_CAP > 0 else 0,
            "WAP": (program_totals["WAP"] / self.config.WAP_CAP) * 100 if self.config.WAP_CAP > 0 else 0
        }
        
        return utilization

class RebateComparator:
    """Compare optimization results with greedy algorithm"""
    
    def __init__(self, config: RebateConfig):
        self.config = config
    
    def compare_approaches(self, household: Dict) -> Dict:
        """Compare optimized vs greedy approaches"""
        
        # Calculate optimized solution
        optimizer = OptimizedRebateCalculator(self.config)
        optimized_result = optimizer.calculate_optimal_rebates(household)
        
        # Calculate greedy solution (simplified version of original)
        greedy_result = self._calculate_greedy_rebates(household)
        
        # Compare results
        comparison = {
            "optimized": optimized_result,
            "greedy": greedy_result,
            "improvement": {
                "absolute": optimized_result["total_rebate"] - greedy_result["total_rebate"],
                "percentage": ((optimized_result["total_rebate"] - greedy_result["total_rebate"]) / 
                              greedy_result["total_rebate"] * 100) if greedy_result["total_rebate"] > 0 else 0
            }
        }
        
        return comparison
    
    def _calculate_greedy_rebates(self, household: Dict) -> Dict:
        """Simplified greedy algorithm for comparison"""
        # This is a simplified version - you could use your original algorithm here
        measures = household["measures"]
        income_level = household["income_level"]
        
        # Initialize caps
        remaining_caps = {
            "HOMES": self.config.HOMES_CAP[income_level],
            "HEAR": self.config.HEAR_CAP,
            "WAP": self.config.WAP_CAP
        }
        
        hear_item_caps = dict(self.config.HEAR_ITEM_CAPS)
        total_rebate = 0
        program_totals = {p: 0 for p in self.config.ALL_PROGRAMS}
        
        for measure in measures:
            measure_name = measure["name"]
            measure_cost = measure["cost"]
            
            # Find best federal program
            best_federal_rebate = 0
            best_federal_program = None
            
            for program in self.config.FEDERAL_PROGRAMS:
                if not self._is_eligible(measure_name, program):
                    continue
                    
                if program == "HOMES" and remaining_caps["HOMES"] > 0:
                    rebate = min(measure_cost, remaining_caps["HOMES"])
                elif program == "HEAR" and remaining_caps["HEAR"] > 0:
                    item_cap = hear_item_caps.get(measure_name, remaining_caps["HEAR"])
                    rebate = min(measure_cost, item_cap, remaining_caps["HEAR"])
                elif program == "WAP" and remaining_caps["WAP"] > 0:
                    rebate = min(measure_cost, remaining_caps["WAP"])
                else:
                    continue
                
                if rebate > best_federal_rebate:
                    best_federal_rebate = rebate
                    best_federal_program = program
            
            # Assign federal rebate
            if best_federal_program:
                remaining_caps[best_federal_program] -= best_federal_rebate
                if best_federal_program == "HEAR" and measure_name in hear_item_caps:
                    hear_item_caps[measure_name] -= best_federal_rebate
                program_totals[best_federal_program] += best_federal_rebate
                total_rebate += best_federal_rebate
            
            # Add Smart Saver if applicable
            if measure_name in self.config.SMART_SAVER_REBATES:
                replacement_type = measure.get("replacement_type", "early")
                smart_saver_cap = self.config.SMART_SAVER_REBATES[measure_name].get(replacement_type, 0)
                smart_saver_rebate = min(smart_saver_cap, measure_cost - best_federal_rebate)
                if smart_saver_rebate > 0:
                    program_totals["SMART_SAVER"] += smart_saver_rebate
                    total_rebate += smart_saver_rebate
        
        return {
            "status": "complete",
            "total_rebate": total_rebate,
            "program_totals": program_totals
        }
    
    def _is_eligible(self, measure_name: str, program: str) -> bool:
        """Check eligibility (same as optimizer)"""
        if program == "HOMES":
            return measure_name in self.config.HOMES_ELIGIBLE
        elif program == "HEAR":
            return measure_name in self.config.HEAR_ELIGIBLE
        elif program == "WAP":
            return measure_name in self.config.WAP_ELIGIBLE
        return False

# --- Utility Functions ---
def pretty_name(name: str) -> str:
    """Convert measure name to pretty format"""
    return name.replace("_", " ").title()

def make_measure(name: str, cost: int, savings: int, replacement_type: str = "early", **kwargs) -> Dict:
    """Create a measure dictionary"""
    measure = {
        "name": name,
        "cost": cost,
        "savings": savings,
        "replacement_type": replacement_type
    }
    measure.update(kwargs)
    return measure

# --- Preset Measures ---
PRESET_MEASURES = [
    make_measure("INSULATION", 9000, 12, "early"),
    make_measure("AIR_SEALING", 4000, 9, "early"),
    make_measure("HEAT_PUMP_DRYER", 800, 2, "early"),
    make_measure("PIPE_INSULATION", 600, 1, "early"),
    make_measure("SMART_THERMOSTAT", 300, 0, "early"),
    make_measure("HEAT_PUMP_16_SEER2", 8500, 14, "early"),
]

# --- Streamlit Interface ---
def main():
    st.set_page_config(page_title="Optimized Rebate Calculator", layout="wide")
    
    st.title("Optimized Rebate Stacking Calculator")
    st.caption("Everblue's prototype rebate stacker across HOMES, HEAR, WAP, and Smart Saver")
    
    # Initialize configuration
    config = RebateConfig()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        show_comparison = st.checkbox("Show Optimization vs Greedy Comparison", value=True)
        show_details = st.checkbox("Show Detailed Breakdown", value=True)
        export_results = st.checkbox("Enable Results Export", value=False)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Household Information")
        
        income = st.selectbox("Income Level:", ["Low", "Moderate"], index=0)
        
        # Get all possible measures
        all_measures = sorted(
            config.HOMES_ELIGIBLE | config.HEAR_ELIGIBLE | 
            config.WAP_ELIGIBLE | set(config.SMART_SAVER_REBATES.keys())
        )
        
        # Number of measures
        num_measures = st.number_input("Number of Measures:", min_value=1, max_value=20, value=len(PRESET_MEASURES))
        
        # Measure inputs
        measures = []
        incomplete = False
        
        for i in range(num_measures):
            with st.expander(f"Measure {i+1}", expanded=i < 3):
                # Use preset if available
                if i < len(PRESET_MEASURES):
                    preset = PRESET_MEASURES[i]
                    default_name = preset["name"]
                    default_cost = preset["cost"]
                    default_savings = preset["savings"]
                    default_rtype = preset.get("replacement_type", "early")
                else:
                    default_name = ""
                    default_cost = 1000
                    default_savings = 5
                    default_rtype = "early"
                
                # Measure selection
                pretty_names = ["Select Option"] + [pretty_name(m) for m in all_measures]
                selected_pretty = st.selectbox(
                    "Measure Type:",
                    pretty_names,
                    key=f"measure_{i}",
                    index=(all_measures.index(default_name) + 1) if default_name else 0
                )
                
                if selected_pretty == "Select Option":
                    incomplete = True
                    continue
                
                measure_name = selected_pretty.upper().replace(" ", "_")
                
                # Cost and savings
                col_a, col_b = st.columns(2)
                with col_a:
                    cost = st.number_input("Cost ($):", min_value=0, value=default_cost, key=f"cost_{i}")
                with col_b:
                    savings = st.number_input("Energy Savings (%):", min_value=0, max_value=100, value=default_savings, key=f"savings_{i}")
                
                # Replacement type for Smart Saver
                replacement_type = "early"
                if measure_name in config.SMART_SAVER_REBATES:
                    replacement_display = st.selectbox(
                        "Replacement Type:",
                        ["Early Installation", "After Failure"],
                        key=f"rtype_{i}",
                        index=0 if default_rtype == "early" else 1
                    )
                    replacement_type = "failure" if replacement_display == "After Failure" else "early"
                
                measure = make_measure(measure_name, cost, savings, replacement_type)
                measures.append(measure)
    
    with col2:
        st.header("Program Capacities")
        
        # Show program caps
        st.metric("HOMES Cap", f"${config.HOMES_CAP[income]:,}")
        st.metric("HEAR Cap", f"${config.HEAR_CAP:,}")
        st.metric("WAP Cap", f"${config.WAP_CAP:,}")
        
        st.divider()
        
        # Quick stats
        if not incomplete and measures:
            total_cost = sum(m["cost"] for m in measures)
            avg_savings = np.mean([m["savings"] for m in measures])
            st.metric("Total Project Cost", f"${total_cost:,}")
            st.metric("Avg Energy Savings", f"{avg_savings:.1f}%")
    
    # Calculate rebates
    st.divider()
    
    if st.button("Calculate Optimal Rebates", type="primary"):
        if incomplete or len(measures) < num_measures:
            st.error("Please complete all measure selections.")
        else:
            household = {"income_level": income, "measures": measures}
            
            # Show loading
            with st.spinner("Optimizing rebate allocation..."):
                if show_comparison:
                    comparator = RebateComparator(config)
                    results = comparator.compare_approaches(household)
                    
                    # Display comparison
                    st.header("Optimization Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Optimized Total", f"${results['optimized']['total_rebate']:,.0f}")
                    with col2:
                        st.metric("Greedy Total", f"${results['greedy']['total_rebate']:,.0f}")
                    with col3:
                        improvement = results['improvement']['absolute']
                        st.metric("Improvement", f"${improvement:,.0f}", f"{results['improvement']['percentage']:.1f}%")
                    
                    # Detailed results
                    if show_details:
                        display_detailed_results(results['optimized'], config)
                
                else:
                    # Just show optimized results
                    optimizer = OptimizedRebateCalculator(config)
                    results = optimizer.calculate_optimal_rebates(household)
                    
                    st.header("Optimization Results")
                    st.metric("Total Rebates", f"${results['total_rebate']:,.0f}")
                    
                    if show_details:
                        display_detailed_results(results, config)

def display_detailed_results(results: Dict, config: RebateConfig):
    """Display detailed optimization results"""
    
    # Program breakdown
    st.subheader("Program Breakdown")
    
    program_names = {
        "HOMES": "HOMES Program",
        "HEAR": "HEAR Program", 
        "WAP": "WAP Program",
        "SMART_SAVER": "Smart Saver Rebates"
    }
    
    cols = st.columns(len(config.ALL_PROGRAMS))
    for i, program in enumerate(config.ALL_PROGRAMS):
        with cols[i]:
            amount = results["program_totals"][program]
            st.metric(program_names[program], f"${amount:,.0f}")
    
    # Capacity utilization
    if "capacity_utilization" in results:
        st.subheader("Program Capacity Utilization")
        util = results["capacity_utilization"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("HOMES", f"{util['HOMES']:.1f}%")
        with col2:
            st.metric("HEAR", f"{util['HEAR']:.1f}%")
        with col3:
            st.metric("WAP", f"{util['WAP']:.1f}%")
    
    # Measure details
    st.subheader("Measure-by-Measure Breakdown")
    
    measure_data = []
    for detail in results["measure_details"]:
        row = {
            "Measure": pretty_name(detail["name"]),
            "Cost": f"${detail['cost']:,}",
            "HOMES": f"${detail['rebates']['HOMES']:,.0f}" if detail['rebates']['HOMES'] > 0 else "-",
            "HEAR": f"${detail['rebates']['HEAR']:,.0f}" if detail['rebates']['HEAR'] > 0 else "-",
            "WAP": f"${detail['rebates']['WAP']:,.0f}" if detail['rebates']['WAP'] > 0 else "-",
            "Smart Saver": f"${detail['rebates']['SMART_SAVER']:,.0f}" if detail['rebates']['SMART_SAVER'] > 0 else "-",
            "Total Rebate": f"${detail['total_rebate']:,.0f}",
            "Net Cost": f"${detail['net_cost']:,.0f}",
            "Rebate %": f"{detail['rebate_percentage']:.1f}%"
        }
        measure_data.append(row)
    
    df = pd.DataFrame(measure_data)
    st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()