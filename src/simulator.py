import numpy as np
from typing import Dict, Any

class SimpleSimulator:
    """
    A simplified 2D simulator for the pedestrian crossing scenario (PoC).
    Calculates key metrics like min_distance and PET based on physical parameters.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the simulator with continuous physical parameters.

        Args:
            params: A dictionary of parameters, converted by the Mapper.
                    Example: {'v_init': 8.33, 'd_entry': 20.0, 'v_ped': 1.25, ...}
        """
        self.params = params
        self.g = 9.81  # Acceleration due to gravity in m/s^2

    def run(self) -> Dict[str, Any]:
        """
        Runs the simulation and computes the outcome metrics.

        Returns:
            A dictionary containing the simulation results, e.g.,
            {'min_distance': 1.2, 'pet': 0.8, 'action': 'Stop'}
        """
        # --- Parameter Extraction and Conversion ---
        v_init_mps = self.params['C2_1_InitSpeed'] / 3.6  # km/h to m/s
        d_entry_m = self.params['C1_1_RelDist']
        v_ped_mps = self.params['C1_2_WalkSpeed'] / 3.6 # km/h to m/s
        max_decel_g = self.params['C2_4_MaxDecel']
        sensor_delay_s = self.params['C2_2_SensorDelay']
        plan_delay_s = self.params['C2_3_PlanDelay']

        # --- Simplified Physics Calculations ---

        # 1. Calculate stopping distance of the vehicle
        reaction_time_s = sensor_delay_s + plan_delay_s
        reaction_distance_m = v_init_mps * reaction_time_s
        braking_distance_m = v_init_mps**2 / (2 * max_decel_g * self.g)
        stopping_distance_m = reaction_distance_m + braking_distance_m

        # 2. Determine vehicle's action (Stop or Pass) - A simplified logic for PoC
        # This logic should ideally be part of the AV's decision model (L2_Decision),
        # but for the simulator, we make a deterministic choice based on physics.
        if stopping_distance_m < d_entry_m:
            action = 'Stop'
            final_distance_to_ped = d_entry_m - stopping_distance_m
        else:
            action = 'Pass'
            # If it cannot stop in time, what would the min_distance be?
            # This is a highly simplified estimation.
            time_to_reach_crossing = d_entry_m / v_init_mps
            final_distance_to_ped = - (stopping_distance_m - d_entry_m) * 0.2 # Heuristic

        min_distance = final_distance_to_ped

        # 3. Calculate a simplified PET (Post Encroachment Time)
        # Time for vehicle to clear the crossing vs. time for pedestrian to arrive.
        # This is a conceptual stand-in for a real PET calculation.
        time_for_vehicle_to_clear = d_entry_m / v_init_mps if v_init_mps > 0 else float('inf')

        # Assume pedestrian is at the edge of a 4m wide road
        time_for_ped_to_arrive = (4.0 / v_ped_mps) if v_ped_mps > 0 else float('inf')

        pet = time_for_ped_to_arrive - time_for_vehicle_to_clear

        # Ensure PET is non-negative and handle edge cases
        if action == 'Stop':
            pet = 5.0 # Assign a large, safe PET value if the car stops
        else:
            pet = max(0, pet)

        return {
            "min_distance": round(min_distance, 2),
            "pet": round(pet, 2),
            "action": action # This is the physically determined action
        }

# --- Example Usage ---
if __name__ == '__main__':
    # Define two test cases: one likely to be a "SafeStop", one a "NearMiss"

    # Case 1: Safe parameters
    safe_params = {
        'C2_1_InitSpeed': 30.0,  # km/h
        'C1_1_RelDist': 40.0,    # m
        'C1_2_WalkSpeed': 4.5,   # km/h
        'C2_4_MaxDecel': 0.8,    # g
        'C2_2_SensorDelay': 0.1, # s
        'C2_3_PlanDelay': 0.1    # s
    }

    # Case 2: Aggressive parameters
    aggressive_params = {
        'C2_1_InitSpeed': 45.0,  # km/h
        'C1_1_RelDist': 25.0,    # m
        'C1_2_WalkSpeed': 8.0,   # km/h
        'C2_4_MaxDecel': 0.4,    # g
        'C2_2_SensorDelay': 0.4, # s
        'C2_3_PlanDelay': 0.3    # s
    }

    print("--- Running Test Case 1 (Safe Scenario) ---")
    simulator_safe = SimpleSimulator(safe_params)
    results_safe = simulator_safe.run()
    print(f"Results: {results_safe}")
    # Expected: 'Stop' action, positive min_distance, high PET

    print("\n--- Running Test Case 2 (Aggressive Scenario) ---")
    simulator_aggressive = SimpleSimulator(aggressive_params)
    results_aggressive = simulator_aggressive.run()
    print(f"Results: {results_aggressive}")
    # Expected: 'Pass' action (can't stop in time), low or negative min_distance, low PET
