import numpy as np
from typing import Dict, Any

class Mapper:
    """
    Maps discrete, sampled states from the Inference Engine to continuous
    physical parameters for the Simulator, based on 'simulation_mapping'.
    """
    def __init__(self, mapping_rules: Dict[str, Any]):
        """
        Initializes the Mapper with mapping rules from the config.
        """
        self.mapping_rules = mapping_rules

    def transform(self, sampled_causes: Dict[str, str]) -> Dict[str, Any]:
        """
        Transforms a dictionary of discrete states into physical parameters.

        Args:
            sampled_causes: A dict like {'C1_1_RelDist': 'DangerZone', ...}.

        Returns:
            A dict of continuous params like {'C1_1_RelDist': 10.5, ...}.
        """
        physical_params = {}
        for node_id, state in sampled_causes.items():
            if node_id not in self.mapping_rules:
                raise ValueError(f"Mapper Error: No mapping rule found for cause node '{node_id}'.")

            rule = self.mapping_rules[node_id][state]

            if 'value' in rule:
                physical_params[node_id] = rule['value']
            elif 'dist' in rule:
                dist_type = rule['dist']
                if dist_type == 'normal':
                    val = np.random.normal(loc=rule['mean'], scale=rule['std'])
                elif dist_type == 'uniform':
                    val = np.random.uniform(low=rule['min'], high=rule['max'])
                elif dist_type == 'exponential':
                    val = np.random.exponential(scale=rule['scale'])
                elif dist_type == 'beta':
                    # Scaled beta distribution
                    val = np.random.beta(a=rule['alpha'], b=rule['beta'])
                    val = rule['scale_min'] + val * (rule['scale_max'] - rule['scale_min'])
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")
                physical_params[node_id] = val
            else:
                 raise ValueError(f"Invalid mapping rule for {node_id}[{state}]: must contain 'value' or 'dist'.")

        return physical_params

class Observer:
    """
    Observes the results from the Simulator and classifies them into discrete
    evidence states for the Inference Engine, based on 'simulation_mapping'.
    """
    def __init__(self, mapping_rules: Dict[str, Any]):
        """
        Initializes the Observer with mapping rules from the config.
        """
        # We only need the rules for evidence nodes
        self.outcome_rules = mapping_rules.get('R1_Outcome', {}).get('rules', [])
        self.pet_bin_rules = mapping_rules.get('E_PET_Bin', {}).get('rules', [])

    def classify(self, sim_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Classifies simulator output into discrete evidence states.

        Args:
            sim_results: A dict from the simulator, e.g., {'min_distance': 1.2, ...}.

        Returns:
            A dict of evidence, e.g., {'R1_Outcome': 'NearMiss', 'E_PET_Bin': 'LT_1s'}.
        """
        evidence = {}

        # Classify R1_Outcome
        outcome_state = self._evaluate_rules(self.outcome_rules, sim_results)
        if outcome_state:
            evidence['R1_Outcome'] = outcome_state
        else:
            raise ValueError(f"Observer Error: Could not classify R1_Outcome for results: {sim_results}")

        # Classify E_PET_Bin
        pet_bin_state = self._evaluate_rules(self.pet_bin_rules, sim_results)
        if pet_bin_state:
            evidence['E_PET_Bin'] = pet_bin_state
        else:
            raise ValueError(f"Observer Error: Could not classify E_PET_Bin for results: {sim_results}")

        return evidence

    def _evaluate_rules(self, rules: list, results: Dict[str, Any]) -> str:
        """
        Evaluates a list of classification rules against simulation results.

        Args:
            rules: A list of rule objects, e.g., [{'state': 'Collision', 'condition': 'min_distance <= 0.0'}].
            results: The simulation results dictionary.

        Returns:
            The name of the state for the first matching rule, or None.
        """
        # Create a context for eval() where all result keys are uppercased for consistency.
        results_upper = {k.upper(): v for k, v in results.items()}
        eval_context = {**results_upper, 'True': True, 'False': False, 'None': None}

        for rule in rules:
            try:
                # Replace common non-Python logical operators with Python ones.
                condition_str = rule['condition'].replace('&&', 'and').replace('||', 'or')

                # WARNING: eval() can be insecure. In a production system, use a safer expression parser.
                # For this PoC, it's a pragmatic way to implement the rule logic from the JSON.
                if eval(condition_str, {}, eval_context):
                    return rule['state']
            except Exception as e:
                print(f"Warning: Could not evaluate condition '{rule['condition']}'. Error: {e}")

        return None


# --- Example Usage ---
if __name__ == '__main__':
    from .config_loader import ConfigLoader

    # Load the full configuration to get mapping rules
    config = ConfigLoader('config/scenario.json').get_simulation_mapping()

    # --- Test Mapper ---
    print("--- Testing Mapper ---")
    mapper = Mapper(config)
    sample_discrete = {'C1_1_RelDist': 'DangerZone', 'C1_2_WalkSpeed': 'Fast'}
    physical = mapper.transform(sample_discrete)
    print(f"Mapped '{sample_discrete['C1_1_RelDist']}' to C1_1_RelDist: {physical['C1_1_RelDist']:.2f}")
    print(f"Mapped '{sample_discrete['C1_2_WalkSpeed']}' to C1_2_WalkSpeed: {physical['C1_2_WalkSpeed']:.2f}")

    # --- Test Observer ---
    print("\n--- Testing Observer ---")
    observer = Observer(config)

    # Test case 1: NearMiss and PET < 1s
    sim_result_1 = {'min_distance': 0.8, 'pet': 0.5, 'action': 'Pass'}
    evidence_1 = observer.classify(sim_result_1)
    print(f"Results {sim_result_1} classified as: {evidence_1}")
    assert evidence_1['R1_Outcome'] == 'NearMiss'
    assert evidence_1['E_PET_Bin'] == 'LT_1s'

    # Test case 2: SafeStop and PET >= 1s
    sim_result_2 = {'min_distance': 10.0, 'pet': 5.0, 'action': 'Stop'}
    evidence_2 = observer.classify(sim_result_2)
    print(f"Results {sim_result_2} classified as: {evidence_2}")
    assert evidence_2['R1_Outcome'] == 'SafeStop'
    assert evidence_2['E_PET_Bin'] == 'GE_1s'

    # Test case 3: Collision
    sim_result_3 = {'min_distance': -0.1, 'pet': 0.1, 'action': 'Pass'}
    evidence_3 = observer.classify(sim_result_3)
    print(f"Results {sim_result_3} classified as: {evidence_3}")
    assert evidence_3['R1_Outcome'] == 'Collision'
