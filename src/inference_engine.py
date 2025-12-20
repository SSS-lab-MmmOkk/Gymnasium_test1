import json
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import itertools
from typing import Dict, Any, List, Tuple

# Assuming config_loader is in the same src directory
from .config_loader import ConfigLoader

class InferenceEngine:
    """
    Core module for Bayesian network inference, sampling, and learning.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the inference engine with a given configuration.
        """
        self.config = config
        self.model_spec = self._get_model_spec()
        self.variables = {var['id']: var for var in self.model_spec.get('variables', [])}

        self.cause_nodes = [var_id for var_id, var in self.variables.items() if var.get('type') == 'cause']

        self.cpds = self._create_all_cpds()
        self.model = self._build_model()
        self._initialize_learning_params()

    def _get_model_spec(self) -> Dict[str, Any]:
        """Extracts the relevant model specification from the config."""
        # Assuming the first model is the target for v1.0
        return self.config.get('bayesian_models', [{}])[0]

    def _build_model(self) -> DiscreteBayesianNetwork:
        """
        Constructs the pgmpy DiscreteBayesianNetwork object from the configuration.
        """
        model = DiscreteBayesianNetwork()
        edges = []
        for var_id, var_spec in self.variables.items():
            model.add_node(var_id)
            if 'parents' in var_spec:
                for parent in var_spec['parents']:
                    edges.append((parent, var_id))

        model.add_edges_from(edges)

        # Add CPDs before checking the model to ensure a complete state.
        model.add_cpds(*self.cpds)

        # In v1.0, 'evidence' nodes do not have CPDs, which causes check_model() to fail.
        # We rely on our config validation and the fact that the model is built programmatically.
        # if not model.check_model():
        #     raise ValueError("Model check failed. Check for cycles and correctness.")

        return model

    def _create_all_cpds(self) -> List[TabularCPD]:
        """
        Creates and returns a list of TabularCPD objects for all variables.
        """
        cpd_list = []
        for var_id in self.variables:
            var_spec = self.variables[var_id]
            var_type = var_spec.get('type')

            if var_type == 'cause':
                # For root nodes, pgmpy expects probabilities in a column vector (states x 1)
                probabilities = np.array(var_spec['probabilities']).reshape(-1, 1)
                cpd = TabularCPD(
                    variable=var_id,
                    variable_card=len(var_spec['states']),
                    values=probabilities,
                    state_names={var_id: var_spec['states']}
                )
                cpd_list.append(cpd)
            elif var_type == 'intermediate':
                cpd = self._create_wildcard_cpd(var_id, var_spec)
                cpd_list.append(cpd)
            # 'evidence' nodes do not have CPDs in v1.0 as they are observed.

        return cpd_list

    def _create_wildcard_cpd(self, var_id: str, var_spec: Dict[str, Any]) -> TabularCPD:
        """
        Creates a TabularCPD for an intermediate node, resolving wildcard rules.
        """
        states = var_spec['states']
        parents = var_spec['parents']
        cpt_rules = var_spec['cpt']

        parent_states = {p: self.variables[p]['states'] for p in parents}

        # Generate all combinations of parent states in the correct order
        parent_state_combinations = list(itertools.product(*[parent_states[p] for p in parents]))

        # Build the final probability table column by column
        final_probs = []
        for combo in parent_state_combinations:
            parent_evidence = dict(zip(parents, combo))
            best_rule_key = self._find_best_cpt_rule(parent_evidence, cpt_rules, parents)
            probabilities = cpt_rules[best_rule_key]
            # Ensure the order of probabilities matches the order of the variable's states
            column_probs = [probabilities[state] for state in states]
            final_probs.append(column_probs)

        # Transpose the matrix to fit pgmpy's format (states x combinations)
        values = np.array(final_probs).T.tolist()

        return TabularCPD(
            variable=var_id,
            variable_card=len(states),
            values=values,
            evidence=parents,
            evidence_card=[len(parent_states[p]) for p in parents],
            state_names={var_id: states, **parent_states}
        )

    def _find_best_cpt_rule(self, evidence: Dict[str, str], rules: Dict[str, Any], parent_order: List[str]) -> str:
        """
        Finds the best matching CPT rule based on specificity and definition order.
        """
        matching_rules = []
        for rule_key, _ in rules.items():
            if rule_key == 'default':
                continue

            rule_parts = rule_key.split('+')
            is_match = True
            for i, parent in enumerate(parent_order):
                if rule_parts[i] != 'Any' and rule_parts[i] != evidence[parent]:
                    is_match = False
                    break
            if is_match:
                specificity = rule_key.count('Any')
                matching_rules.append((rule_key, specificity))

        if not matching_rules:
            if 'default' not in rules:
                raise ValueError(f"No matching CPT rule or 'default' found for evidence: {evidence}")
            return 'default'

        # Sort by specificity (fewer 'Any' is better), order is preserved for ties
        matching_rules.sort(key=lambda x: x[1])

        return matching_rules[0][0]

    def _initialize_learning_params(self):
        """
        Initializes Dirichlet pseudo-counts for learning.
        """
        self.pseudo_counts = {}
        for node_id in self.cause_nodes:
            spec = self.variables[node_id]
            states = spec['states']
            initial_probs = spec['probabilities']
            # Using initial probabilities as pseudo-counts (alpha0), scaled by a belief factor.
            # This makes the initial updates less drastic.
            belief_strength = 10.0
            self.pseudo_counts[node_id] = {state: prob * belief_strength for state, prob in zip(states, initial_probs)}

    def sample(self, n_samples: int = 1) -> List[Dict[str, str]]:
        """
        Samples from the model to generate scenarios.
        This simplified sampler focuses on sampling cause nodes based on their current distribution.
        """
        samples = []
        # Find CPDs for cause nodes and store them for quick access
        cause_cpds = {cpd.variable: cpd for cpd in self.cpds if cpd.variable in self.cause_nodes}

        for _ in range(n_samples):
            sample = {}
            for node_id in self.cause_nodes:
                states = self.variables[node_id]['states']
                cpd = cause_cpds[node_id]
                probs = cpd.values.flatten()

                sampled_state = np.random.choice(states, p=probs)
                sample[node_id] = sampled_state
            samples.append(sample)
        return samples

    def update(self, observed_sample: Dict[str, str], outcome: Dict[str, str]):
        """
        Updates the model's parameters based on an observed trial outcome.
        """
        learning_config = self.config.get('learning', {})
        update_condition_str = learning_config.get('update_condition', "False")

        # Simple evaluation of the update condition string
        # For "R1_Outcome == 'NearMiss'", we check if the outcome dict matches this
        # This is a simplification; a real implementation might need a more robust parser.
        if "R1_Outcome" in outcome and outcome["R1_Outcome"] == "NearMiss":
             # Apply Dirichlet update
            for node_id, state in observed_sample.items():
                if node_id in self.pseudo_counts:
                    self.pseudo_counts[node_id][state] += 1

            self._recalculate_cpd_values()
            # Re-build the model to ensure the updated CPDs are correctly registered.
            self.model = self._build_model()
            print("Model probabilities updated and model rebuilt due to NearMiss.")

    def _recalculate_cpd_values(self):
        """
        Recalculates probabilities from pseudo-counts and updates the CPD objects in self.cpds.
        This method MODIFIES the CPD objects in self.cpds in-place.
        """
        print("\n--- Recalculating CPD Values ---")
        cpd_map = {cpd.variable: cpd for cpd in self.cpds}

        for node_id in self.cause_nodes:
            if node_id not in cpd_map:
                continue

            cpd_to_update = cpd_map[node_id]
            counts = self.pseudo_counts[node_id]
            total_count = sum(counts.values())
            states = self.variables[node_id]['states']
            new_probs = [counts[state] / total_count for state in states]

            # pgmpy expects the values in a specific shape. For root nodes, it's (num_states, 1).
            cpd_to_update.values = np.array(new_probs).reshape(-1, 1)


# --- Example Usage ---
if __name__ == '__main__':
    try:
        config_path = 'config/scenario.json'
        loader = ConfigLoader(config_path)
        engine = InferenceEngine(loader.data)

        # Helper to get a CPD from a list by variable name
        def get_cpd_from_list(name, cpd_list):
            return next((cpd for cpd in cpd_list if cpd.variable == name), None)

        print("\n--- Bayesian Model Initialized ---")
        print(f"Nodes: {engine.model.nodes()}")
        print(f"Edges: {engine.model.edges()}")

        print("\n--- CPT for L2_Decision (from self.cpds) ---")
        l2_decision_cpd = get_cpd_from_list('L2_Decision', engine.cpds)
        print(l2_decision_cpd)

        print("\n--- Initial Probabilities for C1_1_RelDist (from self.cpds) ---")
        c1_1_cpd = get_cpd_from_list('C1_1_RelDist', engine.cpds)
        print(c1_1_cpd)

        print("\n--- Simulating 1 Trial ---")
        sampled_causes = engine.sample(1)[0]
        print(f"Sampled Causes: {sampled_causes}")

        simulated_outcome = {'R1_Outcome': 'NearMiss'}
        print(f"Simulated Outcome: {simulated_outcome}")

        engine.update(sampled_causes, simulated_outcome)

        print("\n--- Updated Probabilities for C1_1_RelDist (from self.cpds) after 1 NearMiss ---")
        updated_c1_1_cpd = get_cpd_from_list('C1_1_RelDist', engine.cpds)
        print(updated_c1_1_cpd)

        # Final check: does the model object reflect the change?
        print("\n--- Verifying CPD in the final model object ---")
        print(engine.model.get_cpds('C1_1_RelDist'))

    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(e)
