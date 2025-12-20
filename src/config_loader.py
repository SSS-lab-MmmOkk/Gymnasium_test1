import json
from typing import Dict, Any

class ConfigLoader:
    """
    Handles loading and validating the scenario configuration file.
    """
    def __init__(self, filepath: str):
        """
        Initializes the loader, loads the data, and validates it.

        Args:
            filepath: Path to the JSON configuration file.
        """
        self.filepath = filepath
        self.data = self._load_config()
        self._validate()

    def _load_config(self) -> Dict[str, Any]:
        """
        Loads the JSON configuration file from the specified path.

        Returns:
            A dictionary containing the configuration data.

        Raises:
            FileNotFoundError: If the config file does not exist.
            json.JSONDecodeError: If the file is not a valid JSON.
        """
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                # To be compliant with pure JSON, we don't use a parser that allows comments.
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {self.filepath}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error decoding JSON from the file: {self.filepath}. Details: {e.msg}", e.doc, e.pos)

    def _validate(self):
        """
        Runs all validation checks on the loaded configuration data.
        """
        print("Validating configuration...")
        self._validate_cause_node_mapping()
        # Add other validation calls here if needed in the future.
        print("Configuration validation successful.")

    def _validate_cause_node_mapping(self):
        """
        Validates that all nodes of type 'cause' have a corresponding entry
        in the 'simulation_mapping' section, as per requirement 8.1.

        Raises:
            ValueError: If a cause node is found without a simulation mapping.
        """
        bayesian_model = self.get_bayesian_model()
        if not bayesian_model:
            raise ValueError("Validation Error: 'bayesian_models' section is missing or empty.")

        variables = bayesian_model.get('variables', [])
        simulation_mapping = self.get_simulation_mapping()

        cause_node_ids = {var['id'] for var in variables if var.get('type') == 'cause'}
        mapped_node_ids = set(simulation_mapping.keys())

        undefined_causes = cause_node_ids - mapped_node_ids

        if undefined_causes:
            error_msg = (
                "Validation Error: The following cause nodes are defined in 'variables' "
                f"but are not defined in 'simulation_mapping': {', '.join(sorted(undefined_causes))}"
            )
            raise ValueError(error_msg)

    def get_bayesian_model(self) -> Dict[str, Any]:
        """
        Retrieves the first Bayesian model definition from the configuration.

        Returns:
            A dictionary defining the Bayesian model, or an empty dict if not found.
        """
        # Assuming we are working with the first model in the list for v1.0
        return self.data.get('bayesian_models', [{}])[0]

    def get_simulation_mapping(self) -> Dict[str, Any]:
        """
        Retrieves the simulation mapping rules from the configuration.

        Returns:
            A dictionary containing the simulation mapping rules.
        """
        return self.data.get('simulation_mapping', {})

    def get_learning_config(self) -> Dict[str, Any]:
        """
        Retrieves the learning process configuration.

        Returns:
            A dictionary containing the learning settings.
        """
        return self.data.get('learning', {})

    def get_cpt_matching_rules(self) -> Dict[str, Any]:
        """
        Retrieves the CPT matching rule configuration.

        Returns:
            A dictionary containing the CPT matching settings.
        """
        return self.data.get('cpt_matching', {})

# --- Example Usage ---
# This block allows for direct execution and testing of the ConfigLoader.
if __name__ == '__main__':
    try:
        # Relative path from the root of the project
        config_path = 'config/scenario.json'
        print(f"Attempting to load configuration from: {config_path}")

        # Instantiate the loader, which automatically loads and validates
        loader = ConfigLoader(config_path)

        print("\n--- Getter Method Tests ---")
        model = loader.get_bayesian_model()
        print(f"Successfully retrieved model: '{model.get('model_id')}'")

        mapping = loader.get_simulation_mapping()
        print(f"Successfully retrieved {len(mapping)} simulation mappings.")

        learning_config = loader.get_learning_config()
        print(f"Learning method identified as: '{learning_config.get('method')}'")

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"\n--- An error occurred ---")
        print(e)
