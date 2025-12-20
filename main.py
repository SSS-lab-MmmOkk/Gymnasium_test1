from src.config_loader import ConfigLoader
from src.inference_engine import InferenceEngine
from src.execution_layer import Mapper, Observer
from src.simulator import SimpleSimulator
import pprint

def main():
    """
    Main function to run the probabilistic scenario generation and validation loop.
    """
    config_path = 'config/scenario.json'
    num_trials = 50

    print("--- Initializing Platform ---")

    # 1. Load Configuration
    loader = ConfigLoader(config_path)
    config_data = loader.data

    # 2. Initialize Core Components
    inference_engine = InferenceEngine(config_data)
    mapper = Mapper(config_data.get('simulation_mapping', {}))
    observer = Observer(config_data.get('simulation_mapping', {}))

    print(f"\n--- Starting Simulation Loop for {num_trials} Trials ---")

    # Store a history of outcomes for final summary
    outcome_counts = {}

    for i in range(num_trials):
        print(f"\n--- Trial {i+1}/{num_trials} ---")

        # --- The Core Loop: Sample -> Map -> Simulate -> Observe -> Update ---

        # 1. SAMPLE from the Bayesian network's current probability distributions
        sampled_causes = inference_engine.sample(1)[0]
        print(f"1. [Sampler] Sampled discrete causes:")
        pprint.pprint(sampled_causes)

        # 2. MAP discrete states to continuous physical parameters
        physical_params = mapper.transform(sampled_causes)
        print(f"\n2. [Mapper] Transformed to physical parameters:")
        pprint.pprint({k: f"{v:.2f}" for k, v in physical_params.items() if isinstance(v, float)})


        # 3. SIMULATE the scenario with the physical parameters
        simulator = SimpleSimulator(physical_params)
        sim_results = simulator.run()
        print(f"\n3. [Simulator] Simulation results:")
        pprint.pprint(sim_results)

        # 4. OBSERVE and classify the results into discrete evidence
        observed_evidence = observer.classify(sim_results)
        print(f"\n4. [Observer] Classified evidence:")
        pprint.pprint(observed_evidence)

        # Keep track of outcomes
        outcome = observed_evidence.get('R1_Outcome', 'Unknown')
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        # 5. UPDATE the Bayesian network's probabilities if the condition is met
        print("\n5. [Updater] Checking for learning condition...")
        # Get probabilities before the update to show the change
        rel_dist_cpd_before = inference_engine.model.get_cpds('C1_1_RelDist')

        inference_engine.update(sampled_causes, observed_evidence)

        # If a NearMiss occurred, show how the probabilities were affected
        if outcome == 'NearMiss':
            print("\n   !!! NearMiss Detected - Probabilities Updated !!!")
            rel_dist_cpd_after = inference_engine.model.get_cpds('C1_1_RelDist')
            print("   --- C1_1_RelDist Probabilities ---")
            print(f"   Before: {rel_dist_cpd_before.values.flatten()}")
            print(f"   After:  {rel_dist_cpd_after.values.flatten()}")
            print("   ----------------------------------")
        else:
            print("   Condition not met. No update performed.")

    print("\n\n--- Simulation Loop Finished ---")
    print("\n--- Summary of Outcomes ---")
    pprint.pprint(outcome_counts)


if __name__ == '__main__':
    main()
