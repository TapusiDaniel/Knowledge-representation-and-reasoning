from typing import List, Dict, Tuple, Any, Iterator
from bayes_net import BayesNet, BayesNode, JunctionTree, normalize
from itertools import product
import numpy as np
from argparse import ArgumentParser, Namespace
import pandas as pd
from tqdm import tqdm

def all_dicts(variables: List[str]) -> Iterator[Dict[str, int]]:
    for keys in product(*([[0, 1]] * len(variables))):
        yield dict(zip(variables, keys))


def cross_entropy(bn1: BayesNet, bn2: BayesNet, nsamples: int = None) -> float:
    cross_ent = .0
    if nsamples is None:
        bn1_vars = bn1.nodes.keys()
        for sample in all_dicts(bn1_vars):
            cross_ent -= np.exp(bn1.sample_log_prob(sample)) * bn2.sample_log_prob(sample)
    else:
        for _ in range(nsamples):
            cross_ent -= bn2.sample_log_prob(bn1.sample())
        cross_ent /= nsamples
    return cross_ent


def read_samples(file_name: str) -> List[Dict[str, int]]:
    samples = []
    with open(file_name, "r") as handler:
        lines = handler.readlines()
        # read first line of file to get variables in order
        variables = [str(v) for v in lines[0].split()]

        for i in range(1, len(lines)):
            vals = [int(v) for v in lines[i].split()]
            sample = dict(zip(variables, vals))
            samples.append(sample)

    return samples


class MLEBayesNet(BayesNet):
    """
    Placeholder class for the Bayesian Network that will learn CPDs using the frequentist MLE
    """
    def __init__(self, bn_file: str="data/bnet"):
        super(MLEBayesNet, self).__init__(bn_file=bn_file)
        self._reset_cpds()

    def _reset_cpds(self) -> None:
        """
        Reset the conditional probability distributions to a value of 0.5
        The values have to be **learned** from the samples.
        """
        for node_name in self.nodes:
            self.nodes[node_name].cpd["prob"] = 0.5

    def learn_cpds(self, samples: List[Dict[str, int]], alpha: float = 1.0) -> None:
        counter = {}

        for node_name in self.nodes:
            node = self.nodes[node_name]
            parents = [parent.var_name for parent in node.parent_nodes] # Extract parent names as strings
            parent_combinations = list(all_dicts(parents)) # Generate all possible combinations of parent values
            counter[node_name] = {tuple(pc.items()): {0: 0, 1: 0} for pc in parent_combinations} # Initialize counts

        # Phase 1: Count occurrences from the samples
        for sample in samples:
            for node_name in self.nodes:
                node = self.nodes[node_name]
                parents = [parent.var_name for parent in node.parent_nodes] 
                parent_values = tuple((p, sample[p]) for p in parents)  # Get parent values for the sample
                counter[node_name][parent_values][sample[node_name]] += 1  # Increment the corresponding count

        # Phase 2: Normalize to compute probabilities (MLE with Laplace smoothing)
        for node_name in self.nodes:
            node = self.nodes[node_name]
            parents = [parent.var_name for parent in node.parent_nodes]
            parent_combinations = list(all_dicts(parents))
            num_values = 2  # Binary variables (0 or 1)

            probabilities = []
            for parent in parent_combinations:
                parent_values = tuple(parent.items())
                total_sum = sum(counter[node_name][parent_values].values())

                for value in [0, 1]: # Apply Laplace smoothing for each value of the node (0 and 1)
                    laplace_smoothed_prob = (counter[node_name][parent_values][value] + alpha) / (total_sum + num_values * alpha)
                    probabilities.append((value, parent, laplace_smoothed_prob)) # Append the probability

            cpd_table = {
                "prob": [p[2] for p in probabilities],
                "value": [p[0] for p in probabilities],
            }
            for parent in parents:
                cpd_table[parent] = [dict(p[1])[parent] for p in probabilities]

            node.cpd = cpd_table


class EMBayesNet(MLEBayesNet):
    def __init__(self) -> None:  
        super(EMBayesNet, self).__init__(bn_file="data/bn_learning") 
        self.cpds = {}

    def learn_cpds(self, samples_with_missing: List[Dict[str, int]], alpha: float = 1.):
        # Initialize CPDs with random probabilities
        for node_name in self.nodes:
            node = self.nodes[node_name]
            parent_names = [p.var_name for p in node.parent_nodes]
            parent_combos = list(product([0, 1], repeat=len(parent_names)))
            for parent_vals in parent_combos:
                p1 = np.random.uniform(0.3, 0.7)
                index_0 = tuple([0] + list(parent_vals))
                index_1 = tuple([1] + list(parent_vals))
                node.cpd.loc[index_0, "prob"] = 1 - p1
                node.cpd.loc[index_1, "prob"] = p1

        # Create Junction Tree
        junction_tree = JunctionTree(self)

        max_iterations = 30
        prev_log_likelihood = float('-inf')

        for iteration in tqdm(range(max_iterations), desc="EM Iterations"):
            # E-step: Compute expected counts
            expected_counts = {node_name: {} for node_name in self.nodes}
            for node_name in self.nodes:
                node = self.nodes[node_name]
                parent_names = [p.var_name for p in node.parent_nodes]
                parent_combos = list(product([0, 1], repeat=len(parent_names)))

                for parent_vals in parent_combos:
                    for val in [0, 1]:
                        index = tuple([val] + list(parent_vals))
                        expected_counts[node_name][index] = 0.0

            # Process each sample
            directed_jt = junction_tree._get_junction_tree()
            log_likelihood = 0.0

            for sample in tqdm(samples_with_missing, desc=f"Processing samples (iteration {iteration + 1})", leave=False):
                # Get observed variables (non-missing values)
                evidence = {var: val for var, val in sample.items() if val != 2}

                # Get junction tree and load factors
                junction_tree._load_factors(directed_jt)
                uncalibrated_jt = junction_tree._incorporate_evidence(directed_jt, evidence)
                
                try:
                    calibrated_jt = junction_tree._run_belief_propagation(uncalibrated_jt)
                except Exception as e:
                    print(f"Error in belief propagation: {e}")
                    continue

                # Update expected counts
                for node_name in self.nodes:
                    node = self.nodes[node_name]
                    parent_names = [p.var_name for p in node.parent_nodes]

                    if node_name in evidence:
                        # For observed variables
                        val = evidence[node_name]
                        parent_vals = []
                        for p in parent_names:
                            parent_vals.append(evidence.get(p, 0))
                        index = tuple([val] + parent_vals)
                        expected_counts[node_name][index] += 1.0
                    else:
                        # For missing values
                        for clique in calibrated_jt.nodes:
                            if node_name in calibrated_jt.nodes[clique]['clique_members']:
                                potential = calibrated_jt.nodes[clique]['potential']
                                normalized_pot = normalize(potential.copy())

                                for val in [0, 1]:
                                    if node_name in normalized_pot.index.names:
                                        mask = normalized_pot.index.get_level_values(node_name) == val
                                        prob = normalized_pot[mask]['prob'].sum()

                                        parent_vals = []
                                        for p in parent_names:
                                            parent_vals.append(evidence.get(p, 0))

                                        index = tuple([val] + parent_vals)
                                        expected_counts[node_name][index] += prob
                                        if prob > 0:
                                            log_likelihood += np.log(prob)
                                break

            # M-step: Update CPDs
            for node_name in tqdm(self.nodes, desc="Updating CPDs", leave=False):
                node = self.nodes[node_name]
                parent_names = [p.var_name for p in node.parent_nodes]
                parent_combos = list(product([0, 1], repeat=len(parent_names)))

                for parent_vals in parent_combos:
                    count_0 = expected_counts[node_name][tuple([0] + list(parent_vals))]
                    count_1 = expected_counts[node_name][tuple([1] + list(parent_vals))]
                    total = count_0 + count_1 + 2 * alpha

                    node.cpd.loc[tuple([0] + list(parent_vals)), "prob"] = (count_0 + alpha) / total
                    node.cpd.loc[tuple([1] + list(parent_vals)), "prob"] = (count_1 + alpha) / total

            # Check convergence
            diff = log_likelihood - prev_log_likelihood
            print(f"Log likelihood difference: {diff}")
            if abs(diff) < 1e-4:
                print(f"\nEM converged after {iteration + 1} iterations")
                break

            prev_log_likelihood = log_likelihood

def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--file",
                            type=str,
                            default="data/bn_learning",
                            dest="file_name",
                            help="Input file")
    arg_parser.add_argument("-s", "--samplefile",
                            type=str,
                            default="bnlearning_samples_missing",
                            dest="samples_file_name",
                            help="Samples file")
    arg_parser.add_argument("-n", "--nsteps",
                            type=int,
                            default=1000,
                            dest="nsteps",
                            help="Number of optimization steps")
    arg_parser.add_argument("--lr",
                            type=float,
                            default=.005,
                            dest="lr",
                            help="Learning rate")

    return arg_parser.parse_args()


def main():
    args = get_args()
    table_bn = BayesNet(bn_file=args.file_name)
    em_bn = EMBayesNet()

    print("========== EM ==========")
    samples = read_samples(args.samples_file_name)
    em_bn.learn_cpds(samples)

    print("Reference BN")
    print(table_bn.pretty_print_str())

    print("MLE BayesNet after learning CPDs")
    print(em_bn.pretty_print_str())


if __name__ == "__main__":
    main()
