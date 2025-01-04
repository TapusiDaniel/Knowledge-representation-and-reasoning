from typing import List, Dict, Tuple, Any, Iterator
from bayes_net import BayesNet, BayesNode
from itertools import product
import numpy as np
from argparse import ArgumentParser, Namespace


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
    def __init__(self, bn_file: str = "data/bnet") -> None:
        super(EMBayesNet, self).__init__(bn_file=bn_file)
        self.cpds = {}  # type: Dict[str, Dict[tuple, float]]

    def learn_cpds(self, samples_with_missing: List[Dict[str, int]], alpha: float = 1.):
    # Step 1: Initialize parameters θ0i,j,k randomly
        for node_name in self.nodes:
            node = self.nodes[node_name]
            parents = [parent.var_name for parent in node.parent_nodes]
            parent_combinations = list(all_dicts(parents))
            
            probabilities = []
            for parent in parent_combinations:
                # Ensure probabilities sum to 1 for each parent combination
                values_prob = np.random.dirichlet([1, 1])
                probabilities.extend([
                    (0, parent, values_prob[0]),
                    (1, parent, values_prob[1])
                ])
                
            cpd_table = {
                "prob": [p[2] for p in probabilities],
                "value": [p[0] for p in probabilities],
            }
            for parent in parents:
                cpd_table[parent] = [dict(p[1])[parent] for p in probabilities]
                
            node.cpd = cpd_table

        # Repeat E-M steps for specified number of iterations
        num_iterations = 100
        for iteration in range(num_iterations):
            # E-step
            # Initialize expected counts N^i,j,k
            expected_counts = {node_name: {tuple(pc.items()): {0: 0.0, 1: 0.0} 
                            for pc in all_dicts([parent.var_name for parent in self.nodes[node_name].parent_nodes])}
                            for node_name in self.nodes}
            
            # Iterate through all samples
            for sample in samples_with_missing:
                # Get observed variables (Xo,d)
                observed_vars = {var: val for var, val in sample.items() if val != 2}
                
                # For each node Xi
                for node_name in self.nodes:
                    node = self.nodes[node_name]
                    parents = [parent.var_name for parent in node.parent_nodes]
                    
                    # Find clique containing Xi and its parents
                    # (Presupunem că avem o metodă pentru a găsi clica relevantă)
                    relevant_vars = set([node_name] + parents)
                    
                    # Compute γ(d)i,j,k using Junction Tree algorithm
                    if node_name in observed_vars:
                        # For observed variables, add direct counts
                        val = sample[node_name]
                        parent_values = tuple((p, sample.get(p, 0)) for p in parents)
                        expected_counts[node_name][parent_values][val] += 1
                    else:
                        # For missing variables, compute probabilities using Junction Tree
                        for parent_combo in all_dicts(parents):
                            parent_values = tuple(parent_combo.items())
                            
                            # Calculate probabilities for both values (0 and 1)
                            for value in [0, 1]:
                                # Create temporary sample with current configuration
                                temp_sample = observed_vars.copy()
                                temp_sample[node_name] = value
                                temp_sample.update(parent_combo)
                                
                                # Calculate γ(d)i,j,k using Junction Tree belief propagation
                                # This should use the clique potentials (φ*Cz)
                                prob = np.exp(self.sample_log_prob(temp_sample))
                                
                                # Add to expected counts
                                expected_counts[node_name][parent_values][value] += prob

            # M-step: Update parameters using expected counts
            for node_name in self.nodes:
                node = self.nodes[node_name]
                parents = [parent.var_name for parent in node.parent_nodes]
                parent_combinations = list(all_dicts(parents))
                
                probabilities = []
                for parent in parent_combinations:
                    parent_values = tuple(parent.items())
                    
                    # Calculate total counts for normalization
                    total = sum(expected_counts[node_name][parent_values].values()) + 2 * alpha
                    
                    # Update probabilities with Laplace smoothing
                    for value in [0, 1]:
                        count = expected_counts[node_name][parent_values][value]
                        prob = (count + alpha) / total
                        probabilities.append((value, parent, prob))
                
                # Update node's CPD
                cpd_table = {
                    "prob": [p[2] for p in probabilities],
                    "value": [p[0] for p in probabilities],
                }
                for parent in parents:
                    cpd_table[parent] = [dict(p[1])[parent] for p in probabilities]
                    
                node.cpd = cpd_table



def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--file",
                            type=str,
                            default="bn_learning",
                            dest="file_name",
                            help="Input file")
    arg_parser.add_argument("-s", "--samplefile",
                            type=str,
                            default="samples_bn_learning",
                            dest="samples_file_name",
                            help="Samples file")
    arg_parser.add_argument("-n", "--nsteps",
                            type=int,
                            default=10000,
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
    em_bn = EMBayesNet(bn_file=args.file_name)

    print("========== EM ==========")
    samples = read_samples(args.samples_file_name)
    em_bn.learn_cpds(samples)

    print("Reference BN")
    print(table_bn.pretty_print_str())

    print("MLE BayesNet after learning CPDs")
    print(em_bn.pretty_print_str())


if __name__ == "__main__":
    main()
