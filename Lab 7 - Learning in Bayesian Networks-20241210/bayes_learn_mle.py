from typing import List, Dict, Tuple, Any, Iterator
from bayes_net import BayesNet, BayesNode
from itertools import product
import numpy as np
from argparse import ArgumentParser, Namespace


def all_dicts(variables: List[str]) -> Iterator[Dict[str, int]]:
    for keys in product(*([[0, 1]] * len(variables))):
        yield dict(zip(variables, keys))


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
    def __init__(self, bn_file: str="bn_learning"):
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
        counts = {}

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
    
    return arg_parser.parse_args()


def main():
    args = get_args()
    table_bn = BayesNet(bn_file=args.file_name)
    mle_bn = MLEBayesNet(bn_file=args.file_name)

    print("========== Frequentist MLE ==========")
    samples = read_samples(args.samples_file_name)
    mle_bn.learn_cpds(samples)

    print("Reference BN")
    print(table_bn.pretty_print_str())

    print("MLE BayesNet after learning CPDs")
    print(mle_bn.pretty_print_str())


if __name__ == "__main__":
    main()
