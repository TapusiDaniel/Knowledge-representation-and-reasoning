class Parser(object):
	@staticmethod
	def parse(file: str):
		'''
		@param file: path to the input file
		:returns Bayesian network as a dictionary {node: [list of parents], ...}
		and the list of queries as [{"X": [list of vars], 
		"Y": [list of vars], "Z": [list of vars]}, ... ] where we want 
		to test the conditional independence of vars1 ⊥ vars2 | cond 
		'''
		bn = {}
		queries = []

		with open(file) as fin:
			# read the number of vars involved
			# and the number of queries
			N, M = [int(x) for x in next(fin).split()]
			
			# read the vars and their parents
			for i in range(N):
				line = next(fin).split()
				var, parents = line[0], line[1:]
				bn[var] = parents

			# read the queries
			for i in range(M):
				vars, cond = next(fin).split('|')

				# parse vars
				X, Y = vars.split(';')
				X = X.split()
				Y = Y.split()

				# parse cond
				Z = cond.split()

				queries.append({
					"X": X,
					"Y": Y,
					"Z": Z
				})

			# read the answers
			for i in range(M):
				queries[i]["answer"] = next(fin).strip()

		return bn, queries

	@staticmethod
	def get_graph(bn: dict):
		'''
		@param bn: Bayesian netowrk obtained from parse
		:returns the graph as {node: [list of children], ...}
		'''
		graph = {}

		for node in bn:
			parents = bn[node]

			# this is for the leafs
			if node not in graph:
				graph[node] = []

			# for each parent add 
			# the edge parent->node
			for p in parents:
				if p not in graph:
					graph[p] = []
				graph[p].append(node)

		return graph

class DSeparation:
    def __init__(self, bn, graph):
        self.bn = bn
        self.graph = graph

    def get_parents(self, node):
        parents = set()
        for potential_parent, children in self.graph.items():
            if node in children:
                parents.add(potential_parent)
        return parents

    def get_children(self, node):
        return set(self.graph.get(node, []))

    def get_descendants(self, start):
        descendants = set()
        queue = [start]

        # Extracting the descendants using BFS
        while queue:
            node = queue.pop(0)
            children = self.get_children(node)
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    queue.append(child)
        return descendants

    def is_active_trail(self, start, end, observed, path=None):
        if path is None:
            if start in observed or end in observed:
                return False
            path = []
        
        if start == end:
            return True

        # Cycle    
        if start in path:
            return False
            
        new_path = path + [start]
        
        # Get neighbours
        parents = self.get_parents(start)
        children = self.get_children(start)
        
        # Exploring through neighbours
        for next_node in parents | children:
            if next_node in path:
                continue
                
            if len(new_path) >= 2:
                prev_node = new_path[-2]
                
                # Checking if it is a V-structure
                if prev_node in parents and next_node in parents:
                    descendants = self.get_descendants(start)
                    if start not in observed and not any(d in observed for d in descendants):
                        continue
                elif start in observed:
                    continue
                    
            if self.is_active_trail(next_node, end, observed, new_path):
                return True
                
        return False

    def check_independence(self, X, Y, Z):
        observed = set(Z)
        
        for x in X:
            for y in Y:
                if self.is_active_trail(x, y, observed):
                    return False
        return True

def solve_queries(bn, graph, queries):
    d_sep = DSeparation(bn, graph)
    results = []
    
    for query in queries:
        result = d_sep.check_independence(query["X"], query["Y"], query["Z"])
        answer = query["answer"]
        
        if answer == "true":
            expected = True
        else:
            expected = False

        if result == expected:
            results.append(True)
        else:
            results.append(False)
        
    return results

if __name__ == "__main__":
    from pprint import pprint
    
    bn, queries = Parser.parse("/home/danyez87/Master AI/KRR/Lab 4 - Bayesian Networks - Conditional Independence-20241107/bn2")
    graph = Parser.get_graph(bn)
    print(graph)
    results = solve_queries(bn, graph, queries)
    
    print("\nResults\n" + "-" * 50)
    for index in range(len(results)):
        result = results[index]
        query = queries[index]

        if result:
            status = 'Correct'
        else:
            status = 'Incorrect'

        print(f"Query {index + 1}: {status}")
        print(f"X: {query['X']}, Y: {query['Y']}, Z: {query['Z']}")
        print(f"Expected: {query['answer']}\n")
        