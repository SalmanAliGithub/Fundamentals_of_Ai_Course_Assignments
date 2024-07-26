
def file_to_graph(file, graph):
    with open(file, 'r') as f:
        for line in f:
            city1, city2, distance = line.strip().split()
            distance = int(distance)
            graph.add_node(city1)
            graph.add_node(city2)
            graph.add_edge(city1, city2, distance)
    return graph


class Graph:
    def __init__(self):
        self.graph = {}
        
    def add_node(self, node):
        if node not in self.graph:
            self.graph[node] = {}

    def remove_node(self, node):
        if node in self.graph:
            self.graph[node]
            for n in self.graph:
                if node in self.graph[n]:
                    del self.graph[n][node]

    def add_edge(self, node1, node2, weight=1):
        self.add_node(node1)
        self.add_node(node2)
        self.graph[node1][node2] = weight
        self.graph[node2][node1] = weight

    def remove_edge(self, node1, node2):
        if node1 in self.graph and node2 in self.graph[node1]:
            del self.graph[node1][node2]
            del self.graph[node2][node1]

    def get_adjacency_list(self):
        return self.graph
    def search(self,item):
        for i in self.graph:
            if item==i:
                return True
        return False

file = 'cities.txt'
graph=Graph()
cities_adj_list = file_to_graph(file,graph).get_adjacency_list()



import timeit

def dfs(graph, source, destination):
    visited = set()
    stack = [(source, [source])]

    while stack:
        current_node, path = stack.pop()
        if current_node == destination:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in reversed(graph.get(current_node, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None


start_time = timeit.default_timer()

pat = dfs(cities_adj_list, "Arad", "Bucharest")

end_time = timeit.default_timer()
time_taken = end_time - start_time

print(pat)
print(f"Time take: {time_taken} sec")