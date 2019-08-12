"""
AUthor: Raúl González Cruz
Date: 9/5/19
"""

import networkx as nx
import numpy, matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd

from scipy.cluster import hierarchy
from scipy.spatial import distance

import random
import math

# Set initial states
def set_states(graph):
	# States :{1,-1}
	possible_states = [-1,1]
	dict_states = []
	for value in range(nx.number_of_nodes(graph)):
		# Add state
		dict_states.append(random.choice(possible_states))
	return dict_states

# Set initial weights
def set_weights(graph):
	# Weights
	m_weights = [[0 for j in range(nx.number_of_nodes(graph))] for i in range(nx.number_of_nodes(graph))]
	for e in graph.edges:
		#graph.add_edge(e[0], e[1], weight=round(random.uniform(0,2),4))
		w = round(random.uniform(-2,2),3)
		#w = round(random.uniform(0,1),4)
		m_weights[e[0]][e[1]] = w
		m_weights[e[1]][e[0]] = w
	return m_weights

# Get energy of graph
def get_energy(m_weights, states):
	energy = 0
	for i in range(nx.number_of_nodes(graph)):
		for j in range(nx.number_of_nodes(graph)):
			energy += m_weights[i][j] * states[i] * states[j]
		#print(u,v,w)
	energy *= -1
	#print(energy)
	return energy

# Asynchronus update
def update_states(number_nodes,m_weights, states):
	unit = random.randint(0,number_nodes-1)
	pound = 0
	for j in range(number_nodes):
		pound += m_weights[unit][j] * states[j]
	if pound >= 0:
		states[unit] = 1
	else:
		states[unit] = -1
	return states

# Get final weight
def update_weights_lo(number_nodes,m_weights_o, m_weights_l):
	m_weights = [[0 for j in range(number_nodes)] for i in range(number_nodes)]
	for i in range(number_nodes):
		for j in range(number_nodes):
			if(i!=j):
				pound = m_weights_o[i][j] + m_weights_l[i][j]
				if pound > 2:
					m_weights[i][j] = 2
				elif pound < -2:
					m_weights[i][j] = -2
				else:
					m_weights[i][j] = pound
	return m_weights

# Synchronus update
def update_weights(number_nodes,m_weights,states):
	delta = 0.001/10*number_nodes
	for i in range(number_nodes):
		for j in range(number_nodes):
			if(i!=j):
				m_weights[i][j] += m_weights[i][j] + (delta*states[i]*states[j])
	return m_weights

# Get a random color given a test case
def get_color():
	r = lambda: random.randint(0,255)
	return '#%02X%02X%02X' % (r(),r(),r())

# Plot the graph
def plot_graph(graph, states):
	# Establish color
	color_map = []
	for index in range(len(states)):
		if states[index] == 1:
			color_map.append("blue")
		else:
			color_map.append("green")	
	pos = nx.circular_layout(graph)         #Circular graph
	arc_weight=nx.get_edge_attributes(graph,'weight') 	# The edge weights of each arcs are stored in a dictionary
	nx.draw(graph,pos,node_color = color_map,with_labels=True,node_size = 250)  # Parameters for plotting
	nx.draw_networkx_edge_labels(graph, pos, edge_color= 'black', edge_labels=arc_weight)

	plt.show()

# Plot energy results
def plotResultsMethod(max_updates, dataB, title):
    x = [i for i in range(max_updates)]
    #print(x)
    #print(y)
    
    index = 1
    for test in dataB:
    	l1, = plt.plot(x,test,get_color(), label='Test '+str(index))
    	index += 1

    plt.title(title)
    plt.xlabel("State updates")
    plt.ylabel("Energy value")
    plt.legend(loc='upper right')
       
    plt.show()

def run_simulation(graph,m_weights,tests):
	number_tests = tests
	N = nx.number_of_nodes(graph)
	relaxation_length = 10*N
	atractors_record = []	
	tests_record = []
	dict_states = []
	for gen in range(number_tests):
		dict_states = set_states(graph)
		energy_record = []
		for i in range(relaxation_length):
			value = get_energy(m_weights, dict_states)
			energy_record.append(value)
			# Update state
			dict_states = update_states(N,m_weights, dict_states)
		print("\nFinished simulation "+str(gen))
		tests_record.append(energy_record)
		atractors_record.append(energy_record[relaxation_length-1]) 

	# Plot results
	print("List of atractors found per test:")
	print(atractors_record)
	print("Best atractor found:")
	print(min(atractors_record))
	plotResultsMethod(relaxation_length,tests_record, "Self-modeling of network without learning")
	return dict_states

def run_simulation_with_learning(graph,m_weights_o,tests):
	number_tests = tests
	N = nx.number_of_nodes(graph)
	relaxation_length = 10*N
	atractors_record = []	
	tests_record = []
	# Variables for atractor
	last_value = 0
	#times_atractor = (3/4**N)*math.log(N,math.e)
	times_atractor = int(relaxation_length*.075)
	print("Relaxation selected", times_atractor)
	dict_states = []
	# Initial learning
	m_weights_l = [[0 for j in range(N)] for i in range(N)]
	for gen in range(number_tests):
		dict_states = set_states(graph)
		m_weights = [[0 for j in range(N)] for i in range(N)]
		energy_record = []
		counter_atractor = 0
		for i in range(relaxation_length):
			# Check repeated atractor
			value = get_energy(m_weights_o, dict_states)
			if last_value == value:
				counter_atractor += 1
			else:
				counter_atractor = 0
				last_value = value
			# Update state
			if counter_atractor > times_atractor:
				# Convergence
				print("Pattern recognized with " + str(value))
				m_weights_l = update_weights(N,m_weights_l,dict_states)
			m_weights = update_weights_lo(N,m_weights_o, m_weights_l)
			dict_states = update_states(N,m_weights, dict_states)
			energy_record.append(value)
		print("\nFinished simulation "+str(gen))
		tests_record.append(energy_record)
		atractors_record.append(energy_record[relaxation_length-1]) 

	# Plot results
	print("List of atractors found per test:")
	print(atractors_record)
	print("Best atractor found:")
	print(min(atractors_record))
	plotResultsMethod(relaxation_length,tests_record, "Self-modeling of network with learning")
	return dict_states

def cluster_graph(graph,name):
	#Clustering
	print("Clustering the graph...")
	path_length=nx.all_pairs_shortest_path_length(graph)
	print(path_length)
	n = len(graph.nodes())
	distances=numpy.zeros((n,n))
	for u,p in path_length:
		for v,d in p.items():
			distances[int(u)-1][int(v)-1] = d
	sd = distance.squareform(distances)
	hier = hierarchy.average(sd)
	hierarchy.dendrogram(hier)
	pylab.savefig(name,format="png")
	print("Clustering completed!")

# Watt-Strogatz small world configuration
tests = 10
number_nodes = 100
graph = nx.random_graphs.watts_strogatz_graph(number_nodes,50,0.5)  #Nodes, most connections, probability of conection
#graph = nx.complete_graph(number_nodes)
print("Number of nodes:")
print(nx.number_of_nodes(graph)) # Prints out the nodes
states = set_states(graph)
m_weights = set_weights(graph)

# Self-modeling without learning
print("\nStarting simulation without learning...")
#states = run_simulation(graph,m_weights,tests)
# Show results
#plot_graph(graph, states)

# Self-modeling with learning
print("\nStarting simulation with learning...")
states = run_simulation_with_learning(graph,m_weights,tests)
# Show results
plot_graph(graph, states)

cluster_graph(graph,"init.png")



#print(list(nx.connected_components(graph)))
#print(graph.edges.data())

# Free scale network
"""
graph = nx.scale_free_graph(100)
graph = nx.to_undirected(graph)
"""

#Worm wiring
"""
print("Processing worm data...")
df=pd.read_csv('herm_full_edgelist.csv', sep=',',header=0)
dict_worm = {}
omit_neurons = ["I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6", "M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5", "MCL", "MCR", "MI", "NSML", "NSMR"]
for value in df.values:
	if value[3].strip()=="chemical" and value[0].strip() not in omit_neurons and value[1].strip() not in omit_neurons and (value[0].strip(),value[1].strip()) not in dict_worm and (value[1].strip(),value[0].strip()) not in dict_worm:
		dict_worm[(value[0].strip(),value[1].strip())] = value[2]
print("Dictionary completed!")
print(len(dict_worm))

graph_worm = nx.Graph()
#Self
# States :{1,-1}
possible_states = [-1,1]
dict_states = {}
for key,value in dict_worm.items():
	graph_worm.add_nodes_from([*key])
	graph_worm.add_edge(*key,weight=value)
	# Add state
	dict_states[key[0]] = random.choice(possible_states)
	dict_states[key[1]] = random.choice(possible_states)

print("Number of nodes")
print(nx.number_of_nodes(graph_worm)) # Prints out the nodes
#print(list(nx.connected_components(graph)))




pos = nx.circular_layout(graph_worm)         #Circular graph
arc_weight=nx.get_edge_attributes(graph_worm,'weight') 	# The edge weights of each arcs are stored in a dictionary
nx.draw(graph_worm,pos,with_labels=True,node_size = 300)  # Parameters for plotting
nx.draw_networkx_edge_labels(graph_worm, pos, edge_color= 'black', edge_labels=arc_weight)

plt.show()



#Hebbian
max_generations = 500
energy_record = []
for index in range(max_generations):
	energy = 0
	#Return a tuple
	for u,v,w in graph_worm.edges(data=True):
		#print(u,v,w)
		energy += w["weight"] * dict_states[u] * dict_states[v]
		# Updat weight
		if dict_states[u] * dict_states[v] > 0:
			w["weight"] -= 0.1
		else:
			w["weight"] += 0.1
	#print("Energy", energy)
	energy_record.append(energy)

# Plot energy
plotResultsMethod(energy_record,max_generations,"Hebbian Learning applied")
"""