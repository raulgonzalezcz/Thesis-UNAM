"""
Author: Raul Gonzalez Cruz
Date: 9/5/19
"""

import networkx as nx
import numpy as np
import numpy, matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
import scipy
import scipy.spatial.distance as ssd

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
		graph.nodes[value]["state"] = random.choice(possible_states)

# Asynchronus update
def update_states(number_nodes,graph):
	unit = random.randint(0,number_nodes-1)
	pound = 0
	for j in list(graph.adj[unit]):
		pound += graph[unit][j]["weightr"] * graph.nodes[j]["state"]
	
	if pound >= 0:
		graph.nodes[unit]["state"] = 1
	else:
		graph.nodes[unit]["state"] = -1

# Set initial weights
def set_weights_o(graph, k, p, mode):
	possible_states = [-1,1]

	# Set WO (no changes in this weight)
	for e in graph.edges:
		#S1. Random sparse symmetric matrices.
		if mode == "S1":
			graph.add_edge(e[0], e[1], weight=random.choice(possible_states))
		
		#S2. Modular consistent connectivity matrix.
		else:
			if math.floor(e[0]/k) == math.floor(e[1]/k):
				graph.add_edge(e[1], e[0], weight=1)
			else:
				graph.add_edge(e[1], e[0], weight=p)
		
		#print("Edge1: ",e[0],e[1],graph[e[0]][e[1]]["weight"])
		#print("Edge2: ",e[1],e[0],graph[e[1]][e[0]]["weight"])
	
	#Check initial configuration of weights
	#print(graph.edges.data())

def set_weights_l(graph):
	# For unsupervised learning, set WL
	for e in graph.edges:
		graph.add_edge(e[0], e[1], weightl=0)

def set_weights_r(graph):
	# For unsupervised learning, set WR
	for e in graph.edges:
		graph.add_edge(e[0], e[1], weightr=graph[e[0]][e[1]]["weight"])

def get_edge_data(graph):
	print(graph.edges.data())

# Set weight used for updating the states
def update_weights_lo(graph):
	for i in list(graph.edges):
		pound = graph[i[0]][i[1]]["weight"] + graph[i[0]][i[1]]["weightl"]
		if pound > 1:
			graph[i[0]][i[1]]["weightr"] = 1
		elif pound < -1:
			graph[i[0]][i[1]]["weightr"] = -1
		else:
			graph[i[0]][i[1]]["weightr"] = pound

# Synchronus update
def update_weightsl(graph,delta):
	for i in list(graph.edges):
		graph[i[0]][i[1]]['weightl'] = graph[i[0]][i[1]]['weightl'] + (delta*graph.nodes[i[0]]["state"]*graph.nodes[i[1]]["state"])


# Get energy of graph
def get_energy(graph):
	energy = 0
	#print(list(graph.edges))
	for i in list(graph.edges):
		energy += graph[i[0]][i[1]]["weight"] * graph.nodes[i[0]]["state"] * graph.nodes[i[1]]["state"]
        #print(u,v,w)
	energy *= -5
	#print(energy)
	return energy

# Get a random color given a test case
def get_color():
	r = lambda: random.randint(0,255)
	return '#%02X%02X%02X' % (r(),r(),r())

# Plot the graph
def plot_graph(graph):
	# Establish color
	color_map = []
	for index in range(nx.number_of_nodes(graph)):
		if graph.nodes[index]["state"] == 1:
			color_map.append("blue")
		else:
			color_map.append("green")	
	pos = nx.circular_layout(graph)         #Circular graph
	#arc_weight=nx.get_edge_attributes(graph,'weight') 	# The edge weights of each arcs are stored in a dictionary
	nx.draw(graph,pos,node_color = color_map,with_labels=True,node_size = 250)  # Parameters for plotting
	#nx.draw_networkx_edge_labels(graph, pos, edge_color= 'black', edge_labels=arc_weight)

	plt.show()

# Plot energy results
def plotResultsMethod(max_updates, dataB, title):
	print("Plotting results of energy")
	x = [i for i in range(max_updates)]
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

# Plot energy frequency results
def plotResultsFrequencyMethod(max_updates, dataNL, dataWL, title):
	width = 0.2
	print("Plotting results of frequency")
	x = [i for i in range(max_updates)]
	#print(x)
	#print(y)
	
	freqNL = {}
	freqWL = {}
	[ freqNL.update( {i:freqNL.get(i, 0)+1} ) for i in dataNL ]
	[ freqWL.update( {i:freqWL.get(i, 0)+1} ) for i in dataWL ]
	print(freqNL)
	print(freqWL)

	NLMeans = []
	WLMeans = []
	indicesNL = []
	indicesWL = []

	for key,value in freqNL.items():
		indicesNL.append(key + width)
		NLMeans.append(value/len(dataNL) )

	for key,value in freqWL.items():
		indicesWL.append(key)
		WLMeans.append(value/len(dataWL) )
	
	width = 0.2
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.bar(indicesNL,NLMeans,width,color='b',label='before learning')
	ax.bar(indicesWL,WLMeans,width,color='r',label='after learning')
	plt.title(title)
	plt.xlabel("Energy")
	plt.ylabel("Frequency")
	plt.legend(loc='upper right')
	
	plt.show()

# Plot energy frequency results
def plotResultsMoveMethod(bestBL, bestDL, bestAL, title):
	print("Plotting results (before, during and after learning)...")
	#print(x)
	#print(y)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	index = 0
	for test in bestBL:
		#print("Line:",test)
		ax.scatter(index, test, c = 'blue')
		index += 2

	plt.axvline(x=index)

	for test in bestDL:
		#print("Line:",test)
		ax.scatter(index, test, c = 'orange')
		index += 2

	plt.axvline(x=index)
	for test in bestAL:
		#print("Line:",test)
		ax.scatter(index, test, c = 'green')
		index += 2
	
	ax.legend(loc='upper right')
	plt.title(title)
	plt.xlabel("Relaxations")
	plt.ylabel("Energy value")
	
	plt.show()

def run_simulation(graph,tests,relaxation_length,label_plot):
	N = nx.number_of_nodes(graph)
	atractors_record = []	
	tests_record = []
	
	for gen in range(tests):
		energy_record = []
		
		for i in range(relaxation_length):
			value = get_energy(graph)
			energy_record.append(value)
			# Update state 
			update_states(N,graph)

		print("\nFinished simulation "+str(gen))
		if len(tests_record) < 10:
			tests_record.append(energy_record)
		atractors_record.append(energy_record[relaxation_length-1])

		# Set new states
		set_states(graph) 

	# Plot results
	print("List of atractors found per test:")
	print(atractors_record)
	print("Best atractor found:")
	print(min(atractors_record))
	plotResultsMethod(relaxation_length,tests_record, label_plot)
	return atractors_record

def run_simulation_with_learning(graph,tests,relaxation_length,label_plot):
	N = nx.number_of_nodes(graph)	
	delta = 0.0001

	atractors_record = []	
	tests_record = []
	
	#times_atractor = int( math.pow(3/4,N)*math.log(N) )
	times_atractor = relaxation_length*.4
	print("Relaxation selected", times_atractor)

	for gen in range(tests):
		energy_record = []

		last_value = 0
		counter_atractor = 0
		#set_weights_r(graph)

		for i in range(relaxation_length):
			value = get_energy(graph)
			"""
			if last_value == value:
				counter_atractor += 1
			else:
				last_value=value
				counter_atractor=0

			if counter_atractor > times_atractor:
				print("Atractor found==,relaxation ", i)
				update_weightsl(graph,delta)
				update_weights_lo(graph)
			"""
			update_weightsl(graph,delta)
			update_weights_lo(graph)
			update_states(N,graph)
			energy_record.append(value)

		print("\nFinished learning simulation "+str(gen))
		if len(tests_record) < 10:
			tests_record.append(energy_record)
		atractors_record.append(energy_record[relaxation_length-1]) 

		# Set new states
		set_states(graph)
		get_edge_data(graph)


	# Plot results
	print("List of atractors found per test:")
	print(atractors_record)
	print("Best atractor found:")
	print(min(atractors_record))
	plotResultsMethod(relaxation_length,tests_record, label_plot)
	return atractors_record

def cluster_graph(graph,name,m_weights):
	#Clustering coefficient based on connections
	cc=nx.average_clustering(graph)
	print("Output of Global CC", cc)
	c=nx.clustering(graph) 
	print("Output of local CC ", c)
	
	print("Clustering the graph...")
	path_length=nx.all_pairs_shortest_path_length(graph)
	# According to average path length
	"""
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
	"""
	# According to weighted edges
	# convert the redundant n*n square matrix form into a condensed nC2 array
	# distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
	distArray = ssd.squareform(m_weights)
	sd = distance.squareform(distArray)
	hier = hierarchy.average(sd)
	hierarchy.dendrogram(hier)
	pylab.savefig(name,format="png")
	print("Clustering completed!")
	

def show_weighted_edges(m_weights):
	print(m_weights)
	

# Configuration variables
tests = 1000
number_nodes = 50
relaxation_length = 20*number_nodes

#k_neighbors = int(number_nodes *.75)
k_neighbors = 7
rewiring = 0.6
edge_creation = 0.2
p = 0.01

# Create graphs
#graph = nx.random_graphs.watts_strogatz_graph(number_nodes,k_neighbors,rewiring)  #Nodes, Each node is joined with its k nearest neighbors in a ring topology, The probability of rewiring each edge
graph = nx.complete_graph(number_nodes)
#graph = nx.erdos_renyi_graph(number_nodes,edge_creation)
print("Number of nodes:")
print(nx.number_of_nodes(graph)) # Prints out the nodes

# Set states and nodes
set_states(graph)
set_weights_o(graph, k_neighbors, p, "S2")
set_weights_l(graph)
set_weights_r(graph)
get_edge_data(graph)
plot_graph(graph)

# Phase 1: Self-modeling without learning
print("\nStarting simulation before learning...")
bestBL = run_simulation(graph,tests,relaxation_length,'Self-modeling of network before learning')
get_edge_data(graph)
# Show results
#plot_graph(graph)

# Phase 2: Self-modeling for learning
print("\nStarting simulation during learning...")
bestDL = run_simulation_with_learning(graph,tests,relaxation_length,'Self-modeling of network during learning')
get_edge_data(graph)
# Show results
# plot_graph(graph)

# Phase 3: Self-modeling with learning
print("\nStarting simulation after learning...")
bestAL = run_simulation(graph,tests,relaxation_length,'Self-modeling of network after learning')
get_edge_data(graph)

# Show comparison
plotResultsMoveMethod(bestBL, bestDL, bestAL, "Atractors states visited before, during and after learning")
plotResultsFrequencyMethod(10*number_nodes, bestBL, bestAL, "Frequency of atractors without and with learning")

# Clustering method
# show_weighted_edges(m_weights)
# cluster_graph(graph,"init.png",m_weights)



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

# srun -t 0-4 -c 8 --mem=15G --x11 --pty bash