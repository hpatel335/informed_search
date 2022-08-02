# coding=utf-8
from audioop import mul
import heapq
import os
import pickle
import math
from tkinter import N

from numpy import empty

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue      = []
        self.counter    = 0       # Create counter  
        self.priorities = []      # List to track priorities 

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        top_entry = heapq.heappop(self.queue)
        return (top_entry[0], top_entry[2])

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue. 
                  [priority, entry]
        """        
        # Index Elements 
        element     = node[1]
        priority    = node[0]

        # Adjust counter to Impliment FIFO for same priority items 
        if priority in self.priorities: 
            self.counter += 1 
        else: 
            self.counter = 1
        self.priorities.append(priority)
        entry = (priority, self.counter, element)

        # Add task to PQ 
        heapq.heappush(self.queue, entry)
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    
    # Constants 
    path     = []
    priority    = 1 
    frontier = PriorityQueue() 
    empty_node     = {'state'     : start, 
                      'parent'    : start, 
                      'path'      : [start],
                      'action'    : [], 
                      'path_cost' : []}
    
    # Carry out BFS i.f.f we start at some location other than goal
    if start != goal: 
        node = empty_node.copy()
        frontier.append((priority, node))
        reached = [start] 

        while not frontier.size() == 0: 
            node = frontier.pop()[1]
            neighbors = graph[node['state']]
            for neighbor in sorted(neighbors): 
                child_node = empty_node.copy()
                child_node['state']     = neighbor 
                child_node['parent']    = node['state']
                child_node['path']      = [*node['path'], neighbor]
                if neighbor == goal: 
                    return child_node['path']
                elif child_node['state'] not in reached: 
                    reached.append(child_node['state'])
                    priority += 1 
                    frontier.append((priority, child_node))
    elif start == goal: 
        return path
    else: 
        return "Path Not Found!"


def uniform_cost_search(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # Constants 
    path     = []
    frontier = PriorityQueue() 
    empty_node     = {'state'     : start, 
                      'parent'    : start, 
                      'path'      : [start],
                      'path_cost' : 0}
    # UCS Algorithm
    if start != goal: 
        node = empty_node.copy()
        frontier.append((1, node))
        reached = {start : node} 

        while not frontier.size() == 0:
            node = frontier.pop()[1] 
            if node['state'] == goal: 
                return node['path']

            neighbors = graph[node['state']]

            for neighbor in sorted(neighbors): 
                child_node = empty_node.copy()
                child_node['state']         = neighbor 
                child_node['parent']        = node['state']
                child_node['path']          = [*node['path'], neighbor]
                child_node['path_cost']     = node['path_cost'] + \
                                              graph.get_edge_weight(node['state'], neighbor)

                r_keys = reached.keys()
                s = child_node['state']
                if s not in r_keys or child_node['path_cost'] < reached[s]['path_cost']: 
                    reached[s] = child_node 
                    frontier.append((child_node['path_cost'], child_node))
    elif start == goal: 
        return path
    else: 
        return "Path Not Found!"


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """
    
    # Impliment Euclidean Distance Heuristic 
    p1 = graph.nodes[v]['pos']
    p2 = graph.nodes[goal]['pos']
    d = (math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))    
    return d 


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # Constants 
    path     = []
    frontier = PriorityQueue() 
    orig_dist = heuristic(graph, start, goal)
    empty_node     = {'state'     : start, 
                      'parent'    : start, 
                      'path'      : [start],
                      'g_cost'    : 0, 
                      'h_cost'    : orig_dist,
                      'f_cost'    : 0 + orig_dist}
    # A* Algorithm
    if start != goal: 
        node = empty_node.copy()
        frontier.append((1, node))
        reached = {start : node} 

        while not frontier.size() == 0:
            node = frontier.pop()[1] 
            if node['state'] == goal: 
                return node['path']
            
            neighbors = graph[node['state']]

            for neighbor in sorted(neighbors): 
                child_node = empty_node.copy()
                child_node['state']         = neighbor 
                child_node['parent']        = node['state']
                child_node['path']          = [*node['path'], neighbor]
                child_node['g_cost']        = node['g_cost'] + \
                                              graph.get_edge_weight(node['state'], neighbor)
                child_node['h_cost']        = heuristic(graph, neighbor, goal) 
                child_node['f_cost']        = child_node['g_cost'] + child_node['h_cost']
                r_keys = reached.keys()
                s = child_node['state']
                if s not in r_keys or child_node['f_cost'] < reached[s]['f_cost']: 
                    reached[s] = child_node 
                    frontier.append((child_node['f_cost'], child_node))
    elif start == goal: 
        return path
    else: 
        return "Path Not Found!"


def bidirectional_ucs(graph, start, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # Initialization 

    node_f = {'state'     : start, 
              'parent'    : start, 
              'path'      : [start],
              'weight'    : 0}
    node_b = {'state'     : goal, 
              'parent'    : goal, 
              'path'      : [goal],
              'weight'    : 0}

    frontier_f      = PriorityQueue()
    frontier_b      = PriorityQueue() 
    frontier_f.append((node_f['weight'], node_f))
    frontier_b.append((node_b['weight'], node_b))
    reached_f       = {start    : node_f}
    reached_b       = {goal     : node_b} 
    final_path      = [] 
    mu              = float('inf')

    # Start Bidirectional UCS 
    if start != goal :

        # Implimenting Algorithm 3.14
        while not frontier_f.size() == 0 and not frontier_b.size() == 0:

            # Forward Pass 
            nodef = frontier_f.pop()[1]
            if nodef['state'] in reached_b.keys() and reached_f[nodef['state']]['weight'] + reached_b[nodef['state']]['weight'] < mu: 
                final_path = reached_f[nodef['state']]['path'][:-1] + [nodef['state']] + reached_b[nodef['state']]['path'][1:]
                mu = reached_f[nodef['state']]['weight'] + reached_b[nodef['state']]['weight']

            for child in graph[nodef['state']]: 
                child_node = node_f.copy()
                child_node['state']         = child 
                child_node['parent']        = nodef['state']
                child_node['path']          = [*nodef['path'], child]
                child_node['weight']        = nodef['weight'] + \
                                              graph.get_edge_weight(nodef['state'], child)
                s = child_node['state'] 
                if s not in reached_f.keys() or child_node['weight'] < reached_f[s]['weight']:
                    reached_f[s] = child_node 
                    frontier_f.append((child_node['weight'], child_node))
                
            # Backward Pass 
            nodeb = frontier_b.pop()[1]
            if nodeb['state'] in reached_f.keys() and reached_f[nodeb['state']]['weight'] + reached_b[nodeb['state']]['weight'] < mu: 
                final_path = reached_f[nodeb['state']]['path'][:-1] + [nodeb['state']] + reached_b[nodeb['state']]['path'][1:]
                mu = reached_f[nodeb['state']]['weight'] + reached_b[nodeb['state']]['weight']   

            for child in graph[nodeb['state']]: 
                child_node = node_b.copy()
                child_node['state']         = child 
                child_node['parent']        = nodeb['state']
                child_node['path']          = [child, *nodeb['path']]
                child_node['weight']        = nodeb['weight'] + \
                                              graph.get_edge_weight(nodeb['state'], child)
                s = child_node['state'] 
                if s not in reached_b.keys() or child_node['weight'] < reached_b[s]['weight']:
                    reached_b[s] = child_node 
                    frontier_b.append((child_node['weight'], child_node))
    
            # Check for Termination Condition 
            if frontier_f.top()[0] + frontier_b.top()[0] >= mu: 
                return final_path 
            
    elif start == goal: 
        return final_path
    else: 
        return "Path Not Found!"


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # Initialization 
    orig_dist = heuristic(graph, start, goal)

    node_f = {'state'     : start, 
              'parent'    : start, 
              'path'      : [start],
              'h_weight'  : orig_dist, 
              'g_weight'  : 0,
              'f_weight'  : orig_dist}
    node_b = {'state'     : goal, 
              'parent'    : goal, 
              'path'      : [goal],
              'h_weight'  : orig_dist, 
              'g_weight'  : 0,
              'f_weight'  : orig_dist}

    frontier_f      = PriorityQueue()
    frontier_b      = PriorityQueue() 
    frontier_f.append((node_f['f_weight'], node_f))
    frontier_b.append((node_b['f_weight'], node_b))
    
    reached_f       = {start    : node_f}
    reached_b       = {goal     : node_b} 
    final_path      = [] 
    mu              = float('inf')

    # Start Bidirectional A* 
    if start != goal :

        # Implimenting Algorithm 3.14
        while not frontier_f.size() == 0 and not frontier_b.size() == 0:

            # Forward Pass 
            nodef = frontier_f.pop()[1]
            if nodef['state'] in reached_b.keys() and reached_f[nodef['state']]['f_weight'] + reached_b[nodef['state']]['f_weight'] < mu: 
                final_path = reached_f[nodef['state']]['path'][:-1] + [nodef['state']] + reached_b[nodef['state']]['path'][1:]
                mu = reached_f[nodef['state']]['f_weight'] + reached_b[nodef['state']]['f_weight']

            for child in graph[nodef['state']]: 
                child_node = node_f.copy()
                child_node['state']         = child 
                child_node['parent']        = nodef['state']
                child_node['path']          = [*nodef['path'], child]
                child_node['g_weight']      = nodef['g_weight'] + \
                                              graph.get_edge_weight(nodef['state'], child)
                child_node['h_weight']      = heuristic(graph, child, goal)
                child_node['f_weight']      = child_node['h_weight'] + child_node['g_weight']

                s = child_node['state'] 
                if s not in reached_f.keys() or child_node['f_weight'] < reached_f[s]['f_weight']:
                    reached_f[s] = child_node 
                    frontier_f.append((child_node['f_weight'], child_node))

                
            # Backward Pass 
            nodeb = frontier_b.pop()[1]
            if nodeb['state'] in reached_f.keys() and reached_f[nodeb['state']]['f_weight'] + reached_b[nodeb['state']]['f_weight'] < mu: 
                final_path = reached_f[nodeb['state']]['path'][:-1] + [nodeb['state']] + reached_b[nodeb['state']]['path'][1:]
                mu = reached_f[nodeb['state']]['f_weight'] + reached_b[nodeb['state']]['f_weight']   

            for child in graph[nodeb['state']]: 
                child_node = node_b.copy()
                child_node['state']         = child 
                child_node['parent']        = nodeb['state']
                child_node['path']          = [child, *nodeb['path']]
                child_node['g_weight']      = nodeb['g_weight'] + \
                                              graph.get_edge_weight(nodeb['state'], child)
                child_node['h_weight']      = heuristic(graph, child, start)
                child_node['f_weight']      = child_node['h_weight'] + child_node['g_weight']
                s = child_node['state'] 
                if s not in reached_b.keys() or child_node['f_weight'] < reached_b[s]['f_weight']:
                    reached_b[s] = child_node 
                    frontier_b.append((child_node['f_weight'], child_node))
    
            # Check for Termination Condition 
            if frontier_f.top()[0] + frontier_b.top()[0] >= mu: 
                return final_path 
            
    elif start == goal: 
        return final_path
    else: 
        return "Path Not Found!"
        

def tridirectional_search(graph, goals): 

    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """   

    # Initialize Containers
    final_path = []
    first, second, third = goals[:] 

    empty_node = {"state"   : "", 
                  "path"    : [], 
                  "weight"  : 0}
    node_a, node_b, node_c = empty_node.copy(), empty_node.copy(), empty_node.copy()  
    node_a["state"], node_b["state"], node_c["state"] = first, second, third 
    node_a["path"], node_b["path"], node_c["path"] = [first], [second], [third] 
    
    explored_a = {first: node_a} 
    explored_b = {second: node_b} 
    explored_c = {third: node_c} 
    frontier_a_dict = {first: node_a}
    frontier_b_dict = {second: node_b} 
    frontier_c_dict = {third: node_c} 
    frontier_a, frontier_b, frontier_c = PriorityQueue(), PriorityQueue(), PriorityQueue() 
    frontier_a.append((0, node_a)) 
    frontier_b.append((0, node_b))
    frontier_c.append((0,node_c)) 

    mu_ab = float('inf')
    mu_bc = float('inf')
    mu_ca = float('inf')  

    solution_a = False
    solution_b = False
    solution_c = False

    if first == second == third: 
        return final_path 

    # Begin main iteration
    while not all([solution_a, solution_b, solution_c]): 
        
        # Thread A 
        if frontier_a.size() != 0: 
            a = frontier_a.pop()[1] 
            a_state = a['state']
            frontier_a_dict.pop(a_state)
            explored_a[a_state] = {'path'    : a['path'], 
                                   'weight'  : a['weight'],
                                   'state'   : a_state}
            
            check_b_dict = {**explored_b, **frontier_b_dict} 
            check_c_dict = {**explored_c, **frontier_c_dict}
            if a_state in check_b_dict.keys() and a['weight'] + check_b_dict[a_state]['weight']< mu_ab:
                mu_ab = a['weight'] + check_b_dict[a_state]['weight'] 
                path_ab = a['path'] + check_b_dict[a_state]['path'][::-1][1:]
            
            if a_state in check_c_dict.keys() and a['weight'] + check_c_dict[a_state]['weight']< mu_ca:
                mu_ca = a['weight'] + check_c_dict[a_state]['weight'] 
                path_ac = a['path'] + check_c_dict[a_state]['path'][::-1][1:]

            for child in graph[a_state]: 
                child_node = empty_node.copy() 
                child_node['state']         = child
                child_node['path']          = [*a['path'], child]
                child_node['weight']        = a['weight'] + \
                                            graph.get_edge_weight(a['state'], child) 

                if child not in explored_a.keys() or explored_a[child]['weight'] > child_node['weight']: 
                    for node in frontier_a: 
                        if child == node[2]['state']: 
                            if node[0] > child_node['weight']: 
                                frontier_a.queue.remove(node)
                                frontier_a.append((child_node['weight'], child_node))
                                frontier_a_dict[child] = child_node
                            elif node[0] < child_node['weight']: 
                                pass
                    if child not in frontier_a_dict.keys(): 
                        frontier_a.append((child_node['weight'], child_node))
                        frontier_a_dict[child] = child_node                       

        # Thread B 
        if frontier_b.size() != 0: 
            b = frontier_b.pop()[1] 
            b_state = b['state']
            frontier_b_dict.pop(b_state)
            explored_b[b_state] = {'path'    : b['path'], 
                                   'weight'  : b['weight'],
                                   'state'   : b_state}
            
            check_a_dict = {**explored_a, **frontier_a_dict} 
            check_c_dict = {**explored_c, **frontier_c_dict}
            if b_state in check_a_dict.keys() and b['weight'] + check_a_dict[b_state]['weight']< mu_ab:
                mu_ab = b['weight'] + check_a_dict[b_state]['weight'] 
                path_ab = check_a_dict[b_state]['path'][:-1] + b['path'][::-1]
            
            if b_state in check_c_dict.keys() and b['weight'] + check_c_dict[b_state]['weight']< mu_bc:
                mu_bc = b['weight'] + check_c_dict[b_state]['weight'] 
                path_bc = b['path'] + check_c_dict[b_state]['path'][::-1][1:]

            for child in graph[b_state]: 
                child_node = empty_node.copy() 
                child_node['state']         = child
                child_node['path']          = [*b['path'], child]
                child_node['weight']        = b['weight'] + \
                                            graph.get_edge_weight(b['state'], child) 
                if child not in explored_b.keys() or explored_b[child]['weight'] > child_node['weight']: 
                    for node in frontier_b: 
                        if child == node[2]['state']: 
                            if node[0] > child_node['weight']: 
                                frontier_b.queue.remove(node)
                                frontier_b.append((child_node['weight'], child_node))
                                frontier_b_dict[child] = child_node
                            elif node[0] < child_node['weight']: 
                                pass
                    if child not in frontier_b_dict.keys(): 
                        frontier_b.append((child_node['weight'], child_node))
                        frontier_b_dict[child] = child_node  

        # Thread C 
        if frontier_c.size() != 0: 
            c = frontier_c.pop()[1] 
            c_state = c['state']
            frontier_c_dict.pop(c_state)
            explored_c[c_state] = {'path'    : c['path'], 
                                   'weight'  : c['weight'],
                                   'state'   : c_state}
            
            check_a_dict = {**explored_a, **frontier_a_dict} 
            check_b_dict = {**explored_b, **frontier_b_dict}
            if c_state in check_a_dict.keys() and c['weight'] + check_a_dict[c_state]['weight']< mu_ca:
                mu_ca = c['weight'] + check_a_dict[c_state]['weight'] 
                path_ac = check_a_dict[c_state]['path'][:-1] + c['path'][::-1]
            
            if c_state in check_b_dict.keys() and c['weight'] + check_b_dict[c_state]['weight']< mu_bc:
                mu_bc = c['weight'] + check_b_dict[c_state]['weight'] 
                path_bc = check_b_dict[c_state]['path'][:-1] + c['path'][::-1]

            for child in graph[c_state]: 
                child_node = empty_node.copy() 
                child_node['state']         = child
                child_node['path']          = [*c['path'], child]
                child_node['weight']        = c['weight'] + \
                                            graph.get_edge_weight(c['state'], child) 
                if child not in explored_c.keys() or explored_c[child]['weight'] > child_node['weight']: 
                    for node in frontier_c: 
                        if child == node[2]['state']: 
                            if node[0] > child_node['weight']: 
                                frontier_c.queue.remove(node)
                                frontier_c.append((child_node['weight'], child_node))
                                frontier_c_dict[child] = child_node
                            elif node[0] < child_node['weight']: 
                                pass
                    if child not in frontier_c_dict.keys(): 
                        frontier_c.append((child_node['weight'], child_node))
                        frontier_c_dict[child] = child_node  

        if frontier_a.top()[0] + frontier_b.top()[0] >= mu_ab: 
            solution_a = True 
        if frontier_a.top()[0] + frontier_c.top()[0] >= mu_ca:
            solution_b = True 
        if frontier_c.top()[0] + frontier_b.top()[0] >= mu_bc:
            solution_c = True 
        if all([solution_c, solution_b, solution_a]): 
            if mu_bc >= mu_ab and mu_bc >= mu_ca: 
                final_path = path_ab[::-1] + path_ac[1:] 
            elif mu_ab >= mu_ca and mu_ab >= mu_bc:
                final_path = path_ac + path_bc[::-1][1:]
            elif mu_ca >= mu_ab and mu_ca >= mu_bc: 
                final_path = path_ab + path_bc[1:]
            return final_path
            

def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # Thread A: first -> second 
    # Thread B: second -> third 
    # Thread C: third -> first

    # Initialize Containers
    final_path = []
    first, second, third = goals[:] 

    empty_node = {"state"   : "", 
                  "path"    : [],
                  "g_cost"  : 0,    # Path_cost 
                  "h_cost"  : 0,    # Heuristic 
                  "f_cost"  : 0}
    node_a, node_b, node_c = empty_node.copy(), empty_node.copy(), empty_node.copy()  
    node_a["state"], node_b["state"], node_c["state"] = first, second, third 
    node_a["path"], node_b["path"], node_c["path"] = [first], [second], [third] 
    ab_dist = heuristic(graph, node_a['state'], second) 
    bc_dist = heuristic(graph, node_b['state'], third)
    ca_dist = heuristic(graph, node_c['state'], first) 
    node_a['h_cost'] = ab_dist
    node_b['h_cost'] = bc_dist  
    node_c['h_cost'] = ca_dist 
    node_a['f_cost'] = node_a['h_cost'] + node_a['g_cost']   
    node_b['f_cost'] = node_b['h_cost'] + node_b['g_cost']   
    node_c['f_cost'] = node_c['h_cost'] + node_c['g_cost']   

    explored_a = {first: node_a} 
    explored_b = {second: node_b} 
    explored_c = {third: node_c} 
    frontier_a_dict = {first: node_a}
    frontier_b_dict = {second: node_b} 
    frontier_c_dict = {third: node_c} 
    frontier_a, frontier_b, frontier_c = PriorityQueue(), PriorityQueue(), PriorityQueue() 
    frontier_a.append((0, node_a)) 
    frontier_b.append((0, node_b))
    frontier_c.append((0,node_c)) 

    mu_ab = float('inf')
    mu_bc = float('inf')
    mu_ca = float('inf')  
    
    p_cost_ab = float('inf')
    p_cost_bc = float('inf')
    p_cost_ca = float('inf')

    solution_a = False
    solution_b = False
    solution_c = False

    run_thread_a, run_thread_b, run_thread_c = True, True, True

    if first == second == third: 
        return final_path 

    # Begin main iteration
    while not all([solution_a, solution_b, solution_c]): 
        
        # Thread A 
        if run_thread_a: 
            if frontier_a.size() != 0: 
                a = frontier_a.pop()[1] 
                a_state = a['state']
                frontier_a_dict.pop(a_state)
                explored_a[a_state] = {'path'    : a['path'],
                                    'g_cost'  : a['g_cost'], 
                                    'h_cost'  : a['h_cost'],
                                    'f_cost'  : a['f_cost'],
                                    'state'   : a_state}
                
                check_b_dict = {**explored_b, **frontier_b_dict} 
                check_c_dict = {**explored_c, **frontier_c_dict}
                if a_state in check_b_dict.keys() and a['f_cost'] + check_b_dict[a_state]['f_cost']< mu_ab:
                    mu_ab = a['f_cost'] + check_b_dict[a_state]['f_cost'] 
                    path_ab = a['path'] + check_b_dict[a_state]['path'][::-1][1:]
                    p_cost_ab = a['g_cost'] + check_b_dict[a_state]['g_cost']
                
                if a_state in check_c_dict.keys() and a['f_cost'] + check_c_dict[a_state]['f_cost']< mu_ca:
                    mu_ca = a['f_cost'] + check_c_dict[a_state]['f_cost'] 
                    path_ac = a['path'] + check_c_dict[a_state]['path'][::-1][1:]
                    p_cost_ca = a['g_cost'] + check_c_dict[a_state]['g_cost']

                for child in graph[a_state]: 
                    child_node = empty_node.copy() 
                    child_node['state']         = child
                    child_node['path']          = [*a['path'], child]
                    child_node['g_cost']        = a['g_cost'] + \
                                                graph.get_edge_weight(a['state'], child) 
                    child_node['h_cost']        = heuristic(graph, child, second)
                    child_node['f_cost']        = child_node['h_cost'] + child_node['g_cost']

                    if child not in explored_a.keys() or explored_a[child]['f_cost'] > child_node['f_cost']: 
                        for node in frontier_a: 
                            if child == node[2]['state']: 
                                if node[0] > child_node['f_cost']: 
                                    frontier_a.queue.remove(node)
                                    frontier_a.append((child_node['f_cost'], child_node))
                                    frontier_a_dict[child] = child_node
                                elif node[0] < child_node['f_cost']: 
                                    pass
                        if child not in frontier_a_dict.keys(): 
                            frontier_a.append((child_node['f_cost'], child_node))
                            frontier_a_dict[child] = child_node          

        # Thread B 
        if run_thread_b:
            if frontier_b.size() != 0: 
                b = frontier_b.pop()[1] 
                b_state = b['state']
                frontier_b_dict.pop(b_state)
                explored_b[b_state] = {'path'    : b['path'],
                                    'g_cost'  : b['g_cost'], 
                                    'h_cost'  : b['h_cost'],
                                    'f_cost'  : b['f_cost'],
                                    'state'   : b_state}
                
                check_a_dict = {**explored_a, **frontier_a_dict} 
                check_c_dict = {**explored_c, **frontier_c_dict}
                if b_state in check_a_dict.keys() and b['f_cost'] + check_a_dict[b_state]['f_cost']< mu_ab:
                    mu_ab = b['f_cost'] + check_a_dict[b_state]['f_cost'] 
                    path_ab = check_a_dict[b_state]['path'][:-1] + b['path'][::-1]
                    p_cost_ab = b['g_cost'] + check_a_dict[b_state]['g_cost']
                
                if b_state in check_c_dict.keys() and b['f_cost'] + check_c_dict[b_state]['f_cost']< mu_bc:
                    mu_bc = b['f_cost'] + check_c_dict[b_state]['f_cost'] 
                    path_bc = b['path'] + check_c_dict[b_state]['path'][::-1][1:]
                    p_cost_bc = b['g_cost'] + check_c_dict[b_state]['g_cost']

                for child in graph[b_state]: 
                    child_node = empty_node.copy() 
                    child_node['state']         = child
                    child_node['path']          = [*b['path'], child]
                    child_node['g_cost']        = b['g_cost'] + \
                                                graph.get_edge_weight(b['state'], child) 
                    child_node['h_cost']        = heuristic(graph, child, third)
                    child_node['f_cost']        = child_node['h_cost'] + child_node['g_cost']

                    if child not in explored_b.keys() or explored_b[child]['f_cost'] > child_node['f_cost']: 
                        for node in frontier_b: 
                            if child == node[2]['state']: 
                                if node[0] > child_node['f_cost']: 
                                    frontier_b.queue.remove(node)
                                    frontier_b.append((child_node['f_cost'], child_node))
                                    frontier_b_dict[child] = child_node
                                elif node[0] < child_node['f_cost']: 
                                    pass
                        if child not in frontier_b_dict.keys(): 
                            frontier_b.append((child_node['f_cost'], child_node))
                            frontier_b_dict[child] = child_node 
        # Thread C 
        if run_thread_c: 
            if frontier_c.size() != 0: 
                c = frontier_c.pop()[1] 
                c_state = c['state']
                frontier_c_dict.pop(c_state)
                explored_c[c_state] = {'path'    : c['path'],
                                    'g_cost'     : c['g_cost'], 
                                    'h_cost'     : c['h_cost'],
                                    'f_cost'     : c['f_cost'],
                                    'state'      : c_state}
                
                check_a_dict = {**explored_a, **frontier_a_dict} 
                check_b_dict = {**explored_b, **frontier_b_dict}
                if c_state in check_a_dict.keys() and c['f_cost'] + check_a_dict[c_state]['f_cost']< mu_ca:
                    mu_ca = c['f_cost'] + check_a_dict[c_state]['f_cost'] 
                    path_ac = check_a_dict[c_state]['path'][:-1] + c['path'][::-1]
                    p_cost_ca = c['g_cost'] + check_a_dict[c_state]['g_cost']
                
                if c_state in check_b_dict.keys() and c['f_cost'] + check_b_dict[c_state]['f_cost']< mu_bc:
                    mu_bc = c['f_cost'] + check_b_dict[c_state]['f_cost'] 
                    path_bc = check_b_dict[c_state]['path'][:-1] + c['path'][::-1]
                    p_cost_bc = c['g_cost'] + check_b_dict[c_state]['g_cost']

                for child in graph[c_state]: 
                    child_node = empty_node.copy() 
                    child_node['state']         = child
                    child_node['path']          = [*c['path'], child]
                    child_node['g_cost']        = c['g_cost'] + \
                                                  graph.get_edge_weight(c['state'], child) 
                    child_node['h_cost']        = heuristic(graph, child, first)
                    child_node['f_cost']        = child_node['h_cost'] + child_node['g_cost']

                    if child not in explored_c.keys() or explored_c[child]['f_cost'] > child_node['f_cost']: 
                        for node in frontier_c: 
                            if child == node[2]['state']: 
                                if node[0] > child_node['f_cost']: 
                                    frontier_c.queue.remove(node)
                                    frontier_c.append((child_node['f_cost'], child_node))
                                    frontier_c_dict[child] = child_node
                                elif node[0] < child_node['f_cost']: 
                                    pass
                        if child not in frontier_c_dict.keys(): 
                            frontier_c.append((child_node['f_cost'], child_node))
                            frontier_c_dict[child] = child_node            
        
        if frontier_a.top()[0] + frontier_b.top()[0] >= mu_ab: 
            solution_a = True 
        if frontier_a.top()[0] + frontier_c.top()[0] >= mu_ca:
            solution_b = True 
        if frontier_c.top()[0] + frontier_b.top()[0] >= mu_bc:
            solution_c = True 

        # if frontier_a.top()[2]['g_cost'] + frontier_b.top()[2]['g_cost'] >= p_cost_ab: 
        #     solution_a = True 
        # if frontier_a.top()[2]['g_cost'] + frontier_c.top()[2]['g_cost'] >= p_cost_ca:
        #     solution_b = True 
        # if frontier_c.top()[2]['g_cost'] + frontier_b.top()[2]['g_cost'] >= p_cost_bc:
        #     solution_c = True       

        if all([solution_c, solution_b, solution_a]): 
            if p_cost_bc >= p_cost_ab and p_cost_bc >= p_cost_ca: 
                final_path = path_ab[::-1] + path_ac[1:] 
            elif p_cost_ab >= p_cost_ca and p_cost_ab >= p_cost_bc:
                final_path = path_ac + path_bc[::-1][1:]
            elif p_cost_ca >= p_cost_ab and p_cost_ca >= p_cost_bc: 
                final_path = path_ab + path_bc[1:]
            return final_path


def compute_landmarks(graph):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


def custom_search(graph, start, goal, data=None):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
