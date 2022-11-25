"""
FIT2004     : Algorithms and Data Structures Assignment 2
Name        : Frederick Nathanael Thunardi
Student Id  : 32687885
Email       : fthu0001@student.monash.edu
Last edited : 16/09/22
"""

import math

# Question 1

class RoadGraph:
    def __init__(self, roads, cafes) -> None:
        """
        Finds the number of vertices in the graph and initialize the graph by
        creating vertices and assign the edges to each vertex. There will be 2 graphs
        created, the original graph, and the reversed graph which is the original graph
        with reversed direction of the edges. Also, sets the waiting time on each cafe.
        This graph uses adjacency list.

        :Input:
        roads: List of tuples containing the roads of the graph in a 
               form of (starting location ID, ending location ID, time taken to travel)
        cafes: List of tuples containing the waiting time to order coffee in a cafe in 
               a form of (cafe ID, waiting time)

        :Output, return or postcondition: Instance variables such us self.locations which is a list 
                                          containing the original graph, self.reverse which is a list
                                          containing the reversed version of the graph, and self.cafes 
                                          which is a list containing all the cafes in the graph.
        :Time complexity: O(|V| + |E|), where |V| is the set of unique locations in roads, and |E| is the
                          set of roads. 
        :Aux space complexity: O(|V| + |E|)
        """
        maximum_vertex = 0
        # Finding the maximum id of the vertices in roads to determine (|V| - 1)
        # O(|E|) time complexity
        for road in roads:
            if road[0] > maximum_vertex:
                maximum_vertex = road[0]
            if road[1] > maximum_vertex:
                maximum_vertex = road[1]
        
        # Initializing the vertices for the original graph and storing it in the instance variable self.locations
        # The length of self.locations is |V| with vertices ranging from 0 to (|V| - 1)
        # O(|V|) time complexity and aux space
        self.locations = list(Vertex(location) for location in range(maximum_vertex + 1))

        # Initializing the vertices for the reverse graph and storing it in the instance variable self.reverse
        # The length of self.reverse is |V| with vertices ranging from 0 to (|V| - 1)
        # O(|V|) time complexity and aux space
        self.reverse = list(Vertex(location2) for location2 in range(maximum_vertex + 1))

        # To initialize the edges to each vertex
        # O(|E|) time complexity and aux since there are |E| set of roads
        for edge in roads:
            # Adding the edges to the vertices in the original graph
            self.locations[edge[0]].add_edge(self.locations[edge[0]],self.locations[edge[1]], edge[2] )

            # Adding the edges to the vertices in the reverse graph (direction of the edges reversed)
            self.reverse[edge[1]].add_edge(self.reverse[edge[1]],self.reverse[edge[0]], edge[2])

        self.cafes = []
        # O(|V|) time complexity, since at worst, the size of cafes is |V|
        for cafe in cafes:
            # Adding the list of cafes to the self.cafes instance variable
            self.cafes.append(cafe[0])

            # Setting the waiting time for each cafe on both version of the graph
            self.locations[cafe[0]].time = cafe[1]
            self.reverse[cafe[0]].time = cafe[1]

    def routing(self, start, end):
        """
        Finds the optimal route from start to end, while also grabbing a coffee
        from one of the cafes. This function uses the Dijkstra algorithm and a reverse
        graph to make it work.

        :Input:
        start: An integer which determines the starting point/ vertex/ location.
        end: An integer which determines the end point/ vertex/ location

        :Output, return or postcondition: Returns a list of integers of the shortest
                                          route from start to end while stopping by a cafe 
                                          to order coffee. But, if there is no such possible
                                          route, it will return None
        :Time complexity: O(|E| log |V|), where V is the set of unique locations in roads, and
                          E is the set roads.
        :Aux space complexity: O(|V | + |E|), to store the vertices and edges in the minheap
        """
        # There's no cafe means there's no possible route from start to finish without
        # passing by a cafe
        if len(self.cafes) == 0:
            return None

        # O(|E| log |V|) time complexity
        # O(|V| + |E|) aux space complexity
        self.dijkstra(end, self.reverse)
        self.dijkstra(start, self.locations)

        # O(|V|) time complexity
        self.add_reverse_cafe()

        # O(|V|) time complexity
        self.add_waiting()

        # O(|V|) time complexity
        # Find the cafe with the most optimal distance and waiting time
        key = self.minimal_cafe()

        # If the most optimal cafe along the way has a distance of math.inf,
        # means there's no possible route
        if key.distance == math.inf:
            return None

        # Returns the route from start to cafe, and cafe to end through backtracking
        # O(|V|) time complexity
        return self.bactracking(self.locations[start], self.locations[key.id])[::-1] + self.bactracking(self.reverse[end], self.reverse[key.id])[1:]

    def dijkstra(self, start, version):
        """
        Uses Dijkstra algorithm to traverse through the graph with the
        purpose of finding the shortest distance of each vertex from
        the starting vertex.

        :Input:
        start: The starting point of the Dijkstra algorithm.
        version: The version of the graph the function will be traversing through, 
                 its either self.locations (original), or self.reversed (reversed) form
                 of the graph.

        :Output, return or postcondition: Sets the minimum distance for each vertex from 
                                          the starting vertex.
        :Time complexity: O(|E| log |V|), where |V| is the set of unique locations in roads, and |E| is the
                          set of roads. Originally, it is O(|E| log |V| + |V| log |V|), but it is dominated by
                          O(|E| log |V|), since E >= V and hence, |E| log |V| >= |V| log |V|.
        :Aux space complexity: O(|V| + |E|), dominated by the MinHeap which stores all the vertices and the edges.

        :Citation: [FIT2004_2022sem02] Lecture04 P1 Graph BFS DFS; Lecture05 P1 Dijkstra by Ian
        """
        # Creates a minheap containing all the vertices and edges
        # O(|V| + |E|) time and aux space complexity
        heap = MinHeap(version[start], version)

        # O(V) time complexity since there is V locations in the heap
        while len(heap) > 1:
            # Served will contain the vertex with the smallest distance that has not yet been 
            # served from the heap
            # O(V log V) time complexity because serve in a min heap is O(log V) and is
            # repeated V times
            served = heap.modified_serve()
            served.final()

            # O(E) time complexity since the for loop will run |E| times, which
            # is the number of roads
            for edge in served.edges:
                u = edge.u
                v = edge.v
                w = edge.w

                # If it has been finalized, then there's no changes needed
                if v.finalized == True:
                    pass

                # If the distanc of edge.v could be smaller, then it would be reduced
                elif v.distance > u.distance + w:
                    v.distance = u.distance + w
                    
                    # So that it could track the previous vertex for the backtracking function
                    v.track = served

                    # O(E log V) time complexity since it is O(log V) to rise in a minheap, and repeated E times.
                    heap.rise(v)
        return

    def bactracking(self, stop, start):
        """
        Uses backtracking to track the route from the start vertex all the way to the stop
        vertex by traversing the previous vertices of the start vertex.

        :Input:
        start: A vertex which is the end of a route. 
        stop: A vertex which is the start of the route, or a sign to stop the bactracking/ backtracking is reached.

        :Output, return or postcondition: Returns a list of integers of a route from stop to start. For example, 
                                          if start is a vertex with id 4, and stop is a vertex with id 1, it will return
                                          a list of integers containing the route from 1 to 4. Otherwise, if the route
                                          between start and stop doesn't exist, it will return None

        :Time complexity: O(|V|), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(|V|)
        """
        pathh = []
    
        pathh.append(start.id)

        current = start.track
        
        # O(V) time complexity since at worst case, the loop traverses all vertices
        while current != stop:
        
        # If there is no more previous path in current, return None
        # this means the there is no route between start and stop
            if current == None:
                return pathh

            # Keep traversing the previous vertex as long as it is not None
            pathh.append(current.id)
            current = current.track
        
        # O(V) aux space since at worst, the len(pathh) is |V|
        pathh.append(stop.id)
        return pathh

    def add_waiting(self):
        """
        Modify the distance of each cafe of the original graph 
        to include the waiting time for a coffee.

        :Input:
        There is no input needed but the instance variable self.cafes will be very handy.

        :Output, return or postcondition: Change the distance for every cafe available to include
                                          the waiting time to order a coffee.

        :Time complexity: O(|V|), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(1), since no extra space needed.
        """
        # O(|V|) time complexity since the loop traverses all the cafe, and
        # at the worst case, there are |V| cafes
        for location in self.cafes:
            self.locations[location].distance += self.locations[location].time

    def add_reverse_cafe(self):
        """
        Modify the distance of each cafe of the original graph
        to include the closest distance of the cafe itself from the end 
        of the routing function.

        :Input:
        There is no input needed but the instance variable self.cafes will be very handy.

        :Output, return or postcondition: Change the distance for every cafe available to include
                                          the closest distance of the cafe itself from the end 
                                          of the routing function. For example, if the start variable
                                          in self.routing is 1 and end variable is 4, the add_reverse_cafe function
                                          will modify each distance of the cafe to include the distance from the cafe itself
                                          to 4.

        :Time complexity: O(|V|), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(1), since no extra space needed.
        """
        # O(|V|) time complexity since the loop traverses all the cafe, and
        # at the worst case, there are |V| cafes
        for cafes in self.cafes:
            self.locations[cafes].distance += self.reverse[cafes].distance

    def minimal_cafe(self):
        """
        Find and return a cafe with the minimum distance.

        :Input:
        There is no input needed but the instance variable self.cafes will be very handy.

        :Output, return or postcondition: Returns a cafe (vertex), with the minimal distance

        :Time complexity: O(|V|), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(1), since constant extra space needed.
        """
        # O(1) aux space
        minimal = self.locations[self.cafes[0]]

        # O(|V|) time complexity since the loop traverses all the cafe, and
        # at the worst case, there are |V| cafes
        for cafes in self.cafes:
            if self.locations[cafes].distance < minimal.distance:
                minimal = self.locations[cafes]
        return minimal

    def __str__(self):
        return_string = "ORIGINAL \n"
        for items in self.locations:
            return_string += "Vertex: " + str(items) +", Distance:" + str(items.distance) + ", Track: " + str(items.track) +  "\n"

        return_string += "RERVERSE \n"

        for reverse in self.reverse:
            return_string += "Vertex: " + str(reverse) +", Distance:" + str(reverse.distance) + ", Track: " + str(reverse.track) +  "\n"

        return return_string

class Vertex:
    def __init__(self, id, time = 0) -> None:
        """
        Initialize a vertex and its attributes.

        :Input:
        id: an integer which is then the vertex id
        time: an integer which is the waiting time, default 0 if it is not a cafe

        :Output, return or postcondition: Initialize a vertex

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space needed.
        """
        #O(1) time and aux since it's all initialization
        self.id = id
        self.edges = []
        self.time = time
        self.track = None
        self.distance = math.inf
        self.finalized = False

    def final(self):
        """
        If a vertex is finalized when doing the dijkstra algorithm, self.finalized is set
        to True and the distance can't be changed.

        :Input:

        :Output, return or postcondition: sets the vertex.finalized to True.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since no extra space needed
        """
        self.finalized = True

    def add_edge(self, u, v, w):
        """
        Assign a directed edge(road) to a vertex by appending the edges to the instance
        variable self.edges in each vertex.

        :Input:
        u: a vertex which is the origin of the edge
        v: a vertex which is the end of the edge, or where the road was going
        w: an integer which determines the distance from u to v.

        :Output, return or postcondition: sets the vertex.finalized to True.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        # O(1) time complexity since append is O(1)
        self.edges.append(Edge(u,v,w))

    def __str__(self):
        return_string = str(self.id)
        return return_string

class Edge:
    def __init__(self, u: Vertex, v: Vertex, w):
        """
        Initialize a directed edge with u being the starting point, 
        v the end point, and w be the weight.

        :Input:
        u: a vertex which is the origin of the edge/ road
        v: a vertex which is the end of the edge/ road, or where the road was going
        w: an integer which determines the distance from u to v.

        :Output, return or postcondition: Initializes a directed edge from u to v
                                          with weight of w.

        :Time complexity: O(1) since it's constant
        :Aux space complexity: O(1), since constant extra space is needed
        """
        self.u = u
        self.v = v
        self.w = w

    def __str__(self):
        return_string = "(" + str(self.u) + ", "+ str(self.v) + ", "+ str(self.w)+ ")"
        return return_string

class MinHeap:
    def __init__(self, source: Vertex, graph):
        """
        Initializes a minheap in a form of a list in the instance variable self.heap
        and a location list which determines the position of each vertex on the heap
        in the self.index instance variable. 

        :Input:
        source: A vertex which determines the start of the dijkstra algorithm
        graph: A list of vertex which represents the graph

        :Output, return or postcondition: A minheap with two instance variables.
                                          The first instance variable is self.heap
                                          which is the heap itself, and self.index. which is to 
                                          determine the position of each vertex in the heap.

        :Time complexity: O(|V| + |E|), where |V| is the set of unique locations in roads, and |E| is the
                          set of roads.
        :Aux space complexity: O(|V| + |E|), since constant extra space is needed
        """
        # Heap representation stored in self.heap
        # O(|V|) time and aux space complexity
        self.heap = [None] * (len(graph)  + 1)

        # Sets the starting vertex to 0 so that it could
        # be the first item in the heap to be served.
        source.distance = 0

        # # Stores the source vertex in the heap
        # self.heap[source.id + 1] = source

        # Stores each vertex in the minheap, leaving the first index to None
        # O(|V| + |E|) time and aux complexity since there are V vertex, and E total roads/ edges
        for i in range(len(graph)):
            self.heap[i + 1] = graph[i]

        # O(|V|) time and aux complexity since there are |V| vertex in the graph
        self.index = [None] * (len(graph))

        # O(|V|) time since there are |V| vertex in self.heap
        for y in range(1, len(self.heap)):
            self.index[self.heap[y].id] = y

        # Organizing the heap so that it could be a minheap
        for i in range(len(self.heap)//2, 0, -1):
                self.sink(self.heap[i])
    
    def head_node_index(self, vertex: Vertex):
        """
        Takes a vertex as an input to find and return the index of
        the head node of the vertex.

        :Input:
        vertex: Vertex whose head node index we want to find

        :Output, return or postcondition: Head node index

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        current_position = self.index[vertex.id]
        head = current_position // 2
        return head

    def left_node_vertex(self, vertex: Vertex) -> Vertex:
        """
        Takes a vertex as an input to find and return 
        the left node of the vertex.

        :Input:
        vertex: Vertex whose left node index we want to find

        :Output, return or postcondition: Left node vertex of the vertex parameter, or
                                          None if the left node vertex doesn't exist.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        current_position = self.index[vertex.id]
        left = current_position * 2

        # If the position of the left node vertex is bigger than len(self.heap),
        # this means that the left node index doesn't exist
        if left >= len(self.heap):
            return None
        return self.heap[left]

    def right_node_vertex(self, vertex: Vertex) -> Vertex:
        """
        Takes a vertex as an input to find and return 
        the right node of the vertex.

        :Input:
        vertex: Vertex whose right node index we want to find

        :Output, return or postcondition: Right node vertex of the vertex parameter, or
                                          None if the right node vertex doesn't exist.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        current_position = self.index[vertex.id]
        right = (current_position * 2) + 1

        # If the position of the right node vertex is bigger than len(self.heap),
        # this means that the right node index doesn't exist
        if right >= len(self.heap):
            return None
        return self.heap[right]

    def leaf_node(self, vertex: Vertex) -> bool:
        """
        To check if the vertex is a leaf node.
        
        :Input:
        vertex: The vertex to check if it's a leaf node

        :Output, return or postcondition: Returns a boolean indicating if the
                                          vertex is a leaf node or not determined
                                          by True or False.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        if vertex == None:
            return True
        current_position = self.heap[self.index[vertex.id]]

        # It can determine if it is a leaf node or not by
        # determining if there is no left and right node, then it is a leaf node.
        if self.left_node_vertex(current_position) == None and self.right_node_vertex(current_position) == None:
            return True
        return False

    def rise(self, Vert: Vertex):
        """
        To make the vertex perform a rise action in the minheap. It will keep rising untill
        the head node's distance is smaller, or there is no head node.
        
        :Input:
        vertex: The vertex that will be performed rise on

        :Output, return or postcondition: Turns the self.heap to be arranged
                                          as a min heap where a child node is always
                                          bigger or equal to the parent node.

        :Time complexity: O(log V), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(1), since constant extra space is needed
        """
        current_position = self.index[Vert.id]
        head = self.head_node_index(Vert)

        # O(log V) time complexity since at worst, it will traverse through log V vertices
        while head != 0 and self.heap[head].distance > self.heap[current_position].distance:

            #O(1) time and aux space
            self.swap(self.heap[head], self.heap[current_position])
            current_position = self.index[Vert.id]
            head = self.head_node_index(Vert)
    
    def swap(self, vert1: Vertex, vert2: Vertex):
        """
        Performs a swap between vertex 1 and vertex 2 on the
        self.heap and self.index instance variables
        
        :Input:
        vert1: A vertex which will be swapped with vert2
        vert2: A vertex which will be swapped with vert1

        :Output, return or postcondition: The position of vert1 and vert2
                                          will be swapped in the self.heap
                                          and also self.index.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since constant extra space is needed
        """
        position1 = self.index[vert1.id]
        position2 = self.index[vert2.id]

        self.heap[position1], self.heap[position2] = self.heap[position2], self.heap[position1]
        self.index[vert1.id], self.index[vert2.id] = self.index[vert2.id], self.index[vert1.id]

    def left_smaller(self, target: Vertex):
        """
        Compares the left and the right node of target,
        if left node's distance is smaller than the right, then it will return 
        True, and False otherwise.
        
        :Input:
        target: Vertex whose left and right node will be compared.

        :Output, return or postcondition: Returns True if the left node's distance is smaller
                                          than the right, also returns True when the right node
                                          doesn't exist. Returns False when the right node's distance
                                          is smaller than the left node.

        :Time complexity: O(1)
        :Aux space complexity: O(1), since no extra space is needed
        """
        if self.right_node_vertex(target) != None and self.left_node_vertex(target).distance > self.right_node_vertex(target).distance:
            return False
        return True

    def sink(self, target: Vertex):
        """
        Performs a sink action on the heap. The target will be performing
        a sink action until the head node's distance is smaller than the
        target's distance.
        
        :Input:
        target: Vertex who will be doing the sink action

        :Output, return or postcondition: Turns a minheap with target being
                                          in the right place, which is having a 
                                          head node's distance smaller than target's.

        :Time complexity: O(log V), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(1), since no extra space is needed
        """
        # O(log V) time complexity since at worst, the loop needs to traverse through log V vertices.
        # While the current node is not a leaf node since a sink cannot be performed on a leaf node.
        while not self.leaf_node(target):
            # Compare the left node's distance with the right, if the left is smaller and also smaller than target's distance,
            # then swap target with left node.
            if self.left_node_vertex(target) != None and target.distance > self.left_node_vertex(target).distance and self.left_smaller(target):
                self.swap(target, self.left_node_vertex(target))

            # Compare the right node's distance with the target distance, if it is smaller, performs a swap
            elif self.right_node_vertex(target) != None and target.distance > self.right_node_vertex(target).distance:
                self.swap(target, self.right_node_vertex(target))

            # Means no swap happened, this means that the left and right node's distance is both bigger
            # than the target's distance
            else:
                break

    def modified_serve(self):
        """
        Gets the vertex with the smallest distance on the minheap. It does this by 
        swapping the item on the second index with the last index. Then, the smallest 
        item is retrieved at the end of the self.heap (minheap representation). The item 
        in the last index who is swapped with the smallest item then will perform 
        a sink to get the minheap into proper position.
        
        :Output, return or postcondition: Returns a vertex with the smallest distance.

        :Time complexity: O(log V), where |V| is the set of unique locations in roads.
        :Aux space complexity: O(1), since no extra space is needed
        """

        # If len(self.heap) == 2, means that there is only 1 item left in the heap,
        # so we can jus tuse pop()
        if len(self.heap) == 2:
            return self.heap.pop()

        # Performs a swap between the vertex with the smallest distance and item with 
        # the last index
        self.swap(self.heap[1], self.heap[-1])
        
        # O(1) time complexity
        return_val = self.heap.pop()

        # The item in the last index who is swapped with the smallest item then will perform 
        # a sink to get the minheap into proper position.
        # O(log V) time complexity
        self.sink(self.heap[1])
        return return_val

    def __len__(self):
        """
        Returns the length of self.heap, which is the minheap representation
        
        :Output, return or postcondition: Returns an integer which is len(self.heap)

        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return len(self.heap)

# Question 2

class Graph:
    def __init__(self, roads) -> None:
        """
        Initialize a graph to be used in optimalRoute. The graph accepts a list of tuples
        which is essentially the edges. Firstly, it finds the maximum vertex id to make a
        graph representation of the roads by adding vertices from 0 up to maximum vertex id.
        This function also adds each edges to the corresponding vertex. This graph uses adjacency list.
        
        :Input:
        roads: List of tuples containing the edges to a graph in a form of (u, v, w),
               where u is the start, v is the end, and w is the weight of the edge.

        :Output, return or postcondition: Returns graph representation in a form
                                          of an instance variable self.route. Self.route
                                          is a list containing all the vertices with all the edges
                                          attached to it.

        :Time complexity: O(|P| + |D|), where |P| is the number of intersection points, and |D| is the number of downhill segments.
        :Aux space complexity: O(|P| + |D|)
        """
        maximum_vertex = 0

        # O(|D|) time complexity since need to loop through every edge/ downhills segments
        # to find the vertex with the maximum id
        for road in roads:
            if road[0] > maximum_vertex:
                maximum_vertex = road[0]
            if road[1] > maximum_vertex:
                maximum_vertex = road[1]
        # O(|P|) time and aux space needed to create |P| vertices
        self.route = list(Vertex(location) for location in range(maximum_vertex + 1))

        # O(|D|) time and aux space needed to append the edges to each vertex
        for edge in roads:
            self.route[edge[0]].add_edge(self.route[edge[0]],self.route[edge[1]], edge[2] )

    def reset(self):
        """
        Resets the distance of each vertex in the graph to -math.inf

        :Output, return or postcondition: Turns all the distance of the vertices in the graph
                                          into -math.inf.

        :Time complexity: O(|P|), where |P| is the number of intersection points.
        :Aux space complexity: O(1)
        """
        for vertex in self.route:
            vertex.distance = -math.inf
    
    def bactracking(self, stop, start):
        """
        Uses backtracking to track the route from the start vertex all the way to the stop
        vertex by traversing the previous vertices of the start vertex.

        :Input:
        start: A vertex which is the end of a route. 
        stop: A vertex which is the start of the route, or a sign to stop the bactracking/ backtracking is reached.

        :Output, return or postcondition: Returns a list of integers of a route from stop to start. For example, 
                                          if start is a vertex with id 4, and stop is a vertex with id 1, it will return
                                          a list of integers containing the route from 1 to 4. Otherwise, if the route
                                          between start and stop doesn't exist, it will return None

        :Time complexity: O(|P|), where |P| is the number of intersection points.
        :Aux space complexity: O(|P|)
        """
        pathh = []

        pathh.append(start.id)

        current = start.track

        # O(|P|) time complexity since at worst case, the loop traverses all vertices
        while current != stop:

        # If there is no more previous path in current, return None
        # this means the there is no route between start and stop
            if current == None:
                return pathh
    
            # Keep traversing the previous vertex as long as it is not None
            pathh.append(current.id)
            current = current.track
        
        # O(|P|) aux space since at worst, the len(pathh) is |P|
        pathh.append(stop.id)
        return pathh

def optimalRoute(downhillScores, start, finish):
    """
    Finds a route from the starting point to the finishing point while only using the downhill segments 
    in the downhillScores while also obtaining the maximum possible score. This function will turn the 
    segments into edges and intersection points into vertices, or in other word a graph. Uses dynamic programming
    to work from bottom up or finish to start to find the most optimal solution all the way up to start.

    :Input:
    downhillScores: List of tuples containing the edges to a graph in a form of (u, v, w),
                    where u is the start, v is the end, and w is the weight of the edge.
    start: An integer which is the starting point of the route
    finish: An integer which is the finishing point of the route

    :Output, return or postcondition: A list of integers containing the optimal route with
                                      the most possible score starting from the input start and
                                      ending with the input finish. Returns None if there is no possible
                                      route from start to finish

    :Time complexity: O(|D|), where |D| is the number of downhill segments. Originally,
                      the complexity is O(|P| + |D|), where |P| is the number of intersection points,
                      but since D >= P, so the complexity is dominated by O(|D|).
    :Aux space complexity: O(|D|), where originally it is O(|P| + |D|), dominated by O(|D|)
    """
    # O(|P| + |D|) time and aux space complexity
    snow_map = Graph(downhillScores)

    # O(|P|) time complexity
    snow_map.reset()

    # Set the starting point's distance to 0, which is finish since it is a
    # reversed graph, and we want to work from bottom to up. 
    snow_map.route[finish].distance = 0 # base case

    # O(|D|) time complexity at worst
    for roads in snow_map.route[start].edges:

        # branch optimal is the optimal score from each branch
        # the if statement helps to choose the biggest score among the edges
        # O(|D|) time complexity at worst
        branch_optimal = recursion(roads)
        if branch_optimal > snow_map.route[start].distance:
            snow_map.route[start].distance = branch_optimal
            snow_map.route[start].track = roads.v
    
    # To check if there is any possible route from start to finish
    # if not, return None
    if snow_map.route[start].distance == -math.inf:
        return None
    
    # Returns the backtracking route from start to finish
    return snow_map.bactracking(snow_map.route[finish], snow_map.route[start])

def recursion(edge: Edge):
    """
    A recursive function so that the dynamic programming on optimalRoute can
    be done top to bottom. This function will recursively loop until it either finds 
    the base case, which is the finish in optimalRoute, or a vertex that has no
    edges and is not the finish.

    :Input:
    edge: an Edge object

    :Output, return or postcondition: The most optimal score from Vertex u to Vertex v.

    :Time complexity: O(|D|), where |D| is the number of downhill segments. At worst, the function
                      will perform a recursive space |D| times, which means, it visits every
                      edge.
    :Aux space complexity: O(|D|), since recursion also needs extra space.
    """
    # O(1) time and aux space complexity
    # Base case of recursive
    if edge.v.distance != -math.inf and edge.u.distance < edge.v.distance + edge.w:
        edge.u.track = edge.v
        return edge.v.distance + edge.w

    # If the edge.v has any edges but it still has -math.inf as distance
    # Means there is still a chance to find the finish
    elif len(edge.v.edges) > 0:

        for x in edge.v.edges:
            
            # If it has more edges to travel, then call recursive 
            # to find the finish vertex
            branch = recursion(x)

            if branch > edge.v.distance:
                edge.v.distance = branch

            if edge.u.distance < edge.v.distance + edge.w:
                edge.u.distance = edge.v.distance + edge.w
                edge.u.track = edge.v
            
        return edge.v.distance + edge.w

    # O(1) time complexity
    # If the edge.v has no edges left and it is not the finish,
    # means that it is a dead end   
    else:
        return edge.v.distance + edge.w