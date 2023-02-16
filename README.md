# Dijkstra-Algorithm
Performing a Dijkstra Algorithm from one point to another point with a condition to pass through a certain point.

For example:

<img width="480" alt="Screenshot 2023-02-16 at 6 14 37 PM" src="https://user-images.githubusercontent.com/103298139/219350281-4fda8b15-e5fc-4906-8154-296c321af32a.png">
In the graph above, the vertex represent places and the one with underscores are cafes. The edges represents the time needed to travel the path. Each cafe has a waiting time and is represented in the second item in the tuple. The input format is [(vertex, vertex, int), (vertex, vertex, int), ...] and [(int, int), (int, int), ...].For example:

```
roads = [(0, 1, 4), (1, 2, 2), (2, 3, 3), (3, 4, 1), (1, 5, 2),
(5, 6, 5), (6, 3, 2), (6, 4, 3), (1, 7, 4), (7, 8, 2),
(8, 7, 2), (7, 3, 2), (8, 0, 11), (4, 3, 1), (4, 8, 10)]

# The cafes represented as a list of tuple
cafes = [(5, 10), (6, 1), (7, 5), (0, 3), (8, 4)]
```

To find the fastest path from one vertex to another vertex with a condition to pass through a cafe plus the waiting time, we can do:

```
mygraph = RoadGraph(roads, cafes)
start = 1
end = 3
mygraph.routing(start, end)

# result
[1,5,6,3]
```
Therefore, the shortest path from 1 to 3 while passing through a cafe is from 1, to 5 then 6 and finally 3.
