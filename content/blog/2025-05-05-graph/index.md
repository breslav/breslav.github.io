---
date: "2025-05-05T00:00:00Z"
tags:
title: Graph Traversals (Briefly)
---

When it comes to graphs, traversal algorithms like BFS and DFS are among the most important algorithms to know and work with. However, the implementation of BFS and DFS can be a bit nuanced with respect to the type of graph we are working with. In this blog post I want to briefly summarize some of these nuances.

## Introduction 

In general we can have undirected and directed graphs, which consist of one or more connected components (**CC**), which can be acyclic or cyclic. The table below lists the different types of graphs that I want to look at in this post.

| Undirected | Directed | Cyclic | Acyclic | Single CC | Rooted | Graph Type Name |
|:----------:|:--------:|:------:|:-------:|:------------------:|:-------:|:---------|
| x |  |x |   | | |Undirected Cyclic Graph | 
| x |  | |x   | | |Undirected Acyclic Graph | 
|  | x |x |   | | |Directed Cyclic Graph | 
|  | x | | x  | | |Directed Acyclic Graph | 
| x |  |  | x | x |x|Rooted Tree | 

## Graphs 
In general when we wish to traverse a graph we have to worry about visiting nodes that we have already visited and getting stuck in cycles. The table below summarizes the concerns for each type of graph. 

| Graph Type Name | Concern |
|:----------|:--------------|
| Undirected Cyclic Graph | Revisit parent or get stuck in cycle | 
| Undirected Acyclic Graph | Revisit parent| 
| Directed Cyclic Graph | Revisit other node or get stuck in cycle | 
| Directed Acyclic Graph | Revisit other node | 
| Rooted Tree | Revisit parent | 

Let's look at how these concerns can arise in a bit more detail:
- If we have an undirected graph and we are performing DFS we can end up revisiting where we immediately came from (which I'll refer to as "parent" for convenience) since edges are bi-directional. If there is a cycle we then have to worry about entering the cycle and not being able to get out.
- If we have a directed graph and we are performing DFS we can end up revisiting a node that was visited earlier on in the DFS (e.g by traversing another part of the graph that also leads to this node). If the directed graph has a cycle we once again have to worry about getting stuck in it.
- If we have a rooted tree it will depend on how the tree is implemented (see [trees](#trees) below) as to whether there is a concern of revisiting the parent.

Note: 
- With BFS we would have the same concerns as DFS.
- If we have an undirected graph and we revisit some node that is not the parent then that means we have a cycle.

### Solution
To avoid these problems graph traversal algorithms maintain a **visited** set which keeps track of what nodes have already been visited in the traversal. This way, when we consider recursing to explore a neighbor or adding it to a queue (for future exploration), we fist check whether it was already visited. This prevents us from revisiting nodes and getting stuck in cycles!

## Cycle Detection Nuances
In general we may not know whether a graph has a cycle or not, which means we may want to detect whether it does. The way cycle detection is done varies depending on whether the graph is undirected or directed.

### Undirected
The general strategy for detecting a cycle in an undirected graph it to perform a DFS where we check whether the node we are going to explore is a node we have already visited. However, since the graph is undirected, when we perform DFS we don't want to consider the edge pointing back to where we came from as an indicator that we have found a cycle. To avoid this "false positive detection" we can pass an additional parameter to the DFS which represents the "parent" or "from" node. Then when looping through neighbors of a node we can ignore the neighbor that we recursed from. If we still encounter a node that was already visited then this means there is a cycle. 

### Directed
For directed graphs there are two strategies for detecting a cycle that are different than the undirected case. We still have the intuition that we want to avoid visiting a node that we have already visited in the past. Edges are directional so we don't have to worry about directly going back to where we came from during DFS. However, as previously mentioned, it is possible for us to visit a node that was already visited during DFS from another part of the graph. 

For example consider paths a->b->c->e and b->d->c, and suppose DFS first visits a,b,c,e and then backtracks and visits d,c. In this example c is revisited but there is no cycle. To avoid visiting c (and subsequently e) a second time we want to maintain a visited set. However, we don't want to incorrectly classify c as representing a cycle in the graph. Therefore, we must maintain our *current* DFS path and a cycle is detected if we visit a node already in our *current* path. To maintain the current DFS path we would append the current node to a list when we enter DFS and pop it from the list when we finished exploration from the node, just prior to returning. 

Another strategy that is available to us for detecting cycles in directed graphs is topological sort. If you perform topological sort on a directed graph with a cycle then the sort will not be able to order nodes that are part of the cycle. As a result, we can use the length of the ordering as an indicator for whether there is a cycle. If the length is less than the number of nodes in the component then there is a cycle. 

To understand why cycles result in an incomplete ordering we need to recall that topological sort continues as long as we have nodes with indegree 0. When a node's dependencies are met the nodes indegree becomes 0. However, in a cycle by definition no node can have its dependencies met because the dependencies eventually include itself. Therefore nodes from the cycle will never be added to the ordering and thus the ordering will be incomplete.

Finally, it's worth noting that topological sort does not apply to undirected graphs since there is no notion of uni-directional dependencies.


## Trees
Finally, I want to talk about a point of confusion I had which in part led to this post. I recalled that when I was implementing BFS and DFS on trees I didn't have to bother keeping track of which nodes were already visited. Trees by definition don't have cycles so we wouldn't need to be concerned with getting stuck in one. However, trees are in general undirected which in theory means we would have to worry about revisiting a node that we just came from. So why wasn't this the case? 

Well it has to do with how (rooted) trees are often implemented. When they are defined as a collection of nodes, each of which can only point to child nodes, then that naturally influences how a traversal will progress (this implementation choice makes a tree "feel" like it is directed). Specifically DFS or BFS can only explore nodes down the tree (and never up) and as a result there is no way to visit an already visited node.

If on the other hand, our rooted tree was implemented with an adjacency list (bi-directional edges) then we would have to worry about visiting already visited nodes (like where we came from during DFS) and thus we would need to maintain a visited set. 

## Conclusion
BFS and DFS are two fundamental algorithms that are commonly applied to trees and graphs. In the general case we need to maintain a visited set in order to avoid pitfalls like revisiting already processed nodes and getting stuck in a cycle. Sometimes trees allow for simpler incarnations of BFS and DFS that don't require a visited set. Lastly, when detecting a cycle the approach varies between undirected and directed graphs. 