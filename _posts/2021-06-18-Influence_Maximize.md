---
layout: post
title: "How Social Networks Work: The Influence Maximization Problem"
author: David Mao
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Today, social networks, such as Facebook, have a profound effect on influencing the way people think. For advertisers and politicians, the question becomes how do you influence the opinions of people using the least amount of resources? The solution to this is an optimization problem known as the Influence Maximization Problem.

## Social Networks as a Graph

Social Networks can be represented as an graph {% raw %} $$G(V,E)$$  {% endraw %}, where {% raw %} $$V$$  {% endraw %} is a set of vertices and {% raw %} $$E$$  {% endraw %} is a set of edges connecting each vertex. For instance, take the example of Facebook. Every user on Facebook would be denoted by a vertex in the graph. If Mary and Sue are friends, then their verticies, denoting their Facebook accounts, are connected by an edge. Additionally, a weight {% raw %} $$W$$  {% endraw %} is assigned to each edge that denotes the amount of influence a friendship holds. For instance, if Mary is highly likely to be influenced by Sue, then the vertex from Sue to Mary would have a high weight.

## The Influence Maximization Problem

To formulate this problem, let us first simplify it a bit. Let us assume that the weights for all our edges is {% raw %} $$1$$  {% endraw %}, meaning that if Sue is influenced, then all her friends become influenced. Then for a set of nodes {% raw %} $$S \in V$$  {% endraw %}, denote {% raw %} $$N(S)$$  {% endraw %} to be the set of all neighbors of {% raw %} $$S$$  {% endraw %}. The influence of {% raw %} $$S$$  {% endraw %}, denoted {% raw %} $$I(S)$$  {% endraw %}, will then be the size of {% raw %} $$N(S)$$  {% endraw %}. The Influence Maximization Problem can be given as such: given a graph {% raw %} $$G(V,E)$$  {% endraw %} denoting a social network, and a positive integer {% raw %} $$K$$  {% endraw %} denoting our budget, or the maximum number of users we can influence, we want to choose {% raw %} $$S$$  {% endraw %} of size {% raw %} $$K$$  {% endraw %} such that {% raw %} $$I(S)$$  {% endraw %} is maximized.

## Worked Example

For this section, I have simulated a social media network graph in graph.txt. Each row denotes a edge in graph, which as previously mentioned corresponds to a being friends. The simulated network is constructed, then plotted below.

```python
import networkx as nx
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
```


```python
df = pd.read_csv('graph.txt', delimiter=" ", header=0)

G = nx.Graph()
V = pd.unique(df.iloc[:,0:2].values.ravel('K'))
for v in V:
    G.add_node(v)
    
for index, row in df.iterrows():
    G.add_edge(row['0'], row['1'])
```


```python
fig, ax = plt.subplots(figsize=(16,16))

node_color = np.repeat('green', len(V))
edge_color = np.repeat('blue', len(df))
width = np.repeat(0.01, len(df))

nx.draw(G, node_size=5, node_color=node_color, edge_color=edge_color, width=width, ax=ax)
```


    
![png](/assets/img/blog2/output_2_0.png)
    
As we can see, there are large groups of vertexes heavily connected with each other, which se can see as communities or friend groups. Lets observe what happens when {% raw %} $$K=1$$  {% endraw %}. Intuitively, we should choose the vertex that has the most connections, i.e the user who has the most friends.

```python
unique, counts = np.unique(df.iloc[:,0:2].values.ravel('K'), return_counts=True)
target = np.argmax(counts)

S = []
NS = []
S.append(target)

for index, row in df.iterrows():
    if row['0']==target:
        S.append(row['1'])
        NS.append(index)
    elif row['1']==target:
        S.append(row['0'])
        NS.append(index)
```


```python
node_color = []

for node in G:
    if node in S:
        node_color.append('Red')
    else:
        node_color.append('Green')
        
edge_color[NS]='red'
width[NS]=1

fig, ax = plt.subplots(figsize=(16,16))

nx.draw(G, node_size=5, node_color=node_color, edge_color=edge_color, width=width, ax=ax)
```


    
![png](/assets/img/blog2/output_4_0.png)

The node chosen seems to be an influencial member of the largest cluster. The question now becomes what happens when we want to maximize influence in our network for choosing more than one vertex.

## An Iterative Approximation Algorithm

It turns out the Influence Maximization Problem, even when all the edge weights are simplified to {% raw %} $$1$$  {% endraw %}, is NP-Hard, meaning that in practice, we need to find another way to approximate {% raw %} $$S$$  {% endraw %}. We can do so by greedily grabbing from the uninfluenced vertices the vertex that maximizes the influence. We do so by taking a vertex {% raw %} $$v \in V$$  {% endraw %} that maximizes the influence {% raw %} $$I(S)$$  {% endraw %}. Then, for the remaining vertices that have not been influence, we choose the next vertex that maximizes the influence in the remaining vertices. The pseudocode for this algorithm is given by:

{% raw %} $$S = \emptyset$$ {% endraw %} <br/>
for {% raw %} $$i=1:K$$ {% endraw %} <br/>
&nbsp;&nbsp;&nbsp;&nbsp;take {% raw %} $$v$$ {% endraw %} that maximizes {% raw %} $$I(S \cup v)-I(S)$$ {% endraw %} <br/>
&nbsp;&nbsp;&nbsp;&nbsp;{% raw %} $$S = S \cup v$$ {% endraw %} <br/>
end <br/>

This influence found by this algorithm is gaurunteed to be:

{% raw %} $$I(S)=[1-(1-\frac{1}{K})^{i+1}]I(Optimal)$$ {% endraw %}

where {% raw %} $$I(Optimal)$$ {% endraw %} is the optimal solution. We can see in the case that $$K=1$$ {% endraw %}, our solution is gaurunteed optimal, as we oberved from the earlier section.

## Worked Example Continued

We can take the pseudocode and turn it into a function in Python. 

```python
def influence_max(K, df):
    nodes = []
    vertex = []
    
    for i in range(K):
        counts = np.bincount(df.iloc[:,0:2].values.ravel('K'))
        target = np.argmax(counts)
        S = []
        NS = []
        S.append(target)
        for index, row in df.iterrows():
            if row['0']==target:
                S.append(row['1'])
                NS.append(index)
            elif row['1']==target:
                S.append(row['0'])
                NS.append(index)
                
        df = df[~df['0'].isin(S)]
        df = df[~df['1'].isin(S)]
        
        nodes = nodes + S
        vertex = vertex + NS
        
    return (nodes, vertex)
```


Returning to our previous example, we can set our budget to be {% raw %} $$K=5$$  {% endraw %}, and see what nodes become activated.

```python
S, NS = influence_max(5, df)
```


```python
node_color = []

for node in G:
    if node in S:
        node_color.append('Red')
    else:
        node_color.append('Green')
        
edge_color[NS]='red'
width[NS]=1

fig, ax = plt.subplots(figsize=(16,16))

nx.draw(G, node_size=5, node_color=node_color, edge_color=edge_color, width=width, ax=ax)
```


    
![png](/assets/img/blog2/output_8_0.png)

Five distinct activated sets are shown in our graph.