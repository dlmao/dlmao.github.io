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

## An Iterative Approximation Algorithm

It turns out the Influence Maximization Problem, even when all the edge weights are simplified to {% raw %} $$1$$  {% endraw %}, is NP-Hard, meaning that in practice, we need to find another way to approximate {% raw %} $$S$$  {% endraw %}. We can do so by taking a vertex {% raw %} $$v \in V$$  {% endraw %} that maximizes the influence {% raw %} $$I(S)$$  {% endraw %}. Then, for the remaining vertices that have not been influence, we choose the next vertex that maximizes the influence in the remaining vertices. The pseudocode for this algorithm is given by:

{% raw %} 
	S = \emptyset
{% endraw %}


## Worked Example

For this section, I have simulated a social media network graph in graph.txt. Each row denotes a edge in graph, which as previously mentioned corresponds to a being friends. 