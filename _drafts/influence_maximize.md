---
layout: post
title: "How Social Networks Work: The Influence Maximization Problem"
author: David Mao
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Today, social networks, such as Facebook, have a profound effect on influencing the way people think. For advertisers and politicians, the question becomes how do you influence the opinions of people using the least amount of resources? The solution to this is an optimization problem known as the Influence Maximization Problem.

## Social Networks as a Graph

Social Networks can be represented as an graph {% raw %} $$G(V,E)$$  {% endraw %}, where {% raw %} $$V$$  {% endraw %} is a set of vertices and {% raw %} $$E$$  {% endraw %} is a set of edges connecting each vertex. For instance, take the example of Facebook. Every user on Facebook would be denoted by a vertex in the graph. If two users are friends, then their vertex are connected by an edge. Additionally, a weight is assigned to each edge that denotes the amount of influence a friendship holds.

## The Influence Maximization Problem

The Influence Maximization Problem can be given as such: given a graph {% raw %} $$G(V,E)$$  {% endraw %} denoting a social network, and a positive integer {% raw %} $$K$$  {% endraw %} denoting our budget, or the maximum number of users we can influence.

## Worked Example