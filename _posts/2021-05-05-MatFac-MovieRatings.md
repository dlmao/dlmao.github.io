---
layout: post
title: "Building a Simple Recommendation System with Matrix Factorization and Gradient Descent"
author: David Mao
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Building a Simple Recommendation System with Matrix Factorization and Gradient Descent

Suppose you watch a movie on Netflix that you really enjoy. After you finish the movie, you give the movie five stars. Immediately afterwards, Netflix reccommends you several other movies that you might also enjoy. What you have witnessed here is a reccomendation system in action. Recommendation systems are machine learning models that seeks to reccommend products to users based on past user interaction. It has applications in many places, mainly in advertisements, shopping websites, and movie applications.

## What Does Matrix Factorization Have to Do With This?

Let's go back to the Netflix Movie example. Suppose on Netflix's site, there are {% raw %} $$i$$  {% endraw %} movies and {% raw %}  $$j$$  {% endraw %} users.