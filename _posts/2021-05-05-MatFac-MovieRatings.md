---
layout: post
title: "How Does Netflix Recommend Your Next Movie: Building a Simple Recommendation System with Matrix Factorization"
author: David Mao
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Suppose you watch a movie on Netflix that you really enjoy. After you finish the movie, you give the movie a five star rating. Immediately afterwards, Netflix reccommends you several other movies that you might also enjoy. What you have witnessed here is a reccomendation system in action. Recommendation systems are machine learning models that seeks to reccommend products to users based on past user interaction. It has applications in many places, mainly in advertisements, shopping websites, and movie applications.

## How Does Matrix Factorization Work?

Let's go back to the Netflix Movie example. Suppose on Netflix's site, there are {% raw %} $$m$$  {% endraw %} users and {% raw %}  $$n$$  {% endraw %} users. Then if each user gave each movie a rating, we would have {% raw %} $$m \times n$$  {% endraw %} ratings, which we can then store into an {% raw %} $$m \times n$$  {% endraw %} matrix, which we will call {% raw %} $$M$$  {% endraw %}, where each row of this matrix represents a user and each column represents a movie.

Now this matrix {% raw %} $$M$$  {% endraw %} can be approximated by two lower dimension matrices by {% raw %} $$M \approx U*V$$  {% endraw %}. Here, {% raw %} $$U$$  {% endraw %} is size {% raw %} $$m \times r$$  {% endraw %},  {% raw %} $$V$$  {% endraw %} is size {% raw %} $$r \times n$$  {% endraw %}, and {% raw %} $$r$$  {% endraw %} is less than the smallest of {% raw %} $$m, n$$  {% endraw %}, but is usually much smaller than both. So what do {% raw %} $$U$$  {% endraw %} and {% raw %} $$V$$  {% endraw %} represent? Since the rows of {% raw %} $$M$$  {% endraw %} correspond to the users and the columns correspond to the movies, we can interpret the rows of {% raw %} $$U$$  {% endraw %} as vectors corresponding to each users and the columns of {% raw %} $$V$$  {% endraw %} as vectors corresponding to each movie.

So now how do we find {% raw %} $$U$$  {% endraw %} and {% raw %} $$V$$  {% endraw %}? In practice, we are given a training set of values in {% raw %} $$M$$ {% endraw %} which we already know the values of. So if each entry of {% raw %} $$M$$  {% endraw %} is denoted {% raw %} $$M_{ij}$$ {% endraw %}, we will denote the training set  {% raw %} $$\Omega_{train}=\{(i,j)\}$$ {% endraw %} where {% raw %} $$M_{ij}$$ {% endraw %} is known. We want {% raw %} $$U*V$$ {% endraw %} to be as close to {% raw %} $$M$$ {% endraw %} as possible for the entries of {% raw %} $$M$$ {% endraw %} that we know. More specifically, we can write this as the following optimization problem:

{% raw %} $$\underset{U,V}{\min}||P_{\Omega_{train}}(U*V-M)||^2_F$$ {% endraw %}

where {% raw %} $$P_{(i,j)}(x)=x$$ {% endraw %} if {% raw %} $$(i,j)\in\Omega_{train}$$ {% endraw %}, and {% raw %} $$||.||^2_F$$  {% endraw %} denotes the Frobenius norm.