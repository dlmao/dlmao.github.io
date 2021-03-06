---
layout: post
title: "How Netflix Recommend Your Next Movie: Building a Simple Recommendation System with Matrix Factorization"
author: David Mao
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

Suppose you watch a movie on Netflix that you really enjoy. After you finish the movie, you give the movie a five star rating. Immediately afterwards, Netflix reccommends you several other movies that you might also enjoy. What you have witnessed here is a reccomendation system in action. Recommendation systems are machine learning models that seeks to recommend products to users based on past user interaction. It has applications in many places, mainly in advertisements, shopping websites, and movie applications.

## How Does Matrix Factorization Work?

Let's go back to the Netflix Movie example. Suppose on Netflix's site, there are {% raw %} $$m$$  {% endraw %} users and {% raw %}  $$n$$  {% endraw %} users. Then if each user gave each movie a rating, we would have {% raw %} $$m \times n$$  {% endraw %} ratings, which we can then store into an {% raw %} $$m \times n$$  {% endraw %} matrix, which we will call {% raw %} $$M$$  {% endraw %}, where each row of this matrix represents a user and each column represents a movie.

Now this matrix {% raw %} $$M$$  {% endraw %} can be approximated by two lower dimension matrices by {% raw %} $$M \approx U*V$$  {% endraw %}. Here, {% raw %} $$U$$  {% endraw %} is size {% raw %} $$m \times r$$  {% endraw %},  {% raw %} $$V$$  {% endraw %} is size {% raw %} $$r \times n$$  {% endraw %}, and {% raw %} $$r$$  {% endraw %} is less than the smallest of {% raw %} $$m, n$$  {% endraw %}, but is usually much smaller than both. So what do {% raw %} $$U$$  {% endraw %} and {% raw %} $$V$$  {% endraw %} represent? Since the rows of {% raw %} $$M$$  {% endraw %} correspond to the users and the columns correspond to the movies, we can interpret the rows of {% raw %} $$U$$  {% endraw %} as vectors corresponding to each users and the columns of {% raw %} $$V$$  {% endraw %} as vectors corresponding to each movie.

So now how do we find {% raw %} $$U$$  {% endraw %} and {% raw %} $$V$$  {% endraw %}? In practice, we are given a training set of values in {% raw %} $$M$$ {% endraw %} which we already know the values of. So if each entry of {% raw %} $$M$$  {% endraw %} is denoted {% raw %} $$M_{ij}$$ {% endraw %}, we will denote the training set  {% raw %} $$\Omega_{train}=\{(i,j)\}$$ {% endraw %} where {% raw %} $$M_{ij}$$ {% endraw %} is known. We want {% raw %} $$U*V$$ {% endraw %} to be as close to {% raw %} $$M$$ {% endraw %} as possible for the entries of {% raw %} $$M$$ {% endraw %} that we know. More specifically, we can write this as the following optimization problem:

{% raw %} $$\underset{U,V}{\min}||P_{\Omega_{train}}(U*V-M)||^2_F$$ {% endraw %}

where {% raw %} $$P_{(i,j)}(x)=x$$ {% endraw %} if {% raw %} $$(i,j)\in\Omega_{train}$$ {% endraw %}.

We can solve this optimization problem using gradient descent. First, lets get some intuition on what gradient descent is doing here. One specific entry {% raw %} $$M_{ij}$$  {% endraw %} is represented by {% raw %} $$U_{i.}*V_{.j}$$  {% endraw %}, which is the ith row of {% raw %} $$U$$  {% endraw %} times the jth row of {% raw %} $$V$$  {% endraw %}. So given a rating {% raw %} $$M_{ij}$$  {% endraw %}, we know something about the the ith user and the jth user, and can adjust {% raw %} $$U_{i.}$$  {% endraw %} and {% raw %} $$V_{.j}$$  {% endraw %} accordingly. Now, to implement this mathematically. Recall our objective function from earlier. Since we are working with the Frobenius norm, with each training point in {% raw %} $$\Omega_{train}$$ {% endraw %}, we want to minimize the following:

{% raw %} $$\underset{U_{i.},V_{.j}}{\min}||P_{\Omega_{train}}(U_{i.}*V_{.j}-M_{ij})||^2$$ {% endraw %}

To find the gradient, we take the derivative of this function is accordance to {% raw %} $$U_{i.}$$ {% endraw %} and {% raw %} $$V_{.j}$$ {% endraw %}. We get the following gradients:

{% raw %} $$\Delta U_{i.}=2V_{.j}(U_{i.}V_{.j}-M_{ij})$$ {% endraw %}

{% raw %} $$\Delta V_{.j}=2U_{i.}(U_{i.}V_{.j}-M_{ij})$$ {% endraw %}

Thus, our gradient descent algorithm becomes:

{% raw %} $$\Delta U_{i.}=U_{i.}-\eta 2V_{.j}(U_{i.}V_{.j}-M_{ij})$$ {% endraw %}

{% raw %} $$\Delta V_{.j}=V_{.j}-\eta 2U_{i.}(U_{i.}V_{.j}-M_{ij})$$ {% endraw %}

## A Coding Example in Python

For this part, we will use a dataset from MovieLens, a movie reccomendation system. The dataset includes movie ratings on a scale of 1 star to 5 stars, from 610 users and 9724 movies, but we only know 100836 (1.7%) of the ratings. The dataset can be found [here](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). The full Python notebook can be found on my personal Github [here](https://github.com/dlmao/MatFac_Recommendation).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```

First, we load our dataset. The first column of our dataset contains ids that represent individual users, the second column contains ids that represent movies, and the third column contains movie ratings corresponding to a specific user and movie.

```python
ratings_list = pd.read_csv('ratings.csv', sep=',', header=0, usecols=[0, 1, 2]).values
ratings_list
```




    array([[1.00000e+00, 1.00000e+00, 4.00000e+00],
           [1.00000e+00, 3.00000e+00, 4.00000e+00],
           [1.00000e+00, 6.00000e+00, 4.00000e+00],
           ...,
           [6.10000e+02, 1.68250e+05, 5.00000e+00],
           [6.10000e+02, 1.68252e+05, 5.00000e+00],
           [6.10000e+02, 1.70875e+05, 3.00000e+00]])




Now, we construct matrix {% raw %} $$M$$  {% endraw %}. To make things easier, we will replace user ids with values from 0 to n-1 and movie ids with values from 0 to m-1.

```python
# replace movie id and user id with new ids ranging from 0 to n-1, m-1 for easier indexing
n, indices = np.unique(ratings_list[:,1], return_inverse=True)
ratings_list[:,0] = ratings_list[:,0] - 1
ratings_list[:,1] = indices
```

Matrix {% raw %} $$M$$  {% endraw %} is a {% raw %} $$610 \times 9724$$  {% endraw %} matrix with rows as users and columns as movies. Since we do not interact with the values of {% raw %} $$M$$  {% endraw %} that we do not know, we will just keep them as zero.

```python
m = 610
n = 9724
ratings_mat = np.zeros((m, n))
for i in range(len(ratings_list)):
    ratings_mat[int(ratings_list[i,0]), int(ratings_list[i,1])] = ratings_list[i,2]

ratings_mat
```




    array([[4. , 0. , 4. , ..., 0. , 0. , 0. ],
           [0. , 0. , 0. , ..., 0. , 0. , 0. ],
           [0. , 0. , 0. , ..., 0. , 0. , 0. ],
           ...,
           [2.5, 2. , 2. , ..., 0. , 0. , 0. ],
           [3. , 0. , 0. , ..., 0. , 0. , 0. ],
           [5. , 0. , 0. , ..., 0. , 0. , 0. ]])




Now, we will construct {% raw %} $$\omega_{train}$$  {% endraw %} and {% raw %} $$\omega_{test}$$  {% endraw %}. Taking the indexes of our known values, (the first two columns of our initial data), we will then do a 0.2 split test train split.

```python
idx = ratings_list[:,0:2]
idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=1) # split known points into test and train set
idx_train.shape, idx_test.shape
```




    ((80668, 2), (20168, 2))




Next, we will then create two functions to calculate the two gradients for {% raw %} $$U$$  {% endraw %} and {% raw %} $$V$$  {% endraw %} we found earlier, {% raw %} $$\Delta U_{i.}=2V_{.j}(U_{i.}V_{.j}-M_{ij})$$ {% endraw %}, {% raw %} $$\Delta V_{.j}=2U_{i.}(U_{i.}V_{.j}-M_{ij})$$ {% endraw %}.

```python
def gradu(ui, vj, mij):
    return 2*vj*(np.dot(ui, vj) - mij) # calculate gradient for U

def gradv(ui, vj, mij):
    return 2*ui*(np.dot(ui, vj) - mij) # calculate gradient for V
```

Finally, we implement the gradient descent algorithm. The first thing we need to do is initialize the values of {% raw %} $$U$$  {% endraw %} and {% raw %} $$V$$  {% endraw %} given a value of {% raw %} $$r$$  {% endraw %}, which we will chose to equal 10. Next, for a specified number of iterations, we will, for every point in our training set, subtract the gradient times the learning rate. We choose here the the learning rate to be 0.001 and iterations to be 20.

```python
def MatFac_train(train, M, m, n, r=10, eta=0.001, MAX_ITER=20, seed=1):
    np.random.seed(seed)
    # initialize U and V
    U = np.random.random_sample((m, r))
    V = np.random.random_sample((r, n))
    itr = 0
    while itr<MAX_ITER:
        itr = itr + 1
        for k in range(len(train)):
            i = int(train[k,0])
            j = int(train[k,1])
            mij = M[i,j]
            delt_u = gradu(U[i,:], V[:,j], mij)
            U[i,:] = U[i,:] - eta * delt_u # gradient descent for U
            delt_v = gradv(U[i,:], V[:,j], mij)
            V[:,j] = V[:,j] - eta * delt_v # gradient descent for V
            
    return U, V
```


```python
U, V = MatFac_train(idx_train, ratings_mat, m, n)
U.shape, V.shape
```




    ((610, 10), (10, 9724))




Now we need a measure to judge the accuracy of our prediction matrices. For that, we will use the MSE, which is calculated by:

{% raw %} $$MSE=\frac{1}{size(\Omega_{test})}\sum_{i,j\in \Omega_{test}}(U_{i.}*V_{.j}-M_{ij})$$  {% endraw %}.

```python
def test_MSE(test, M, U, V):
    tot_sum = 0
    l = len(test)
    for k in range(l):
        i = int(test[k,0])
        j = int(test[k,1])
        tot_sum = tot_sum + (np.dot(U[i,:], V[:,j]) - M[i,j])**2
        
    return tot_sum/l
```


```python
test_MSE(idx_test, ratings_mat, U, V)
```




    0.8340972471734859


With only 1.7% of our data known, we were able to predict user ratings with an MSE of 0.83, meaning on average, we were able to predict ratings to be within one star of what users actually felt about the movie. 

## Future Tasks

For Matrix Factorization, the choice of  {% raw %} $$r$$  {% endraw %} matters a lot. Try and see what different values of  {% raw %} $$r$$  {% endraw %} do. Additionally, for gradient descent, the learning rate combined with the number of iterations is also important for determining the optimal value of our optimization problem. Try different combinations of these values to see if a better minimum can be found.
