## Blbmethods

Blbmethods is a R package that implements the bag of little bootstrap algorithms for

* Linear Regression

* Logistic Regression

* Random Forests

You can install it from my personal Github using the command

```r
devtools::install_github("dlmao/blbmethods")
```

### Limitations of Boostrapping

Boostrapping falls into a category of techniques know as resampling methods. The idea behind boostraping is to artificially generate many "new" datasets from one dataset. The efficacy of boostrapping depends on two things: the size of the dataset and the number of resamplings. Both need to be high in order to have accurate boostrapping results. However, when the size of the dataset becomes too large, in otherwords dealing with "big data", the computational efficiency of performing boostrap dramatically decreases.

### What is Bag of Little Boostrap?

Enter bag of little boostrap (blb). Blb is a technique that combines the map reduce algorithm with boostrapping to create a robust and computationally efficient resampling method that is scalable for large amounts of data. First, the dataset is divided into $n$ approximatly equal sized subsets. Boostrapping is then performed on each of these subsets. The results of each subset are then averaged.