## Blbmethods

Blbmethods is a R package that implements the bag of little bootstrap algorithms for

* Linear Regression

* Logistic Regression

* Random Forests

You can install it from my personal Github using the R command

```r
devtools::install_github("dlmao/blbmethods")
```

The full source repository can be found [here](https://github.com/dlmao/blbmethods).

### Limitations of Bootstrapping

Bootstrapping falls into a category of techniques know as resampling methods. The idea behind bootstrapping is to artificially generate many "new" datasets from one dataset. The efficacy of bootstrapping depends on two things: the size of the dataset and the number of resamplings. Both need to be high in order to have accurate bootstrapping results. However, when the size of the dataset becomes too large, in otherwords dealing with "big data", the computational efficiency of performing bootstrap dramatically decreases.

### What is Bag of Little Boostrap?

Enter bag of little boostrap (blb). Blb is a technique that combines the map reduce algorithm with bootstrapping to create a robust and computationally efficient resampling method that is scalable for large amounts of data. First, the dataset is divided into n approximatly equal sized subsets. Bootstrapping is then performed on each of these subsets. The results of each subset are then averaged.

### Package Functions and Performance Metrics

A vignette describing the functions and performance metrics of the package can be found [here](/assets/vignette/UsingBlbmethods.html)