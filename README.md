# Clasificación de cubiertas forestales - Kaggle Competition

## Overview

- **Start Date:** Oct 25, 2023
- **End Date:** Nov 19, 2023

### Description

The competition aims to identify the type of forest cover in a region based on cartographic variables. Ecosystem managers often need basic descriptive information about forest lands to support decision-making processes. Predictive models can be used to obtain this information.

The goal is to build a predictive model to predict the type of coverage in natural forests. The study area includes four wilderness areas in North America. Twelve cartographic measures related to seven types of forests were used as observations.

The provided dataset, derived from the UCI repository, has been slightly distorted to avoid the identification of test labels.

### Evaluation

The evaluation metric for this competition is the average cost, i.e., the average of the costs associated with the classification of each sample.

The cost matrix is given by:

```
C = [[ 0,   5,  1, 1, 1,  1, 1],

     [10,   0,  1, 1, 1,  1, 1],
     
     [20,  20,  0, 5, 5, 50, 5],
     
     [20,  20, 10, 0, 1, 50, 5],
     
     [20, 100,  5, 1, 0,  5, 5],
     
     [ 5,  10, 10, 5, 1,  0, 1],
     
     [10,   5,  1, 1, 1,  1, 0]]
```

The cost is calculated as:

`score = np.mean([C[int(i)][int(j)] for i, j in zip(y, p)])`

where y is a list or array with the actual classes, and p is the classifier's predictions.
