---
date: "2025-03-24T00:00:00Z"
tags:
title: Reviewing Decision Tree Models 
---

Decision Trees (and their counterparts) form another class of fundamental ML models and in this blog post I'd like to briefly review some of the key concepts behind them. 

### Introduction 
The simplest way to think of a decision tree is that it is a tree of rules which can be used to make predictions both numerical (like in the case of regression) and categorical (like in the case of classification). 

When we make a prediction with a single decision tree on a test point, we start by taking our input features \(x\) and answering a sequence of questions about \(x\). These questions can be:
- Is the numeric feature \(x_i\) >= \(3.2\)?
- Is the categorical feature \(x_j\) == Sunny?
- Is the ordinal feature \(x_k\) < 3 Stars?

As each question in the tree is answered we traverse down the tree until the last question which brings us to some leaf node. We can think of the leaf node as corresponding to the subset of training data whose features also have the same answers to these questions. 

In other words we have placed our test datapoint into the same bucket as a bunch of training data points and we then make a prediction about our test point by aggregating the labels of the training points in this bucket (strong K-NN vibes). If we are performing classification we may use majority voting for aggregation and if we are performing regression we may use averaging for aggregation.  

#### Example
Let's take a look at a classification example where we want to classify iris flowers (a classic toy [dataset](https://archive.ics.uci.edu/dataset/53/iris)) as one of 3 classes. Our training set consists of 150 data points each containing 4 features (sepal length, sepal width, petal length, petal width). The decision tree that was generated for this problem is shown below (more on how it is generated later). 

We see that the root of the tree asks whether the petal length is less than or equal to \(2.45\). If our input answers this question as yes (True) then we proceed to the left otherwise to the right (False). In this tree the left node is a leaf node and if you land there then using the majority vote scheme you would predict the class as being class 0 (setosa). If we land in the right node then we have another question to answer which is whether the petal width is less than or equal to \(1.75\). Based on the answer to that question we would proceed until we land on a leaf node where we would make our prediction. 

{{< imgproc iris_dt Resize "900x" "Example Decision Tree" >}} Example Decision Tree. (From scikit-learn docs.) {{< /imgproc >}}

### Training
At this point an obvious question emerges:
> How is a decision tree constructed (or learned) and what is it optimizing? 

Conceptually, we want a decision tree to ask questions that split the training data in such a way that leaf nodes are relatively "pure". We can think of purity in the classification context as meaning the leaf nodes contain training samples that are predominantly of the same class (the closer to one class the more pure). In the regression context, purity would mean the leaf node contains training samples whose labels (target values) are similar to each other (the more similar the more pure). 

Intuitively, if a decision tree has splits that lead to relatively pure leaf nodes, then we can argue the tree has learned patterns in feature space which can be used to discriminate between different classes or target values. This means we have a discriminative model that can help us with classification or regression. 

#### Measuring Split Purity 
For any node in our decision tree we would like to measure how good of a split it is and to do so we would like a measure of purity (or impurity).

For classification, Gini impurity and Entropy are two of the most common measures of purity. 

The Gini impurity (opposite of purity) score is given by:

$$
\begin{align}
Gini(node) = 1 - \sum_{i=1}^k p_i^2 \\
\end{align}
$$

Here \(k\) denotes the number of different classes spanned by the samples in the node, and \(p_i\) is the proportion of samples that belong to class \(i\). We can see that if all samples are of the same class then the proportion for that class becomes 1 while all others become 0 leading to an impurity score of 0 (equivalently the most pure). Alternatively, if we had say \(2\) classes and the samples were evenly split we would have an impurity score \(0.5\) meaning not that pure.

Entropy, which will look familiar if you've also been looking at cross-entropy, is given by:

$$
\begin{align}
Entropy(node) = - \sum_{i=1}^k p_i \log(p_i) \\
\end{align}
$$

Here once again \(k\) denotes the number of different classes spanned by the samples in the node, and \(p_i\) is the proportion of samples that belong to class \(i\). We can see that if all samples are of the same class then the proportion for that class becomes 1 while all others become 0 leading to an entropy of 0 (the most pure). Alternatively, if we had say \(2\) classes and the samples were evenly split we would have an entropy of \(-\frac{1}{2} \log(\frac{1}{2}) - \frac{1}{2} \log(\frac{1}{2}) = 1 \), indicating not pure. 

For regression, a common way to measure purity is to use mean squared error (MSE). In this case we assume that a node will make a prediction (\(\hat{y}\)) that is equal to the mean target value of the \(n\) samples in the node. The MSE of a given node is given by:


$$
\begin{align}
MSE(node) = \frac{1}{n} \sum_{i=1}^n  (\hat{y}-y_i)^2 \\
\end{align}
$$

We can see when all samples in a node contain the same target value then the MSE will be 0. As the samples deviate farther and farther from each other (and thus the mean) the MSE will go up.

#### Optimization
So far we have touched on one property we would like in a decision tree, namely relatively pure leaf nodes. On the other hand we may be able to achieve this with a very large and complicated tree. This leads us to the second ideal property which is a decision tree that is relatively small in size.

Unfortunately, finding the optimal tree is known to be an NP-hard problem. It's not computationally feasible to try all permutations of feature splits where the feature splits themselves depend on both the number of features and the range of values they can take on. 

In practice decision trees are grown/constructed/trained using a greedy algorithm. That is you would begin building a tree by selecting the single split that provides the largest increase in purity (equivalently largest decrease in impurity) between parent and children. The purity of the children can be evaluated by summing their individual purities weighted by the proportion of samples that belong to each child. This greedy procedure proceeds recursively until either some heuristic stopping criteria is reached or no further split can be found. 

#### Regularization




### Ensemble Decision Tree Models
which can be applied to either regression or classification problems.  Decision trees are also often used as a building block in ensemble models like random forests and gradient boosted trees.he
<!-- To make a prediction you apply the decision rules in a particular order on some input data and use the training data points that fall under the same leaf node to make a prediction. -->


<!-- ### Introduction 
The simplest way to think of a decision tree is that it is a model that encodes a hierarchy of decision rules which can be applied to either regression or classification problems.  Decision trees are also often used as a building block in ensemble models like random forests and gradient boosted trees.

Decision trees (and derivative models) are particularly well suited for dealing with tabular data where it's common to find features of mixed types. For instance consider a movie rating dataset where you may find a mix of numeric, categorical, and ordinal input features. 

It's also worth noting that decision tree based models are very different from other classic models like [linear regression]({{< ref "/blog/2025-03-03-lr" >}} ""), [logistic regression]({{< ref "/blog/2025-03-10-logr" >}} "") and neural networks. A few big differences include:
- Decision trees don't make an underlying assumption about the relationship between input features and output. 
- Decision trees are not defined by a fixed number of model parameters (e.g they are non-parametric). 
- Decision trees are not trained via gradient-based optimization algorithms.  -->



### Other Nuances
Like in our discussion of nuances in [linear regression]({{< ref "/blog/2025-03-03-lr#normalization" >}} ""), there are several things you want to keep in mind before applying logistic regression. Specifically, we still need to normalize our data and adding regularization is a good idea. 

#### Quality of Fit 
While there are some notions of evaluating the quality of the fit of the model from a regression point of view, the most practical way to evaluate a logistic regression model is through the lens of classification.

Broadly speaking that means we would make predictions on some validation set and formulate a \(2 \times 2\) confusion matrix which can then be used to generate a variety of metrics like accuracy, precision, recall etc. We can also look at tradeoffs between metrics (like precision and recall) when we vary the threshold used for classification. 

#### Conclusion
Logistic regression is not only a classic ML (and stats) model but it is also one of the first models to try when working on a binary classification problem. Additionally, a lot of the ingredients in logistic regression (cross-entropy loss, gradient descent, linear combinations of features) show up in neural networks which means learning about logistic regression can help you to understand neural networks. 

Finally, we see that scikit has the following [API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for logistic regression. As an aside it looks like scikit implements alternative optimization algorithms that don't include gradient descent. The default is [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) which is a quasi-Newton method that appears to converge faster than gradient descent. The API also defaults to using L2 regularization.

#### References
A few misc references that I consulted in addition to all the ones linked above:
- [CMU Lecture Slides](https://web.stanford.edu/~jurafsky/slp3/5.pdf)