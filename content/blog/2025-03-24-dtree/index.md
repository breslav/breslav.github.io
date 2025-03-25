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

Conceptually, we want a decision tree to ask questions that split the training data in such a way that leaf nodes are relatively "pure". We can think of purity in the classification context as meaning a leaf node contains training samples that are predominantly of the same class (the closer to one class the more pure). In the regression context, purity would mean a leaf node contains training samples whose labels (target values) are similar to each other (the more similar the more pure). 

Intuitively, if a decision tree has splits that lead to relatively pure leaf nodes, then we can argue the tree has learned patterns in feature space which can be used to discriminate between different classes or target values. This means purity helps us produce a discriminative model that can be used for classification and regression. 

#### Measuring Split Purity 
When we build a decision tree we end up specifying a tree of splits. When a split is applied to a node we get children nodes whose purity we would like to measure. 
This means we need a measure of purity (or impurity) to understand whether our split is leading us towards pure leaf nodes.

For **classification**, Gini impurity and Entropy are two of the most common measures of purity. 

The Gini impurity score (opposite of purity) is given by:

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

For **regression**, a common way to measure purity is to use mean squared error (MSE). In this case we assume that a node will make a prediction (\(\hat{y}\)) that is equal to the mean target value of the \(n\) samples in the node. The MSE of a given node is given by:


$$
\begin{align}
MSE(node) = \frac{1}{n} \sum_{i=1}^n  (\hat{y}-y_i)^2 \\
\end{align}
$$

We can see when all samples in a node contain the same target value then the MSE will be 0. As the samples deviate farther and farther from each other (and thus the mean) the MSE will go up. Note in this case MSE is equivalent to the variance of the target values.

#### Optimization
So far we have touched on one property we would like in an ideal decision tree, namely relatively pure leaf nodes. On the other hand we may be able to achieve this by creating a very large and complicated tree. This leads us to a second property we would like in an ideal decision tree which is that it has a relatively small size.

Unfortunately, finding the global optimal tree is known to be an NP-hard problem. It's not computationally feasible to try all permutations of feature splits where the feature splits themselves depend on both the number of features and the range of values they can take on. 

In practice decision trees are grown (trained) using greedy algorithms. The general flow of these greedy algorithms is as follows:
- First initialize the root node of a tree by selecting the single split that provides the largest increase in purity (equivalently largest decrease in impurity) between root and the resulting children. 
  - The purity of the children can be evaluated by summing their individual purities weighted by the proportion of samples that belong to each child. 

- Next select a new split for each child that provides the largest increase in purity (or decrease in impurity) between child and grandchildren.
- Repeat the previous step recursively until either some stopping criteria is reached or no further split can be found to increase purity. 

The time complexity of the greedy algorithm is roughly \(O(mn\log(n))\) where \(n\) is the number of training samples and \(m\) is the number of features. Some nuances on this time complexity are discussed [here](https://sebastianraschka.com/pdf/lecture-notes/stat451fs20/06-trees__notes.pdf).

It's worth noting that there are a variety of decision tree algorithms and some of the most popular ones are ID3, C4.5, and CART. In general these different algorithms can vary in what kind of splitting measure they use, whether the splits are binary or n-ary, and how they perform pruning (see next section).


#### Regularization

One challenge with decision trees is that if you allow them to keep growing in size they can easily overfit to the training data. There are several regularization techniques to reduce the freedom and complexity of a decision tree. These include:
- Limiting the maximum depth of the tree. (Very common.)
- Limiting the number of features being evaluated for each split.
- Limiting the number of leaf nodes.
- Setting a minimum number of samples for splitting to be allowed.
- Setting a minimum number of samples a node must have in order to be created.

We can also think of these limits as forms of pre-pruning (proactively preventing the growth of the tree). There are also post-pruning techniques where you allow the tree to first grow without restriction and then afterwards nodes are pruned.


### Pros and Cons 
There are several pros and cons to consider before applying decision trees to your problem.

- Pro: Decision trees are interpretable. For any particular leaf node you can find the path from the root to it and this path represents a sequence of questions and answers (rules) that lead to a certain prediction. Interpretability is really important in some domains.

- Pro: Decision trees can provide a notion of feature importance. That is we can look at how much purity was gained when splitting on a particular feature and use this as a signal of which features are most important.

- Pro: Decision trees can happily work with mixed data types. 
- Pro: You don't need to worry about scaling/normalizing input features.

- Con: Decision trees by nature make splits that are orthogonal to the axes (e.g \(x < 3\) or \(y > 5\)) which means that the results of a decision tree can be sensitive to the orientation of the data. If some decision boundary is not axis aligned then it will be harder to model it cleanly with a decision tree. Rotating your data with PCA may help with this problem but there is no guarantee. 

- Con: Decision trees are prone to overfitting. This means small changes to the training data (or hyperparameters) can lead to very different models. We will see that ensemble models can help to reduce the variance of the model.


### Ensemble Learning

In ML we can often come up with a better predictor by aggregating the predictions of many different models. This general concept is known as ensemble learning and can be informally thought of "the wisdom of the crowd" (not necessarily the most intuitive). 

Given that decision trees can be prone to overfitting, creating ensembles of decision trees is one way to get better performance and reduce model variance. The most popular method of ensembling decision trees is called random forests.

In general, random forests use two strategies for creating an ensemble of different decision trees: 
- The first is that each decision tree is trained using a different subset of training data. The subset of training data is usually generated by sampling with replacement and leads to a diversity of trees since they each model different datasets. This overall approach is known as bagging (short for bootstrap aggregating).
- Additionally, when growing the decision trees, each tree will only consider a random subset of features when determining the best splits. This further helps to decorrelate the different trees since they will make predictions based on different features.

Another popular way to ensemble decision trees is to construct decision trees sequentially so that later trees can learn to correct the mistakes of previous trees. This strategy is broadly known as boosting with two of the most popular methods being AdaBoost and gradient boosting.

At a very high level AdaBoost creates an ensemble of trees by identifying which training samples were misclassified (or had a large residual in the regression context) and increasing the weight of those samples so that they become more influential in the construction of the next tree (e.g the optimal split is influenced more by those samples).

Like AdaBoost, gradient boosting creates a sequence of decision trees but instead of changing the weights of training samples, the objective is instead to have subsequent trees fit the errors of their predeceessors directly.

There is a lot more depth to these various flavors of decision tree ensembles that are beyond the scope of this blog post. Gradient boosting in particular is still a relatively state of the art technique when it comes to prediction tasks associated with tabular data. 

An interesting question that also arises with respect to ensembles of decision trees is how does that impact the interpretability. It turns out we can still assess feature importance from the learned trees. Additionally, there are methods like [Shapley values](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) that we can leverage.

<!-- To make a prediction you apply the decision rules in a particular order on some input data and use the training data points that fall under the same leaf node to make a prediction. -->


<!-- ### Introduction 
The simplest way to think of a decision tree is that it is a model that encodes a hierarchy of decision rules which can be applied to either regression or classification problems.  Decision trees are also often used as a building block in ensemble models like random forests and gradient boosted trees.

Decision trees (and derivative models) are particularly well suited for dealing with tabular data where it's common to find features of mixed types. For instance consider a movie rating dataset where you may find a mix of numeric, categorical, and ordinal input features. 

It's also worth noting that decision tree based models are very different from other classic models like [linear regression]({{< ref "/blog/2025-03-03-lr" >}} ""), [logistic regression]({{< ref "/blog/2025-03-10-logr" >}} "") and neural networks. A few big differences include:
- Decision trees don't make an underlying assumption about the relationship between input features and output. 
- Decision trees are not defined by a fixed number of model parameters (e.g they are non-parametric). 
- Decision trees are not trained via gradient-based optimization algorithms.  -->




#### Conclusion

#### References
A few misc references that I consulted in addition to all the ones linked above: