---
date: "2025-03-02T00:00:00Z"
tags:
title: Reviewing Linear Regression
---

Linear regression is a foundational model both in the machine learning context as well as the statistics context. While at a surface level it's generally considered a simple model, under the hood there is a fair amount of math and nuance. In this blog post I want to review the most important concepts associated with linear regression (at least from a ML point of view). 

### Core
Linear regression models the relationship between some dependent variable \(y\) and independent variable \(x\) as being linear. With a single independent variable it is commonly expressed as: \(y = wx + b\) or equivalently \(y = w_{0} + w_{1}x\). In multiple linear regression where we have multiple independent variables it can be expressed as: \(y = w_{0} + w_{1}x_{1} + w_{2}x_{2} + \cdots + w_{n}x_{n}\) which in vector notation is \(y = \vec{w} \cdot \vec{x}\). 

Once we have a model the next step is to fit it to some data \(\{(\vec{x_{i}},y_{i})\}\). Fitting or training the model means we need to find the values of the parameters of the model such that they optimize some loss/objective. In linear regression a commonly chosen objective is minimizing the sum of squared errors. That is we want to find \(w\) that minimizes: 

\[ \sum_{i=1}^{m} (\vec{w} \cdot \vec{x_{i}} - y_{i})^{2}\] 
Where \(m\) is the number of training data points we have.

#### Closed Form Solution
It turns out there is a closed form solution to the above minimization problem. First, we can rewrite the loss in matrix form as minimizing 

$$(Xw - y)^{T}(Xw - y)$$ 

where \(y\) is an \(m\) dimensional vector of targets (dependent values), \(X\) is a (\(m \times d+1)\) dimensional matrix containing the features (independent values) as rows, and \(w\) is a \(d+1\) dimensional vector representing the weights and bias of the model. For simplicity we can just treat the bias term as the first weight and augment the input features with a \(1\) and consider both to be \(d\) dimensional.

Rewriting and expanding the product we have:

$$ 
\begin{align}
= (w^{T}X^{T} - y^{T})(Xw-y) \\ 
= w^{T}X^{T}Xw - w^{T}X^{T}y - y^{T}Xw + y^{T}y \\
=  w^{T}X^{T}Xw - 2w^{T}X^{T}y + y^{T}y \\
\end{align}
$$

We know that this loss is convex ([wiki](https://en.wikipedia.org/wiki/Linear_regression)) and therefore it has a global minimum which we can find by setting the derivative to 0.
This yields our solution for \(w\):

$$ 
\begin{align}
\frac{\partial}{\partial W} (w^{T}X^{T}Xw - 2w^{T}X^{T}y + y^{T}y) = 0 \\ 
2X^{T}Xw -2X^{T}y = 0 \\
X^{T}Xw = X^{T}y \\
w = (X^{T}X)^{-1}X^{T}y \\
\end{align}
$$

In practice there are several downsides to using this closed form solution for \(w\).

First it can be expensive to compute which can be seen by examining the [complexity of matrix operations](https://en.wikipedia.org/wiki/Computational_complexity_of_mathematical_operations). The first multiplication term is \(O(d^{2}m)\) and its inversion is \(O(d^{2.37})\). In aggregate computing \(w\) is \(O(d^{2}m + d^{2.37})\) which is polynomial in the number of features and linear in the number of training samples. When \(d\) and \(m\) are very large not only is it costly in time but we also have to store a \(d \times d\) matrix in memory.

A second concern is that the matrix \(X^{T}X\) needs to be invertible. If one feature in \(X\) is a linear combination of other features then \(X\) will not have full rank and therefore \(X^{T}X\) will not be invertible.  

#### Gradient Descent Solution 

For larger scale linear regression tasks the practical alternative to the closed form solution is one based on gradient descent. Consider mini batch gradient descent where we will minimize the following loss on each mini batch. 

\[ \sum_{i=1}^{\frac{m}{n}} (\vec{w} \cdot \vec{x_{i}} - y_{i})^{2}\] 

This is identical to our original loss but instead of being based on \(m\) examples it'll be on \(\frac{m}{n}\) examples where \(n\) is the number of mini batches. It's easy to see the gradient with respect to \(w\) is:

\[ \sum_{i=1}^{\frac{m}{n}} 2(\vec{w} \cdot \vec{x_{i}} - y_{i})\vec{x_{i}}\] 

This means that computing the gradient for a single mini batch is \(O(d\frac{m}{n})\) and for \(n\) mini batches becomes \(O(dm)\), which is significantly better than the closed form solution. Even if convergence requires \(k\) epochs the overall complexity \(O(dmk)\) is still going to be much better than the polynomial in \(d\).

#### Aside: How do we know this loss is convex?
In our closed form solution we took for granted that our loss function is [convex](https://en.wikipedia.org/wiki/Convex_function) but how do we know that? The way we know the sum of squared errors loss is convex is from computing the [Hessian](https://en.wikipedia.org/wiki/Convex_function) matrix and seeing that it is [positive semidefinite](https://en.wikipedia.org/wiki/Convex_function). This is what allows us to know that the minima we found is the global minimum. Starting with the first derivative we [previously](#closed-form-solution) calculated we can see that the 2nd derivative (our Hessian) is the term \(X^{T}X\). The definition of positive semi definite is that \(z^{T}Mz >= 0\) for all \(z\). Applying this to our case we have:

$$
\begin{align}
z^{T}X^{T}Xz \; >= 0 \quad \forall z  \quad (?) \\
(Xz)^{T}(Xz) \; >= 0  \quad \forall z  \quad (?) \\
\lVert Xz \rVert_2^2 >=0 \; \quad \forall z  \quad (?) \\
\end{align}
$$

We can see the above expression boils down to an inner product of a vector with itself which is going to be equal to the sum of its squared values or equivalently the magnitude squared which is also known as the L2 norm squared. The sum of squared values will always be \(>=0\) for all vectors and therefore our 2nd derivative is positive semi definite and therefore our loss is convex! 
### Other Nuances
There are a number of other nuances worth mentioning as they add to our broader understanding of using linear regression in practice.

#### Normalization
In practice when we apply linear regression to some arbitrary set of features we need to worry about the scale of those features. If some features take on very large values then the model will care mostly about those features because the squared loss will be most sensitive to their weights. This comes at the expense of features that happen to have smaller values but may be as important semantically. 

A few common options for normalization of feature values include:
- Subtracting the mean and dividing by the standard deviation which results in values with 0 mean and unit standard deviation.  
- Subtracting the min and dividing by max minus min which results in values between 0 and 1. 

Normalization is also necessary if you want to reliably analyze the influence of particular features on the target, e.g how much does the square footage feature influence the sale price. 

#### Regularization
The goal of our linear regression model is to make predictions on inputs we haven't seen before. In order for the model to generalize well it needs to not overfit to the training data. One common strategy to prevent overfitting is to add regularization terms to the loss function. The two most common forms of regularization are:
- L1 regularization (also known as LASSO) is where we add a term to the loss of the form \(\lambda \lVert w \rVert_{1}\). \(\lambda\) is a hyper-parameter that determines how much regularization we want to add and the L1 norm itself is a sum of the absolute values of the weight coefficients. L1 regularization has the effect of moving coefficients to zero (encouraging sparsity) which can be helpful for feature selection where we want to remove features that are not relevant for predicting the target.
- L2 regularization (also known as Ridge) amounts to adding a term to the loss of the form \(\lambda \lVert w \rVert_{2}\). The L2 norm, which is equivalent to the magnitude of the weight vector, encourages weight coefficients to be smaller and is especially sensitive to larger weights since they become squared in the norm. In practice it is more convenient to use the squared L2 norm which is easier to differentiate and doesn't require computing a square root.
- You can also use a combination of L1 and L2 regularization known as [elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization) regularization.

As an important aside, the reason that L1 regularization moves weights to 0 while L2 moves weights to be smaller has to do with their respective gradients. For L1 the gradient with respect to the weights is a constant (\(1\) or \(-1\)) where as for squared L2 it depends on \(w\). Therefore during gradient descent L1 will update the weights by a constant amount during each update where as with L2 the update to the weights become smaller and smaller as the weight itself is smaller making it harder to reach 0. A nice illustration of this idea is [here](https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models).

#### Multicollinearity
A topic that comes up with respect to linear regression is [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity). We saw that if we have features that are a linear combination of other features then we can't apply the closed form solution to linear regression. When features have a *nearly* linear relationship there is still the problem that the estimate of weights is unstable (e.g weight values are very sensitive to small changes in the data). This makes sense because if you have two variables that are interchangeable (like degrees in Fahrenheit or Celcius) then how do you decide which one to weigh more when either one will do. 

One way to be robust to multicollinearity is to use [regularization](#regularization) which can encourage redundant features either be dropped (L1) or for their importance to be more evenly distributed by penalizing large weights (L2).

#### Quality of Fit 
Earlier we saw that the sum of squared errors is what guided the optimization of the linear regression model parameters. It's natural then to think about the quality of the fit (e.g how well does the model fit the data) in the same terms. A key problem with this approach is that sum of squared errors is hard to interpret as the value depends on the number of data points and the scale is based on squared values. 

This leads us to several better alternatives:
- Mean Squared Error is an improvement as it takes into consideration the number of data points.
- Root Mean Squared Error is more interpretable because it will be on the same scale as the target. So if the target is height in inches then the error will also be in inches.
- Mean Absolute Error is also on the same scale as the target and is less sensitive to outliers.
- Finally, a commonly reported measure in linear regression is the \(R^2\) value which is also known as the [coefficient of determination](https://online.stat.psu.edu/stat500/lesson/9/9.3). The measure is between 0 and 1 and represents the proportion of variation in the target variable that is explained by the model. 

#### Polynomial Regression 
While we usually think about linear regression in terms of fitting a line or hyperplane to a set of points, the linear regression machinery can be used to fit non linear models so long as they are linear with respect to the parameters. The primary example of this is in polynomial regression where we model the target as: 

$$
y = w_{0} + w_{1}x + w_{2}x^{2} + \dots + w_{n}x^{n}.
$$ 

In this case we can use linear regression to fit an nth order polynomial to a set of points. The Wikipedia on polynomial regression has a visual [example](https://en.wikipedia.org/wiki/Polynomial_regression). 

#### Conclusion
While linear regression is considered to be one of the simplest kinds of machine learning models we see it is anything but simple. In this post I've presented what linear regression is, what loss is being optimized, how the optimization is performed and some of the nuances to be aware of before applying it in practice. 

Largely, I have focused on linear regression through the lens of ML but I wanted to mention that there are also various statistical framings of linear regression. In my opinion the statistical view is more difficult to understand. One connection between the two views that I wanted to point out is that if you [work out](https://statproofbook.github.io/P/slr-mle.html) the maximum likelihood estimate for linear regression you end up with a term in the maximization that is identical to the sum of squared errors loss (within a constant factor)!  

If you're looking for a good starting point to play with linear regression consider the scikit library. The library documenation shows how to use [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) and also points to linear regression functions that incorporate the different kinds of [regularization](#regularization) we discussed.

Stay tuned for a follow up post on logistic regression which is the cousin of linear regression.


#### References
A few misc references that I consulted in addition to all the ones linked above:
- [Stack Exchange](https://stats.stackexchange.com/questions/278755/why-use-gradient-descent-for-linear-regression-when-a-closed-form-math-solution#:~:text=If%20you%20apply%20mini%20batch,lot%20of%20time%20on%20calculations)
- [CMU Slides](https://www.cs.cmu.edu/~mgormley/courses/10601-f21/slides/lecture8-opt.pdf)
- [Matrix Calculus](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
