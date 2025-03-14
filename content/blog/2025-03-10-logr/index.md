---
date: "2025-03-10T00:00:00Z"
tags:
title: Reviewing Logistic Regression
---

After my post reviewing [linear regression]({{< ref "/blog/2025-03-03-lr" >}} "") it was only natural to review its relative logistic regression.
Logistic regression is another foundational model in machine learning and statistics. In this blog post I want to review the most important concepts associated with logistic regression (at least from a ML point of view). 

### Introduction 
Logistic regression is a popular regression model that is used to model the probability of some binary outcome. For example we can use it to model the probability that an email is spam or that a customer will buy a bicycle today. 

In addition to regression we can also view logistic regression as a way of performing binary classification. This is done by taking the probability of an outcome predicted by linear regression and applying a threshold to it. In other words we create a classifier with the following rule:
> If \(p(y=1) >= \tau\) then predict \(x\) as being a member of the class \(y=1\) otherwise predict \(x\) as being a member of the other class \(y=0\). (In practice we would select \(\tau\) by leveraging a validation set.)

### Model
Mathematically, logistic regression models the probability \(p\) of an outcome \(y=1\) as follows:  

$$
p(y=1) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x})}}
$$

Looking at this model you will notice we have a linear expression with weights \(\vec{w}\) being multiplied by the input features \(\vec{x}\). This product is then passed through the sigmoid function (which is also known as the logistic function). The role of the sigmoid function is to take any real valued number and transform it into a value between 0 and 1 which can be interpreted as a probability. 

As I alluded to earlier, I think of logistic regression as a relative/cousin of linear regression because it shares the linear term that is at the heart of [linear regression]({{< ref "/blog/2025-03-03-lr" >}} ""). 
>Recall the linear term is: \(y = w_{0} + w_{1}x_{1} + w_{2}x_{2} + \cdots + w_{n}x_{n}\) or equivalently \(y = \vec{w} \cdot \vec{x}\).


#### Log Odds Interpretation
An alternative way to interpret logistic regression is that it is modeling the linear relationship between input features and the log odds of the binary outcome. We can see this mathematically as follows:

$$
\begin{align}
odds = \frac{p(y=1)}{1-p(y=1)} \\
\log\;odds = \log\left(\frac{p(y=1)}{1-p(y=1)}\right) \\
 = \log(p(y=1)) - \log(1-p(y=1)) \\
 = -\log(1 + e^{-(\vec{w} \cdot \vec{x})}) - \log\left(\frac{e^{-(\vec{w} \cdot \vec{x})}}{1 + e^{-(\vec{w} \cdot \vec{x})}}\right) \\
 = -\log(1 + e^{-(\vec{w} \cdot \vec{x})}) - \log(e^{-(\vec{w} \cdot \vec{x})}) + \log(1 + e^{-(\vec{w} \cdot \vec{x})}) \\
 = - \log(e^{-(\vec{w} \cdot \vec{x})}) \\
 = \vec{w} \cdot \vec{x} \\
\end{align}
$$

I find this interpretation of logistic regression harder to digest than that of linear regression. For one thing log odds might not be all that intuitive unless you like log and are a gambler. 

Let's try and make it less abstract. Odds comes up in betting and if two teams are equally favored to win then the odds ratio would be 1/1. If one team has a \(0.75\) probability of winning and the other team has a \(0.25\) probability of winning that would correspond to an odds ratio of 3/1. Ignoring log for now we have some notion that the probability of some binary outcome (e.g this team wins the game) is linear in the features. Presumably for sports we would have features like how winning the team is and we would expect the more winning it is the higher the odds (and log odds) of it winning are.

### Optimization

Once again we have to ask the question: how do we train/fit the model on/to our data \(\{(\vec{x_{i}},y_{i})\}\), where \(y_{i}\) is the binary valued outcome. In order to fit or train the model we need to find the values of the parameters \(w\) that optimize some loss/objective. In logistic regression the commonly chosen objective is minimizing the cross-entropy loss (also referred to as the log loss). That is we want to find \(w\) that minimizes: 

\[ Loss = -\sum_{i=1}^{m} \Bigl(y^{i}\;\log(p_{y^{i}=1}) + (1-y^{i})\;\log(1-p_{y^{i}=1})\Bigr) \] 
Where \(m\) is the number of training data points we have.

We can intuitively understand the loss by looking at what loss value is produced for the different combinations of true outcome and predicted probability. 
- Label \(y=1\),  model probability \(p(y=1)=1\) --> \(Loss= 0\) 
- Label \(y=0\), model probability \(p(y=1)=0\) --> \(Loss= 0\) 
- Label \(y=1\), model probability \(p(y=1)=0\) --> \(Loss=+\infty\)
- Label \(y=0\), model probability \(p(y=1)=1\) --> \(Loss=+\infty\)  

In summary, as the predicted probability trends farther away from the label the loss shoots up towards positive infinity. As the predicted probability trends closer to the label the loss approaches 0. Hence we want to find parameters that minimize the loss on our dataset!

#### Cross Entropy From MLE
Another way to motivate the use of the cross-entropy loss is to view it as emerging from a maximum likelihood estimation (MLE) derivation. In particular for logistic regression we want to find the model parameters \(\theta\) that maximize this conditional likelihood term:

$$
\begin{align}
\underset{\theta}{\operatorname{argmax}} p(\vec{y}|\vec{x}) \\
= \underset{\theta}{\operatorname{argmax}} \prod_{i=1}^{m} p(y_i|x_i) \\
= \underset{\theta}{\operatorname{argmax}}\: \log\left(\prod_{i=1}^{m} p(y_i|x_i)\right) \\
= \underset{\theta}{\operatorname{argmax}} \sum_{i=1}^{m} \log p(y_i|x_i) \\
= \underset{\theta}{\operatorname{argmin}} -\sum_{i=1}^{m} \log p(y_i|x_i) \\
= \underset{\theta}{\operatorname{argmin}} -\sum_{i=1}^{m} \log\left(p^{y_i}(1-p)^{1-y_i}\right) \\
= \underset{\theta}{\operatorname{argmin}} -\sum_{i=1}^{m} \log\left(p^{y_i}\right) + \log\left((1-p)^{1-y_i}\right) \\
= \underset{\theta}{\operatorname{argmin}} -\sum_{i=1}^{m} y^i\,\log(p) + (1-y^i)\,\log(1-p) \\
\end{align}
$$

Looking at the last term of this derivation we see that maximizing the likelihood is equivalent to minimizing the cross-entropy loss we introduced above. Here \(p\) represents the probability our logistic regression model assigns to the \(i\)-th data point being a member of the binary class we associated with \(y=1\).  

A few notes on the math:
- Usually maximum likelihood estimation (MLE) uses a likelihood term which is defined as the probability of observing our data given the class, e.g \(p(\vec{x}|\vec{y})\). However, in the case of logistic regression the common MLE derivation uses a different definition of likelihood which is referred to as a conditional likelihood. I don't yet fully understand why this definition is used, I think it may have to do with the fact that in this problem it's easier to model this conditional than the standard likelihood. 
- The initial conditional probability is decomposed into a product over samples because we assume our data samples are independent. 
- In this optimization we have a product of probabilities which are always positive and therefore we can transform our optimization with log without changing the solution. 
- We turn a maximization objective into an equivalent minimization problem by adding a negative.
- We replace our conditional probability with the [Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution) distribution which models the probability of a particular binary outcome like obtaining a heads when flipping a coin. This makes sense since logistic regression is modeling the probability of some binary outcome which we can also think of as a coin toss. Some other derivations I've encountered start w/ this substitution. 

#### Gradient Descent Solution 

Gradient descent is an extremely powerful tool and so perhaps unsurprisingly we once again look to gradient descent to help us find the parameters \(w\) that minimize the cross-entropy loss. In order to apply gradient descent we need the gradient of the loss with respect to the parameters. 

To help us with the calculation of the derivative let's first look at some intermediary results. We begin with the derivative of the sigmoid function which our loss depends on.

$$
\begin{align}
\sigma(z) = \frac{1}{1 + e^{-z}} \\
\frac{\partial}{\partial z} \sigma(z) = \frac{(1 + e^{-z})0 - 1(-e^{-z})}{(1+e^{-z})^2} \\
\frac{\partial}{\partial z} \sigma(z) = \frac{e^{-z}}{(1+e^{-z})^2} \\
\frac{\partial}{\partial z} \sigma(z) = \frac{e^{-z}}{(1+e^{-z})} \frac{1}{1+e^{-z}} \\
\frac{\partial}{\partial z} \sigma(z) = \frac{e^{-z}}{(1+e^{-z})} \sigma(z) \\
\frac{\partial}{\partial z} \sigma(z) = (1-\sigma(z)) \sigma(z) \\
\end{align}
$$

Now let's redo the above using our linear expression \(w \cdot x\) plugged in for \(z\):

$$
\begin{align}
\sigma(w \cdot x) = \frac{1}{1 + e^{-w \cdot x}} \\
\frac{\partial}{\partial w} \sigma(w \cdot x) = \frac{(1 + e^{- w \cdot x})0 - 1(-xe^{-w \cdot x})}{(1+e^{-w \cdot x})^2} \\
\frac{\partial}{\partial w} \sigma(w \cdot x) = \frac{xe^{-w \cdot x}}{(1+e^{-w \cdot x})^2} \\
\frac{\partial}{\partial w} \sigma(w \cdot x) = \frac{xe^{-w \cdot x}}{(1+e^{-w \cdot x})} \frac{1}{1+e^{-w \cdot x}} \\
\frac{\partial}{\partial w} \sigma(w \cdot x) = \frac{xe^{-w \cdot x}}{(1+e^{-w \cdot x})} \sigma(w \cdot x) \\
\frac{\partial}{\partial z} \sigma(w \cdot x) = x(1-\sigma(w \cdot x)) \sigma(w \cdot x) \\
\end{align}
$$

Now let's leverage the above gradient to help us with the derivation of the gradient of the cross-entropy loss.

$$
\begin{align}
Loss =  -\sum_{i=1}^{m} y^i\,\log(p) + (1-y^i)\,\log(1-p) \\
Loss =  -\sum_{i=1}^{m} y^i\,\log(\sigma(w \cdot x)) + (1-y^i)\,\log(1-\sigma(w \cdot x)) \\
\frac{\partial Loss}{\partial w} =  -\sum_{i=1}^{m}  \frac{\partial}{\partial w} \Bigl(y^i\,\log(\sigma(w \cdot x)) + (1-y^i)\,\log(1-\sigma(w \cdot x))\Bigr) \\
\frac{\partial Loss}{\partial w} =  -\sum_{i=1}^{m}  \Bigl(y^i\,\frac{x(1-\sigma(w \cdot x)) \sigma(w \cdot x)}{\sigma(w \cdot x)}\Bigr) + \Bigl((1-y^i)\,\frac{-x(1-\sigma(w \cdot x)) \sigma(w \cdot x)}{(1-\sigma(w \cdot x))} \Bigr) \\
\frac{\partial Loss}{\partial w} =  -\sum_{i=1}^{m}  \Bigl(y^i\,x(1-\sigma(w \cdot x))\Bigr) + \Bigl(-(1-y^i)\,x\sigma(w \cdot x) \Bigr) \\
\frac{\partial Loss}{\partial w} =  -\sum_{i=1}^{m}  \Bigl(y^i\,x - y^i\,x\sigma(w \cdot x)\Bigr) + \Bigl(-\,x\sigma(w \cdot x) + y^i\,x\sigma(w \cdot x) \Bigr) \\
\frac{\partial Loss}{\partial w} =  -\sum_{i=1}^{m}  \Bigl(y^i\,x  -\,x\sigma(w \cdot x)  \Bigr) \\
\frac{\partial Loss}{\partial w} =  -\sum_{i=1}^{m}  x(y^i -\,\sigma(w \cdot x))
\end{align}
$$

As a sanity check, our derivation result matches the result from these Stanford [slides](https://web.stanford.edu/~jurafsky/slp3/5.pdf). Hooray! Additionally, the gradient for an individual weight \(w_j\) would be the same as above but with the multiplier being \(x_j\). So semantically, the gradient depends on the difference between the true class (represented with a probability of 1 or 0) and the predicted probability of belonging to class 1. This difference is then scaled by the input \(x_j\).


### Other Nuances
Like in our discussion of nuances in [linear regression]({{< ref "/blog/2025-03-03-lr#normalization" >}} ""), there are several things you want to keep in mind before applying logistic regression. Specifically, we still need to normalize our data and adding regularization is a good idea. 

#### Quality of Fit 
While there are some notions of evaluating the quality of the fit of the model from a regression point of view, the most practical way to evaluate a logistic regression model is through the lens of classification.

Broadly speaking that means we would make predictions on some validation set and formulate a \(2 \times 2\) confusion matrix which can then be used to generate a variety of metrics like accuracy, precision, recall etc. We can also look at tradeoffs between metrics (like precision and recall) when we vary the threshold used for classification. 

#### Multinomial Logistic Regression 
Logistic regression can also be extended to the case of multiple classes which is known as multinomial logistic regression. In this case the model is predicting a proability distribution over more than two classes. As a result instead of a single weight vector \(w\) we would have a weight matrix \(W\) of \(k \times n\) dimensions, where \(k\) is the number of classes and \(n\) is the dimension of the input (plus 1 for a bias term). We then compute \(Wx\) which produces a \(k\) dimensional vector representing un-normalized scores for each class. 

These un-normalized scores (also referred to as logits) are fed through the softmax function in order to get a probability distribution (probabilities sum to 1). 

$$
softmax(y_i) = \frac{e^{y_i}}{\sum_{i=1}^{k} e^{y_i}}
$$

If \(y_i\) is the un-normalized score for class \(i\) then the softmax function converts it to a probability value. **Note:** Regardless of whether the un-normalized score is positive or negative if we exponentiate it, we will get a positive value. Since the denominator is just a sum of the numerators for each class we are left with a probability distribution in the end. 

As with logistic regression, multinomial logistic regression also uses the cross-entropy loss in optimization. The cross-entropy loss generalized to \(k\) classes is as follows: 

$$
Loss =  -\sum_{i=1}^{m} \sum_{j=1}^{k} y_j^i\,\log(p_j) \\
$$ 

Notice that for any particular data point \(y_j\) will be 1 when \(j\) matches the class of the label and 0 otherwise. Therefore the loss for \(y_j\) only depends on the log of the probability predicted for that same class. Once again the closer this probability is to 1 the closer the loss is to 0 and the closer this probability is to 0 the closer the loss approaches positive infinity. 

We can also notice that our ordinary cross-entropy loss for the binary case is just a special case of this loss where \(k=2\).  

#### Conclusion
Logistic regression is not only a classic ML (and stats) model but it is also one of the first models to try when working on a binary classification problem. Additionally, a lot of the ingredients in logistic regression (cross-entropy loss, gradient descent, linear combinations of features) show up in neural networks which means learning about logistic regression can help you to understand neural networks. 

Finally, we see that scikit has the following [API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for logistic regression. As an aside it looks like scikit implements alternative optimization algorithms that don't include gradient descent. The default is [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) which is a quasi-Newton method that appears to converge faster than gradient descent. The API also defaults to using L2 regularization.

#### References
A few misc references that I consulted in addition to all the ones linked above:
- [CMU Lecture Slides](https://web.stanford.edu/~jurafsky/slp3/5.pdf)