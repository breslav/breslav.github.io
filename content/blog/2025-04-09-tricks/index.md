---
date: "2025-04-09T00:00:00Z"
tags:
title: Vanishing and Exploding Gradients
---

Let's start with a brief history lesson. If we go back in the [history of neural networks](https://en.wikipedia.org/wiki/History_of_artificial_neural_networks) we can see that backpropagation was developed in the 1970s and popularized in the 1980s. Despite the brilliance of backpropagation, deep neural networks (DNNs) were very difficult to train. In 1991 Sepp Hochreiter analyzed the vanishing gradient problem which was a major hurdle for training DNNs. Fast forward to today and you'll notice that training DNNs involves using a number of tricks many of which evolved to tackle vanishing (and exploding) gradients. 

In this blog post I want to summarize the problem of vanishing and exploding gradients and summarize some of the (imperfect) tricks for mitigating them. 


## Vanishing Gradients
As mentioned, one of the barriers to effectively training DNNs in the early days was the problem of vanishing gradients. Vanishing gradients refers to the phenomenon that as backpropagation proceeds you can end up with gradients that are so small that they effectively have no impact on the weights that they correspond to. If many weights are barely able to change (particularly at earlier layers) this can lead to poor or unstable training. In essence the network is unable to learn effectively and training will result in a poor solution. 

Let's examine how vanishing gradients can arise:
- During backprop we are interested in updating each weight (and bias) so that it reduces the loss. This means we need to compute the gradient of the loss with respect to each weight (and bias).
- When we want to compute this gradient for some intermediate weight in the network we will employ the chain rule. In general the chain rule will result in a sum of products, e.g imagine a neuron with two paths of gradients flowing into it. 
- If you have a product with many terms and those terms are relatively small then you will end up with a really small value. This is exactly what can happen when you are calculating the gradient for weights in earlier layers of the network. In other words the gradient begins to vanish!

It turns out that the vanishing gradient problem can be made *worse* by the activation function used as well as the magnitude of the weights in the network. Let's see why this would be.

### Impact of Activation Function and Weights
If we zoom into the chain rule we can make a few observations. Let's consider that in some early layer of our network we have an activation function \(\theta\) which takes in as input \(w_1x_1 + w_2x_2 + b\), giving us the function: 

$$
y = \theta(w_1x_1 + w_2x_2 + b)
$$

During backprop we would use the chain rule to compute the gradient that will be used to update the weights and bias. Let's focus on \(w_1\) in this example.

$$
\frac{\partial{L}}{\partial{w_1}} = \frac{\partial{L}}{\partial{y}} \frac{\partial{y}}{\partial{w_1}}  
$$

The second term in this product of gradient becomes:

$$
\frac{\partial{y}}{\partial{w_1}} = \theta'(w_1x_1 + w_2x_2 + b) x_1
$$

We now see that this gradient depends on the derivative of our activation function (at some input value) and it also depends on the value of one of the inputs (\(x_1\)) into the neuron. If the derivative of our activation functions produces small gradients then that would be one way our gradient is pushed to be small. Additionally if the weights of earlier layers are very small that could result in the input to our neuron also being very small. If both of these factors occur we now have a product of two small values which is even smaller. These problems compound with many layers and as we go further back in the network (to earlier layers)!  


### Tricks 

Several tricks have been proposed to alleviate the problem of vanishing gradients.

- At some point the ReLU activation function was proposed which largely replaced activation functions like sigmoid (and tanh). The sigmoid function has a gradient which approaches 0 for both very small and very large inputs. With our analysis above we can now see how the gradient of the activation function can impact other gradients. ReLU on the other hand has a constant derivative of 1 for all positive values, which means that very small gradients are eliminated. (ReLU is still problematic when the input is negative leading to a gradient of 0). 

- Skip connections are another trick that in a literal sense side step the problem of vanishing gradients. In particular a skip connection results in a function that looks like \(y = f(x) + x\). During backprop when we are interested in \(\frac{\partial{y}}{\partial{x}}\) we can see that regardless of how much the gradient vanishes due to \(f(x)\), we still have a gradient of 1 added to our vanishing gradient. In otherwords this allows the gradient from layers closer to the loss to continue flowing to earlier layers.

- Weight initialization schemes are another trick that can help with vanishing gradients. Schemes like [Glorot/He initialization](https://www.youtube.com/watch?v=ScWTYHQra5E) propose uniform/Gaussian distributions from which to sample weight values. The shape of these distributions also depends on the fan-in and/or fan-out of the layer. At a high level these schemes help keep activation values and gradients from getting too large or too small and have been shown to help with the vanishing gradient problem (see page 254 from the Glorot initialization [paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) for example). 

- Batch normalization (batch norm) can (at least in theory) help with vanishing gradients because it allows the network to learn to shift the distribution of values going into activation functions in such a way that it reduces small gradients. For example with sigmoid, batch norm could keep shift the input values so they are centered on the linear part of the sigmoid activation which would avoid small gradients. In practice I'm not sure whether it has actually been demonstrated that batch norm achieves this. [As an aside, the original batch norm paper makes several claims about why it works well but the latest [research](https://proceedings.neurips.cc/paper_files/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf) concludes that batch norm works well because it helps in generating a smoother loss surface].

## Exploding Gradients
A twin problem to vanishing gradients is known as exploding gradients. Exploding gradients refers to the phenomenon where gradients become too large during training which can lead to problems like numerical instability or difficulty in training. Exploding gradients can arise in similar ways as vanishing gradients. Recall that during backprop we can have gradient computations that involve the product of a large number of terms. If these terms are large, then we can end up with gradients that are enormous and potentially overflow. Even if gradients don't overflow they can result in the optimizer overshooting a good local minima, or oscillating, or even diverging.

As with vanishing gradients, the gradient computation can be sensitive to the magnitude of the weights in the network. Therefore we would like to avoid large weight values which could produce exploding gradients. 

### Tricks
Some of the tricks used to deal with vanishing gradients are also effective for dealing with exploding gradients. Particularly techniques that influence the scale of inputs to an activation or outputs of an activation (e.g weight initialization schemes and batch norm). 

Another simple way to prevent exploding gradients is with gradient clipping. One approach to gradient clipping is to clip components of a gradient that are larger than some value, but this has the downside of potentially making a drastic change to the direction of the gradient. A second approach is to clip the magnitude of the gradient thereby preserving the direction of the gradient (this can come at the cost of taking more time to converge).

## Conclusion
In this blog post we looked at how the problems of vanishing and exploding gradients arise when training DNNs. We also looked at several tricks that have been employed to mitigate these problems including:
- Weight initialization schemes
- Better activation functions like ReLU
- Skip connections
- Batch normalization 
- Gradient clipping

An important point to make here is that these tricks are far from perfect solutions and they generally have their own problems. Researchers will continue trying new ideas and eventually some or all of the current tricks will be replaced with new ones. 

A few examples of this taking place:

- ReLU has a gradient of 0 for inputs that are negative. This leads to the problem of dying neurons. Specifically, 
if the weights that influence a ReLU to be 0 are not updated then the neuron will forever output 0. More recently, alternative activation functions like leaky ReLU have been proposed to avoid this problem (leaky ReLU has a small negative gradient).

- Batch normalization, while effective for larger batch sizes, has been shown to be ineffective when batch sizes are small (e.g consider the distributed training setting). [Alternative normalizations](https://arxiv.org/pdf/1803.08494), like layer norm, have been proposed that avoid a dependency on batch size (layer norm normalizes each data point independently based on values in the layer).

- While Glorot and He initializations were popular it seems that simpler initialization schemes with fixed variance have also been successfully used (see [GPT2](https://huggingface.co/docs/transformers/en/model_doc/gpt2)).

### References
All references are linked inline!


