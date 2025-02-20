---
date: "2025-02-20T00:00:00Z"
tags:
title: Catching Up On Vision (Round 2)
---

While starting to read the [DINO]() paper, I realized it wasn't going to make much sense without more background. To fill in on some missing background I decided to skim the [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/pdf/2006.07733) paper.

#### Bootstrap Your Own Latent (BYOL)

The goal of [BYOL](#references) (published in 2020) is to learn a "good" image representation without the need for labels. We would define a representation to be "good" if it can be used to perform many downstream tasks well. Traditionally models like CNNs would learn image representations through supervised training, e.g by learning to classify images in ImageNet. If instead a model could learn in a self- supervised way, that would open up the door to using much larger datasets and in turn we would expect the model to learn better representations. 

The way BYOL works is depicted in the figure below and it involves the following steps: 
- Take an image \(x\) and perform two different data augmentations on it, leading to \(v\) and \(v'\).
- Input each augmented image into two different neural networks, one having parameters \(\theta\) which is part of the "online" network and the other \(\xi\) which is part of the "target" network. In the paper these neural networks are ResNet CNNs.
- The online network has two additional MLPs that follow the ResNet. The output of the online network can be thought of as some function of \(z_{\theta}\), where \(z_{\theta}\) is a lower dimensional representation of the first augmented image \(v\).
- The target network has one additional MLP that generates a lower dimensional representation of the second augmented image \(v'\) called \(z'_{\xi}\).
- Now the goal is for the two networks to output the same vector which is what the loss function is designed to encourage. There are also several interesting quirks about training:
  - Only the online network is trained. In other words each training step only modifies the weights \(\theta\).
  - The target network weights are taken as an exponential moving average. In other words the target network weights are updated to be a combination of itself and the latest online weights.
- Lastly, after training most of the architecture is thrown away. The only part that is kept is the now trained ResNet which hopefully has learned to produce a "good" image representation.
 
![BYOL Architecture](/images/byol.png)

My high level understanding of this model is that it learns that different augmentations of an image don't change the underlying meaning of the image. If we learn a good representation for the original dog image then we would expect the two augmentations to be related by a relatively simple function (represented by \(q_{\theta}\)). 

What I found more challenging to wrap my head around is that the target network is based on the online network. Initially it may more or less be outputting some random vector which the online network is being trained to match. It sort of feels like as the online network gets better at representation it can better learn from the different augmentations to improve more and so you have this positive feedback loop.  

Another interesting quirk of this paper is that there is nothing that obviously prevents the networks from learning a trivial solution to the problem like always output the same vector for every image which would guarantee a low loss (this is known as mode collapse). The authors say that in practice this problem does not arise. 

For this paper I once again recommend the commentary from [Yannic Kilcher's](#references) YT Video which helpded my understanding of the paper. He also talks about earlier works for self-supervision which relied on using negative samples (e.g augmentations of some object that is different than the input) to help avoid mode collapse and how it's a big deal to not need those negative samples.











#### Conclusion & Lingering Questions

#### References
- [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/pdf/2006.07733)
- [Yannic Kilcher's YouTube Video](https://www.youtube.com/watch?v=YPfUiOMYOEE)


