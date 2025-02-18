---
date: "2025-02-18T00:00:00Z"
tags:
title: Catching Up On Vision
---

When I began my PhD in computer vision the leading approaches of the day were models like support vector machines, probabilistic graphical models, and decision trees. Fast-forward to 2016 when I was graduating and the deep learning revolution was underway. Leading approaches were convolutional neural networks like [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [VGG](https://arxiv.org/pdf/1409.1556), and [ResNet](https://arxiv.org/pdf/1512.03385). 

I had the fortune to play with CNNs for the final project of my PhD.  Specifically, I experimented with finetuning a VGG-16 CNN on a very small dataset for the task of pose estimation. The [results](https://arxiv.org/pdf/2010.11929) weren't very good which I think was due to a combination of having a very limited training set and targeting a domain too different from the source domain. I was working with grayscale images of moths captured in a laboratory which looked nothing like the [ImageNet](https://www.image-net.org) images used to train the base model.  

Since graduating and entering industry my career moved away from computer vision and so I stopped attending (LLM pun intended) to where the field was headed. 

In this blog post I am taking a tiny step towards understanding some of the key advancements in the field of computer vision *today*. A natural starting point is to ask what role do transformers play in computer vision? 

#### Vision Transformer
In research it's very common to look at successful approaches in one area and try them out in another. So it's no surprise that following the success of transformers in NLP some Google researchers applied them to the task of image classification. The now well cited [Vision Transformer](https://arxiv.org/pdf/2010.11929) paper showed that a transformer could outperform CNNs on image classification while using less compute.  

Zooming in a little, we see that in order to apply a transformer to an image we need some way to turn it into a sequence of vectors. The approach taken in this paper is to slice up an image into small patches (e.g \(16\times16\) pixels.), flatten each patch, and then order them from top to bottom and left to right. Then a learned linear projection is applied to each image patch to produce embeddings. Like in the original transformer paper, positional embeddings are added to the input (image) embeddings. The resulting vectors are fed into a transformer encoder. 

In order to perform classification a classification head was added to the transformer encoder along with a classification token (learnable embedding) at position 0. During training, large labeled image datasets (like ImageNet and successors) were used with a standard classification loss to train the model (instead of self-supervision). The authors report that when fine-tuning for specific tasks the pre-trained prediction head is replaced by a zero initialized feed forward layer. 

Another important point this paper made is that transformers didn't do better than CNNs on smaller datasets like the original ImageNet (~1M images). It's wasn't until they used a ~300M image dataset that the transformer outperformed. One reason why transformers may need to see a lot more data before they are competitive with CNNs is that they don't have the same inductive bias. CNNs are designed to learn patterns (filters) that are translationally invariant. This is useful because what makes a face a face has more to do with local structure than where a face appears globally in an image. Transformers on the other hand allow for an image patch to attend to any other image patch globally and so it would take longer for the model to learn that when you see a cat eye you should be paying attention to patches near by as they are the most salient. A natural question is how much relative positional embeddings would have helped in this regard. 











#### Conclusion & Lingering Questions

#### References
These are the references I found to be helpful:  






