---
date: "2025-01-28T00:00:00Z"
tags:
title: Positional Embeddings are Strange 
---

Recently I've been reviewing the "basics" of large language models and decided to finally peek into the details of positional embeddings which I had ignored in the past. In this post I want to share a few high level concepts that I've picked up from reviewing this topic.  

#### Positional Embedding Motivation
In the foundational [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, positional embeddings are introduced as a way to add ordering information to token embeddings so that the transformer model has some way of understanding the order of the tokens. To state the somewhat obvious, we want language models to understand word order (and by extension token order) because word order impacts the semantics of what is being said. 

Now, we should pause and ask: 
> Why does the transformer model needs some additional mechanism to understand the order of words, aren't we already feeding the words in to the model in an ordered way?

The reason is that the transformer architecture is based on self-attention which produces the same set of output vectors regardless of input order. Conceptually, if you are producing the same set of vectors for different word orderings then how can you differentiate between the different meanings (or lack of meaning) represented by different permutations of a sentence. So in short the self-attention mechanism is why transformers need some way of encoding the order of input tokens. 

#### Sinusoidal Positional Embeddings
In the foundational paper mentioned above, the authors used what appears (at least to me) to be a very weird choice for representing the absolute position of a token. The position is "encoded" into a d dimensional vector by evaluating a series of sine and cosine waves of varying frequencies. The sine or cosine wave being evaluated at some dimension i will have an angular frequency value that also depends on the dimension i. The absolute position of the token in the input sequence is then used to evaluate the sinusoid at a specific point in time leading to a concrete value for that dimension. 

To lessen my confusion I took to Google and found several helpful blogs and videos which I will link below. There were a few concepts that I found interesting and helpful in gaining at least some intuition for these sinusoidal embeddings and I wanted to put them in my own words here. 

- The first concept comes from a thought experiment where we consider representing the position (a number) in binary. As an example, the 8th position or number 8 is represented in binary as 1000, or 01000 if using more bits. Since we can add an arbitrary number of 0's to the more significant bits we have found a way to encode a position into d binary values or equivalently a d dimensional vector. Additionally, if we consider neighboring positions, e.g counting up from 0 to 1 to 2 and so forth, we will notice that the least significant bit will flip between 0 and 1 with the highest frequency and as we go to more significant bits they flip with less frequency. 

- If we now look at binary as encoding a number with a collection of square waves of differening frequencies, then it becomes much easier to interpret sinusoidal embeddings as being very similar but with sinusoids (smooth and continuous) instead of square waves. 

- The second concept I picked up is the idea that it would helpful if positional embeddings also encoded something about relative distances (not just absolute positions). It turns out that by using sine and cosine to encode position you get the property that the encodings of different positions are related to each other through rotations. This then implies that if you have two positional embeddings and you examine the rotation between them you can say something about the relative distance between the positions they encode. 

- The third concept which I don't really follow is the notion that it is beneficial to have different frequencies represented in different dimensions of the embedding vector as it somehow helps the model handle shorter range and longer range dependencies between tokens. The best explanation I've put together so far is that using different frequencies is just how the math works out, similar to how representing numbers in binary can be thought of as using square waves of various frequencies. 



<!-- In practice I'm not sure how this relative distance would be decoded since the positional embeddings are added to token embeddings and so it would seem like  -->

<!-- In the context of self-attention it makes sense that we may want to generally attend more to nearby tokens than ones far away, and so it becomes advantageous to have an "easy" way of inferring relative token position.  -->


#### Rotary Position Embedding (RoPE)

Since the Attention Is All You Need paper is relatively old at this point (8 years ago as of this writing), I also wanted to get a sense of what state of the art looks like for encoding position. This led me to a paper on [Rotary Position Embeddings](https://arxiv.org/pdf/2104.09864v5) published in 2023. 

RoPE's approach to positional embeddings is derived from the objective of finding a function f, that produces a dot product with a specific property. Specifically, if we have a word embedding vector x representing a token at position n, and we have a word embedding vector y representing a token at position m, then we would like the dot product to only depend on x,y and the relative position m-n.  

The paper shows that if f is chosen to be a rotation matrix, then the dot product will satisfy the objective above. For a d dimensional word embedding you would in theory construct a d by d block diagonal rotation matrix, where you have a different rotation amount for every 2 dimensions.  

As before, there were several resources that helped me better understand what RoPE is doing and here are my main takeaways.

- The main concept of RoPE is that using rotation leads to dot products that reflect relative positions between pairs of vectors. As an example, consider the phrase "blue dog", where "blue" has position 1, and "dog" has position 2. Using RoPE we would apply some rotation to the vector representing "blue" and some other rotation to "dog" (assume 2D vector for simplicity). Then suppose the phrase has changed to "we painted a blue dog", now "blue" is in position 4 and "dog" is in position 5. Since the dot product only depends on the difference in token positions, we have the nice property that the dot product of a key associated with "blue" and a query associated with "dog" will not have changed because their relative positions remained the same. If on the other hand we had used absolute embeddings then the dot product would have changed. 





#### Conclusion



#### References





<!-- 
using the book [Build A Large Language Model From Scratch](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167). In the past when I've reviewed LLMs I'd often refer to the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, but I didn't pay much attention to the positional embeddings aspect.  

 took a detour into positional embeddings.  -->



