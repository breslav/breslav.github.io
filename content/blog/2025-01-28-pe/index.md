---
date: "2025-01-28T00:00:00Z"
tags:
title: Positional Embeddings are Strange 
---

Recently I've been reviewing the "basics" of large language models and decided to finally peek into the details of positional embeddings which I had ignored in the past. In this post I want to share what I've learned from reviewing this topic.  

#### Positional Embedding Motivation
In the foundational [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, positional embeddings are introduced as a way to add ordering information to token embeddings so that the transformer model has some way of understanding the order of the tokens. To state the somewhat obvious, we want language models to understand word order (and by extension token order) because word order impacts the semantics of what is being said. 

A question arises: 
> Why does the transformer model needs some additional mechanism to understand the order of words, aren't we already feeding the words in to the model in an ordered way?

The reason is that the transformer architecture is based on self-attention which produces the same set of output vectors regardless of input order. Conceptually, if you are producing the same set of vectors for different word orderings then how can you differentiate between the different meanings (or lack of meaning) represented by different permutations of a sentence. So in short the self-attention mechanism is why transformers need some way of encoding the order of input tokens. 

#### Sinusoidal Positional Embeddings
In the foundational paper mentioned above, the authors encode the absolute position of a token by constructing a \(d\) dimensional vector composed of a series of sine and cosine waves of varying frequencies. The sine or cosine waves used for some dimension \(i\) will have an angular frequency value that also depends on \(i\). The absolute position of the token in the input sequence is then used to evaluate the sinusoid at a specific point in time leading to a concrete value for that dimension. 

To lessen my confusion I took to Google and found several helpful blogs and videos which I will link [below](#references). There were a few concepts that I found interesting and helpful in gaining at least some intuition for these sinusoidal embeddings and I wanted to put them in my own words here. 

- The first concept comes from a thought experiment where we consider representing the position in binary. As an example, the 8th position or number 8 is represented in binary as 1000, or 01000 if using more bits. Since we can add an arbitrary number of 0's to the more significant bits we have found a way to encode a position into \(d\) binary values or equivalently a \(d\) dimensional vector. Additionally, if we consider neighboring positions, e.g counting up from 0 to 1 to 2 and so forth, we will notice that the least significant bit will flip between 0 and 1 with the highest frequency and as we go to more significant bits they flip with less frequency. 

- If we now look at binary as encoding a number with a collection of square waves of differening frequencies, then it becomes much easier to interpret sinusoidal embeddings as being very similar but with sinusoids (smooth and continuous) instead of square waves. 

- The second concept I picked up is the idea that it would helpful if positional embeddings also encoded something about relative distances (not just absolute positions). It turns out that by using sine and cosine to encode position you get the property that the encodings of different positions are related to each other through rotations. This then implies that if you have two positional embeddings and you examine the rotation between them you can say something about the relative distance between the positions they encode. Yet the embeddings themselves encode absolute positions. 

- The third concept which I don't really follow is the notion that it is beneficial to have different frequencies represented in different dimensions of the embedding vector as it somehow helps the model handle shorter range and longer range dependencies between tokens. The best explanation I've put together so far is that using different frequencies is just how the math works out, similar to how representing numbers in binary can be thought of as using square waves of various frequencies. 



<!-- In practice I'm not sure how this relative distance would be decoded since the positional embeddings are added to token embeddings and so it would seem like  -->

<!-- In the context of self-attention it makes sense that we may want to generally attend more to nearby tokens than ones far away, and so it becomes advantageous to have an "easy" way of inferring relative token position.  -->


#### Rotary Position Embedding (RoPE)

Since the Attention Is All You Need paper is 8 years old as of this writing, I also wanted to get a sense of what state of the art looks like for encoding position. This led me to a popular paper published in 2023 titled [Rotary Position Embeddings (RoPE)](https://arxiv.org/pdf/2104.09864v5). RoPE has been used in models like LLama 3 from Meta. 

RoPE's approach to positional embeddings is derived from the objective of finding a function \(f\), that produces a dot product with a specific property. Specifically, if we have a word embedding vector \(x\) representing a token at position \(n\), and we have a word embedding vector \(y\) representing a token at position \(m\), then we would like their dot product to only depend on \(x\), \(y\) and their relative position \(m-n\).  

The paper shows that when \(f\) is chosen to be a rotation matrix the dot product satisfies the desired objective. To apply \(f\) to a \(d\) dimensional word embedding you would in theory construct a \(d \times d\) block diagonal rotation matrix, where the amount of rotation changes every 2 dimensions. In practice applying \(f\) is efficient because the matrix is very sparse and thus a full matrix multiply is not needed. 

As before, there were several [references](#references) that helped me better understand what RoPE is doing and here are my main takeaways.

- The main concept behind RoPE is that rotation provides a way of encoding relative positional information between vectors being dot producted. Again this property falls out from the underlying math. Essentially we have a dot product of two vectors that have a rotation applied to them, something like \(R_1x \cdot R_2y => x^{T}R_1^{T} R_2y\) which leads to \(x^{T}R_1^{-1}R_2y\) which involves a rotation that only depends on the *relative* difference between rotations.

- Another concept has to do with motivating why encoding relative positions is valuable. Consider the phrase "blue dog", where "blue" has position 1, and "dog" has position 2. Using RoPE we would apply some rotation (say 10 degrees) to the vector representing "blue" and some other rotation to "dog" (say 20 degrees) (assume 2D vector for simplicity). Then suppose the phrase has changed to "we painted a blue dog", now "blue" is in position 4 (say this corresponds to a 40 degree rotation) and "dog" is in position 5 (a 50 degree rotation). Since the dot product only depends on the difference in rotation and thus relative token positions, we have the nice property that the dot product of a key associated with "blue" and a query associated with "dog" will not have changed because their relative rotations (20-10 = 50-40) remained the same. If on the other hand we had used absolute embeddings then the dot product would have changed. 

#### Conclusion

My main takeaway is that researchers have identified interesting mathematical tricks that fufill the goal of allowing LLMs to understand the position of tokens (with a particular emphasis on *relative* position). Like much of the field of ML, the success of an approach is primarily driven by how well it works in practice. Questions like "why does this work so well?" and "does this really make sense?" often require additional research. As an example there is [this paper](https://arxiv.org/pdf/2410.06205) that re-examines RoPE. 

This post serves as my non-comprehensive partial understanding of this space and there are still many aspects [I don't fully understand](#lingering-questions).
In the interest of time, I'm moving on to reviewing the core attention mechanism of the transformer, but I think it's fair to say that positional embeddings are kind of strange. 

If you made it this far, thanks and check out the The Door's song [People Are Strange](https://en.wikipedia.org/wiki/People_Are_Strange) which partly inspired the title of this post.

#### Lingering Questions
Some questions that came to mind during my research:
- If you add positional embeddings to word embeddings how do you expect the model to separate the two signals back out?
- Is it some strange coincidence that sinusoids are at the heart of both of these papers/techniques? Is it just that they are a great way to encode values?
- Why exactly do we want frequency/angles of sinusoids to vary as a function of dimension?

#### References
These are the references I found to be helpful:  
- [Hugging Face Blog](https://huggingface.co/blog/designing-positional-encoding) provides a way to view different encodings as a natural progression.
- [Eleuther AI Blog](https://blog.eleuther.ai/rotary-embeddings/) focuses on rotary embeddings.
- [Amirhossein's Blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) focuses on explaining the use of sinusoidal functions.
- [Jia-Bin Huang's YT Video](https://www.youtube.com/watch?v=SMBkImDWOyQ) has nice visuals related to RoPE.
- [Efficient NLP YT Video](https://www.youtube.com/watch?v=o29P0Kpobz0) another video with nice visuals on RoPE.
- [RoPE Paper](https://arxiv.org/pdf/2104.09864v5)
- [Paper Re-examining Rope](https://arxiv.org/pdf/2410.06205) more recent analysis of RoPE.



<!-- 
using the book [Build A Large Language Model From Scratch](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167). In the past when I've reviewed LLMs I'd often refer to the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, but I didn't pay much attention to the positional embeddings aspect.  

 took a detour into positional embeddings.  -->



