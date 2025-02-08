---
date: "2025-02-07T00:00:00Z"
tags:
title: Brief Notes on Attention Efficiency
---

As part of my ongoing review of LLMs, I revisited the core computation performed during self attention. Like in my previous reviews, I focused on the
idea of there being three important learnable projections that map our token embeddings to queries, keys, and values which are then used to re-represent (add context) the token embeddings. One aspect of attention that I glossed over in the past is the efficiency of this computation.  

<!-- In this blog post I aim to make note of a few briefly stopic the efficiency of attention and some of the techniques that have come out to make it more efficient.  -->

### How Efficient is Self Attention?
Looking at the [Multi-Query Attention](#references) paper we can see that they report multi-headed attention as having a time complexity of \(\Theta(bnd^{2})\) and a memory complexity of \(O(bnd + bhn^{2} + d^{2})\). Our goal in this section is to see if this makes sense at a high level.

Let's start by reviewing the time complexity component and I'll make a few simplifying assumptions:
- Assume the Q,K,V projection tensors map our \(d\) dimensional vectors into \(k\) dimensions where \(k = d/h\).
- Assume that the number of attention heads \(h\) is a small constant and therefore can be ignored in the analysis. 

Therefore:
- We apply three projections to our (batched) input which has shape \([b,n,d]\), which results in applying \(b\) projections of size \(d x k\) for each of \(n\) tokens. So \(dk\) multiplications for a single token and so roughly \(bndk\) for all tokens, across all batches. Since \(k\) is \(d\) scaled down by \(h\) we get \(bndd/h\). Then ignoring the small constant \(h\) we get \(bnd^{2}\).
- Next we need to multiply the queries and keys together, put each row through a softmax which we ignore for now, and then multiply the attention weights by the values. So \([b,n,k]\) multiplied by \([b,n,k]\) involves \(bkn^{2}\) multiplications and \([b,n,n]\) multiplied by \([b,n,k]\) also involves \(bkn^{2}\) multiplications.
- The sum of these two is \(bnd^{2} + bkn^{2}\) and with \(k \thickapprox d\) (ignoring small constant \(h\)) we would have \(bnd^{2} + bdn^{2}\). 

Now we see that the paper only contains the first term and ignores the second. Why might this be? My guess is that there is an assumption being made that \(d >> n\) in which case the first term dominates the second term. On the other hand it's not obvious that this is a good assumption today where context lengths have grown. 

Another confusion I've had is seeing the complexity of attention commonly being reported as quadratic in \(n\), while my computation above shows there is also the first term which is quadratic in \(d\). Then I realized that the commonly reported complexity is only considering the computation of the attention weights and applying them to the values. Instead my starting point was based on the above paper which also includes the projection step in its computation. So my conclusion is that the overall work needed is quadratic both in \(d\) and \(n\).

#### Memory Complexity



#### Lingering Questions
Some questions that came to mind during my research:
- If you add positional embeddings to word embeddings how do you expect the model to separate the two signals back out?
- Is it some strange coincidence that sinusoids are at the heart of both of these papers/techniques? Is it just that they are a great way to encode values?
- Why exactly do we want frequency/angles of sinusoids to vary as a function of dimension?

#### References
These are the references I found to be helpful:  
- [Multi-Query Attention Paper](https://arxiv.org/pdf/1911.02150)






