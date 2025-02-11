---
date: "2025-02-07T00:00:00Z"
tags:
title: Brief Notes on Attention Efficiency
---

As part of my ongoing review of LLMs, I revisited the core computation performed during self attention. Like in my previous reviews, I focused on the
idea of there being three important learnable projections that map our token embeddings to queries, keys, and values which are then used to re-represent (add context to) the token embeddings. One aspect of attention that I glossed over in the past is the efficiency of this computation.  

<!-- In this blog post I aim to make note of a few briefly stopic the efficiency of attention and some of the techniques that have come out to make it more efficient.  -->

### How Efficient is Self Attention?
Looking at the [Multi-Query Attention](#references) paper we can see that they report multi-headed attention as having a time complexity of \(\Theta(bnd^{2})\) and a memory complexity of \(O(bnd + bhn^{2} + d^{2})\). Our goal in this section is to see if this makes sense at a high level.

Let's start by reviewing the time complexity component and I'll make a few simplifying assumptions:
- Assume the Q,K,V projection tensors map our \(d\) dimensional embedding vectors into \(k\) dimensions where \(k = d/h\). Here \(h\) is the number of attention heads.
<!-- - Assume that the number of attention heads \(h\) is a small constant and therefore can be ignored in the analysis.  -->

Therefore:
- We apply three projections (for each of \(h\) attention heads) to our (batched) input which has shape \([b,n,d]\). This results in applying \(bh\) projections of size \(d\,x\,k\) for each of \(n\) tokens. So \(dk\) multiplications for a single token and \(bhndk\) for all tokens, across all attention heads and all batches. Given our assumption this becomes \(bhndd/h\) which simplifies to \(bnd^{2}\).
- Next we need to multiply the queries and keys together, put each row through a softmax which we ignore for now, and then multiply the attention weights by the values. So \([b,h,n,k]\) multiplied by \([b,h,n,k]\) involves \(bhkn^{2}\) multiplications and \([b,h,n,n]\) multiplied by \([b,h,n,k]\) also involves \(bhkn^{2}\) multiplications. This is equivalent to \(bhn^{2}d/h = bdn^{2}\).
- Now summing these two terms yields \(bnd^{2} + bdn^{2}\). 

<!-- At this point in my analysis I became confused because the above paper only mentions the first term and ignores the second. Additionally, it is commonly said that attention is quadratic in \(n\) without mentioning anything about the first term.  -->

At this point we see a discrepency between my analysis and what the paper reports. The paper only contains the first term and ignores the second. Why might this be? My guess is that the paper assumes \(d >> n\) in which case the first term dominates and the second term can be ignored. While at the time of the paper that may have been a good assumption it's not obvious that it still holds as context lengths have grown larger and larger. 

Another confusion I've had is seeing the complexity of attention commonly being reported as quadratic in \(n\), while my computation above shows that we also have the first term which is quadratic in \(d\). This confusion was resolved when I realized that the commonly reported complexity is only considering the computation of the attention weights and applying them to the values. Instead my starting point was based on the above paper which also includes the projection steps in its computation. So while the attention computation is quadratic in \(n\) we must also consider the pre step of calculating the Queries, Keys, and Values which is qudratic in \(d\).

As for memory complexity, it's easier to see where it comes from by looking at the paper directly and noting the shapes of all the tensors that need to be stored during the computation. Like time complexity we see that memory is also quadratic in \(n\) and \(d\). 

#### Conclusion & Lingering Questions
Given that we want LLMs to be able to handle long input sequences it is reasonable to be concerned by the overall quadratic dependence on sequence length. This leads to a few of my lingering questions... 
- What are some of the key methods that have been developed to improve the effiency of self attention? While I've stumbled upon some works (e.g Flash Attention), I haven't explored the subject much.
- Are these advancements in attention efficiency the primary enabler of long context windows? I also assume that [relative positional embeddings]({{< ref "/blog/2025-01-28-pe" >}} "") play a helping role here.

As usual if you've made it this far thanks for the read. If you're looking for some calming music check out the [Peaceful Meditation playlist](https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u?si=d4272fc6cb434d24) on Spotify.
<!-- **Multi-Query Attention** proposes that instead of having \(h\) separate query, key, value projections they only use one set for both key and value. In other words multiple projections are only maintained for the query. While it's less obvious from our complexity terms how much this helps, intuitively it still reduces time and memory, e.g with 16 attention heads we would have 32 fewer projections to perform and store. It's also worth noting that this paper emphasizes that memory bandwidth is a bottleneck so the less you have to write/read to/from memory the better. -->

#### References
These are the references I found to be helpful:  
- [Multi-Query Attention Paper](https://arxiv.org/pdf/1911.02150)






