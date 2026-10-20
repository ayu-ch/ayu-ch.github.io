---
title: The KV cache and the systems around it
description: "Adding a KV cache to GPT-2 written from scratch in NumPy, checking that it's exact, measuring it, and how PagedAttention and prefix caching in vLLM manage and reuse it."
series: Inference
part: 3
image: /assets/images/kv-cache-paging.png
---

In the [previous blog](/2026/10/16/implementing-gpt-2-from-scratch.html), we built GPT-2 from scratch. At the very end, I showed that storing the keys and values of the earlier tokens made generating 60 tokens almost 4× faster. In this one, I will build that cache properly, see how fast it is, and then look at what happens when you have to keep it for thousands of users at once. That last part is where most of the interesting work in inference serving is.

## What's wrong in our loop?

Here's the generation loop from the last blog, in short:

```python
for _ in range(n_new):
    logits = gpt2(ids)            # runs the WHOLE sequence through the model
    ids.append(int(np.argmax(logits[-1])))
```

Every step runs every token through all 12 layers again, just to get logits for the last position. When the text is 500 tokens long, we recompute 499 tokens that we had already computed in the previous step, and the step before that, and so on.

How bad is that? Here is the time for one decode step, depending on how many tokens are already there, without and with a cache:

```
tokens already there     one step without cache     one step with cache
        16                       37 ms                    20 ms
       128                      167 ms                    23 ms
       256                      435 ms                    24 ms
       512                      795 ms                    22 ms
     1,000                    2,400 ms                    24 ms
```

I ran this on my laptop's CPU.

Without the cache, each step gets slower as the text grows, and at 1,000 tokens one step takes 2.4 seconds. With the cache, every step takes about the same time, no matter how long the text is. That's **100× faster** at 1,000 tokens. The 4× from the last blog was only for 60 tokens. The longer the text, the bigger the difference.

## Why the keys and values never change

This whole caching idea depends on the claim that once a token has been processed, its keys and values in every layer stay the same forever, however many tokens come after it. Let's see layer by layer:

1. Token *i*'s vector is `wte[token] + wpe[i]`. That depends only on the token itself and its position.
2. Because of the causal mask, token *i* only looks at tokens 0 to *i*. Tokens that come later are invisible to it.
3. Layer norm and the MLP work on each token on its own, so they only depend on token *i*'s own vector.
4. So after layer 0, token *i*'s vector depends only on tokens 0 to *i*. The same reasoning works for layer 1, then layer 2, and all the way up to layer 11.

So everything computed for token *i*, including its key and value in every layer, is fixed the moment it's computed. Recomputing it at the next step gives exactly the same numbers, which is pure waste.

<details markdown="1">
<summary>Then why not cache the queries too?</summary>

A query is only used once: to compute its own token's output, by comparing it with the keys before it. Once token *i*'s output is computed, its query is never needed again.

When a new token comes in, the only thing to compute is **the new token's** output. For that, we need the new token's query, and the keys and values of every token so far:

```
new token's query  ──compared with──►  K of tokens 0, 1, 2, ..., n
                                       V of tokens 0, 1, 2, ..., n  ──► new token's output
```

The old queries play no part in it. Hence it's called a **KV** cache.

This also shows why the cache only works for models with a causal mask, like GPT-2 and every chat model. In a model like BERT, every token looks at every other token, including the later ones. So adding a token changes everything before it, and nothing can be reused.

</details>

## Adding the cache to GPT-2

We need somewhere to keep the keys and values. For every layer and every head, that's one row of 64 numbers per token. GPT-2 can handle at most 1,024 tokens, so we can just reserve space for all of them up front:

```python
# (layers, heads, max tokens, head size)
k_cache = np.zeros((12, 12, 1024, 64), dtype=np.float32)
v_cache = np.zeros((12, 12, 1024, 64), dtype=np.float32)
```

Now attention gets one more argument, `start`: how many tokens are already in the cache. The new tokens are written right after them:

```python
def attention(h, layer, start):
    T = h.shape[0]
    qkv = h @ W_attn + b_attn
    q, k, v = map(split_heads, np.split(qkv, 3, axis=-1))

    # write the new keys and values after the cached ones
    k_cache[layer, :, start:start + T] = k
    v_cache[layer, :, start:start + T] = v
    K = k_cache[layer, :, :start + T]
    V = v_cache[layer, :, :start + T]

    scores = q @ K.transpose(0, 2, 1) / 64 ** 0.5      
    mask = np.arange(start + T)[None, :] <= (start + np.arange(T))[:, None]
    scores = np.where(mask, scores, -np.inf)

    out = softmax(scores) @ V 
    out = out.transpose(1, 0, 2).reshape(T, 768)
    return out @ W_proj + b_proj
```

Compared with the last blog, only one thing has changed:

- The keys and values come from the cache and also from the current input. 

Let's look at the shapes when we generate the 7th token. The 6 prompt tokens are already in the cache, so `start = 6` and `T = 1`:

```
q        (12, 1, 64)      one query per head, only for the new token
K, V     (12, 7, 64)      6 cached + 1 new
scores   (12, 1, 7)       the new token against all 7 keys
```

The newest token is allowed to look at everything, so during decode the mask actually does nothing. It matters during the prefill, where it's the same triangle as last time. It also works for the case in between, where some tokens are cached, and a few new ones come in together. For example, with `start = 4` and `T = 2`.

This case comes back later in the blog, when parts of a prompt are already cached.

The model itself barely changes. We only need the logits of the last position:

```python
def gpt2(ids, start):
    x = wte[ids] + wpe[start:start + len(ids)]
    for layer in range(12):
        x = x + attention(ln(x, f"h.{layer}.ln_1"), layer, start)
        x = x + mlp(ln(x, f"h.{layer}.ln_2"), layer)
    return ln(x[-1], "ln_f") @ wte.T                   # (50257,): last position only
```

Now generation is just prefill, followed by decode steps that each send in just one token

```python
def generate(prompt_ids, n_new):
    ids = list(prompt_ids)
    logits = gpt2(ids, start=0)                        # prefill: the whole prompt at once
    for _ in range(n_new):
        nxt = int(np.argmax(logits))
        ids.append(nxt)
        logits = gpt2([nxt], start=len(ids) - 1)       # decode: only the new token
    return ids
```

<details markdown="1">
<summary>Is it actually correct?</summary>

A cache is an optimization, so it must not change the answers. With greedy sampling, the model with the cache has to pick exactly the same tokens as the model without it.

</details>

## How big does it get?

For every token, the cache keeps one key and one value, in every layer, for every head:

$$
\text{bytes per token} = 2 \times \text{layers} \times \text{KV heads} \times \text{head size} \times \text{bytes per number}
$$

For GPT-2 small in float32, that's 2 × 12 × 12 × 64 × 4 bytes = **72 KiB per token**, and 72 MiB for the full 1,024 tokens.

That doesn't sound like much. But here are some more models, all in 16-bit numbers (2 bytes):

```
model              layers   KV heads   head size   per token    one long request
Llama 2 7B           32        32         128       512 KiB      4K tokens:    2 GiB
Llama 3.1 8B         32         8         128       128 KiB      128K tokens: 16 GiB
Llama 3.1 70B        80         8         128       320 KiB      128K tokens: 40 GiB
```

A single 128K-token request on Llama 3.1 8B needs 16 GiB of cache, which is about as much memory as the model's weights. And a server doesn't have one request. It has dozens or hundreds of them at the same time, each with its own cache. 32 users with 8K tokens each on the same model is another 32 GiB.

So for long texts and many users, the problem isn't really the model anymore. It's the cache.

(You might wonder why Llama 3.1 8B has only 8 KV heads when Llama 2 7B has 32. That's one of the tricks to shrink the cache, and we will get to it near the end.)

## The cache decides how many users you can serve

In the first blog, I said decode is memory bound: to generate one token, the GPU reads all of the model's weights and does very little math with them. Batching fixes that, because the weights are read once and used for every user in the batch. But the caches can't be shared. Every user has their own, and every decode step has to read all of them.

To see this, I ran the decode step on my laptop's GPU (PyTorch, float16), with every sequence already holding 512 tokens, and increased the batch size:

```
sequences    time per step    tokens per second    KV cache size
    1           3.2 ms               315               18 MiB
    8           3.4 ms             2,400              144 MiB
   32           6.7 ms             4,800              577 MiB
  128          23.7 ms             5,400            2,308 MiB
  256          45.4 ms             5,600            4,617 MiB
```

- **From 1 to 8 sequences, the time barely changes**, so we get almost 8× more tokens for free. That's batching doing its job.
- **After 32, doubling the batch doubles the time**, and throughput stops growing at around 5,400 tokens per second. The weights are shared, but each sequence brings 18 MiB of its own cache, and reading those becomes most of the work.
- **At 256 sequences, the cache is 4.6 GiB** on a 6 GB GPU. The next doubling doesn't fit.

So the memory you have for the cache decides how many sequences you can batch together, and that decides your throughput. Every byte wasted is fewer users per GPU.

## The systems built around it

So the cache is big, it grows with every token, every user has their own, and how well you manage it decides how many users you can serve. Most of what a serving engine like [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang) does is about this. The two most important ideas are **PagedAttention** and **prefix caching**, so let's go through those in detail, and then look at the rest more briefly.

### The problem with reserving memory

When a request comes in, you know how long its prompt is, but not how long the answer will be. The model decides when to stop. Our GPT-2 dealt with that by reserving space for the maximum of 1,024 tokens up front, in one long chunk. Older serving systems did the same, and it wastes memory in three ways:

```
one request's reserved chunk (max 4,096 tokens):

[ prompt | answer so far |  will be used later  |     never used     ]
  real tokens              reserved, empty now    reserved, wasted forever
```

- **Reserved for later:** space for tokens that will come, but sits empty until they do.
- **Never used:** space for tokens that never come at all, because the answer stopped early. You reserved 4,096 tokens and the answer was done after 600.
- **Gaps between requests:** requests finish at different times and leave holes behind. A hole can be too small for the next request, even if there's enough free memory in total, just not in one piece.

How much does that add up to? I simulated 2,000 requests with prompt and answer lengths (some short and some long), and counted how much of the reserved memory actually held real tokens at the end:

```
reserve the max length (4,096) for every request:     12.8%
reserve the prompt + max answer length (1,024):       37.4%
small blocks of 16 tokens, only when needed:          98.6%
```

That's before even counting the gaps between requests, so the first two are actually a bit worse. And since the memory decides the batch size, this turns directly into users. With 4 GiB of cache for Llama 3.1 8B (128 KiB per token) and a typical request of 440 tokens, you can serve **8** requests at once if you reserve the maximum, **24** if you reserve the prompt plus the maximum answer length, and **73** if you only use what's needed!

### PagedAttention

**PagedAttention**, from the [vLLM paper](https://arxiv.org/abs/2309.06180), solves this the same way operating systems solved it for normal memory decades ago.

A program thinks its memory is one long, continuous range of addresses. In reality, the operating system splits memory into small **pages**, gives a program a new page only when it needs one, and puts those pages wherever there's space. A **page table** remembers which real page holds each part of the program's memory. The program never notices.

PagedAttention does exactly this with the KV cache:

- The GPU memory for the cache is one big **pool of blocks**. Each block holds the keys and values of a fixed number of tokens, usually 16, in every layer.
- A request starts with only as many blocks as its prompt needs. It gets **a new block only when its last one is full.**
- The blocks can be **anywhere** in the pool. They don't need to be next to each other.
- Each request has a **block table**: the list of its blocks, in order. That's its page table.

```
block pool in GPU memory:
[ A0 ][ B0 ][ C0 ][ C1 ][ A1 ][ B1 ][ free ][ A2 ][ free ][ free ] ...
   0     1     2     3     4     5      6      7      8       9

block tables:
request A:  [0, 4, 7]    tokens 0-15 in block 0, 16-31 in block 4, 32-47 in block 7
request B:  [1, 5]
request C:  [2, 3]
```

So what does that fix?

- **Nothing is reserved for later.** A request only ever holds the blocks it has filled, plus one partly filled block at the end.
- **Nothing is wasted when an answer stops early.** It never had more blocks than it used.
- **No gaps.** When a request finishes, its blocks go back to the free list, and any request can use any free block. There's no such thing as a hole that's too small.

The only waste left is the unused part of each request's **last** block, at most 15 tokens. That's where the 98.6% comes from.

**But attention needs the keys in order**, and now they're scattered. That's the "attention" part of PagedAttention. It's a special attention kernel that takes the block table as an input. While computing the scores for a request, it looks up each block in the table and reads that block's keys and values straight from the pool. The scattering costs very little, because a block of 16 tokens is still one continuous piece of memory that can be read efficiently. A simple implementation that first *copies* the blocks into one array would be much slower, since it would copy the whole cache every step.

**Why 16 tokens per block?** It's a trade-off. Bigger blocks waste more space in the last block of every request. Smaller blocks mean longer block tables, more lookups, and smaller memory reads, which the GPU handles less efficiently. vLLM uses 16 by default.

The vLLM paper reports **2–4× more throughput** than the systems before it, mostly because so many more requests fit in the same memory.

And there's one more thing blocks make possible: **sharing**. A block is just a number in a table, so two block tables can point to the same block. Each block keeps a **reference count**: how many requests are using it. It only goes back to the free list when the count drops to zero.

Where does sharing help? Say you ask for 4 different answers to the same prompt (`n=4` in the API). The prompt's keys and values are the same for all 4, so all 4 block tables point to the **same** prompt blocks, and the prompt is stored once instead of 4 times. Only when an answer writes into a shared block is that block copied first, so the others don't see the change. That's called **copy on write**, which is again borrowed from operating systems.

Sharing within one request is nice. Sharing **between** requests is even better, and that's prefix caching.

### Prefix caching

Many requests start with exactly the same text: the same system prompt, the same instructions, the same document with different questions, or the same chat history with one new message at the end. And a token's keys and values depend only on the tokens **before** it, so if the beginning is the same, its cache is the same too. **Prefix caching** keeps the blocks around after a request finishes, so later requests can reuse them instead of computing them again.

To find them, vLLM gives every **full** block a **hash**, computed from its 16 tokens **and the hash of the block before it**:

```
block 0:  hash0 = hash(tokens 0-15)
block 1:  hash1 = hash(hash0 + tokens 16-31)
block 2:  hash2 = hash(hash1 + tokens 32-47)
```

So a block's hash depends on every token from the start up to the end of that block, just like its keys and values do. When a new request comes in, the engine hashes its blocks and walks through them from the start. Every block that's already cached goes straight into the new block table, and only the rest of the prompt goes through the model. That's the `start > 0`, `T > 1` case from our mask earlier.

```
request 1:  [system prompt ......][data .................][question 1]
             computed, then cached ─────────────────────►
request 2:  [system prompt ......][data .................][question 2]
             reused from the cache, nothing computed      only this part is computed
```

Only full blocks are cached, since a partly filled block can still change. The match always starts from the beginning: everything after the first different token has a different hash. And when memory runs out, the least recently used cached blocks are thrown away, so a hit is likely but never guaranteed. (SGLang does the same thing with a **radix tree** instead of hashes, which isn't limited to multiples of 16 tokens.)

I tried this with vLLM, serving Qwen2.5-1.5B on an RTX 5070 Ti. I sent the same ~5,900-token prompt three times, changing only the question at the end, and measured the time to the first token:

```
                          request 1    request 2    request 3
prefix caching on           311 ms        30 ms        27 ms
prefix caching off          351 ms       344 ms       341 ms
```

Requests 2 and 3 each reused 5,920 tokens, exactly 370 full blocks, and only the last 17 tokens were computed. The remaining 30 ms isn't the model anymore, it's the fixed overhead around it: the HTTP request, tokenizing 5,900 tokens on the CPU, and looking up 370 hashes.

This is why API providers charge less for "cached input tokens". It's also why you should put the parts of your prompt that don't change at the start, and the parts that do at the end. Something like the current time at the top of a system prompt changes the hash of every block after it, and then nothing can be reused.

### Making it smaller

The other way to fit more is to make the cache itself smaller:

- **Fewer KV heads (MQA and GQA).** In GPT-2, each of the 12 heads has its own keys and values. With **grouped-query attention**, several query heads share one key and value head. Llama 3.1 8B has 32 query heads but only 8 KV heads, which is exactly why its cache is 4× smaller than Llama 2 7B's in the table above, with almost no loss in quality. **Multi-query attention** takes it all the way down to a single KV head.
- **Compressing it (MLA).** DeepSeek's models store a small compressed vector per token instead of full keys and values, and expand it when needed. DeepSeek-V3 has 61 layers and 671 billion parameters, but its cache is about 69 KiB per token. That's less than our tiny GPT-2 in float32!
- **Fewer bytes per number.** Storing the cache in 8-bit floats (FP8) instead of 16-bit halves it, usually with very little effect on the output.
- **Not keeping all of it.** Some models only attend to the last few thousand tokens in some of their layers (**sliding window attention**), so those layers only need to keep that many. Remember the attention sink from the last blog, where many heads put a lot of weight on the first token? It turns out that if you throw away old tokens, you have to keep those first few, or the model falls apart. That's the idea behind [StreamingLLM](https://arxiv.org/abs/2309.17453).

### When it still doesn't fit

Even with all of that, a busy server runs out of cache memory. Then it has to choose:

- **Preemption.** Pause some requests and free their blocks. When there's space again, either **recompute** their cache from the text (a prefill, which is fast), or **swap** it out to the CPU's memory and copy it back later.
- **Offloading.** Keep the cache that isn't needed right now, like old conversations, in CPU memory or even on disk, so it can be brought back instead of recomputed.
- **Splitting prefill and decode.** Big deployments run prefill on some GPUs and decode on others, because one is compute bound and the other is memory bound. The cache is computed on the first group and sent over the network to the second, so moving the cache becomes a problem of its own.

## Wrapping up

So the KV cache comes from a simple fact: because of the causal mask, a token's keys and values never change. Storing them makes every decode step process one token instead of the whole text, which was 100× faster at 1,000 tokens. But the cache grows with every token and every user, and its size decides how many users a GPU can serve. That's why engines page it, share it and shrink it.

There's still a lot more to inference than the cache, and I will keep writing about it as I learn. I hope this one helped someone:)

If you find anything wrong, please let me know!

*[KV cache]: The stored keys and values of all the tokens processed so far, in every layer, so they don't have to be computed again.
*[causal mask]: The mask that stops each token from looking at the tokens after it.
*[prefill]: Running the whole prompt through the model at once, which also produces the first new token.
*[decode]: Generating the answer one token at a time, after the prefill.
*[greedy]: Always picking the most likely token. The output is the same every time.
*[float32]: A number stored in 32 bits (4 bytes). float16 and bfloat16 use 16 bits (2 bytes), and FP8 uses 8 bits.
*[KiB]: Kibibyte, 1,024 bytes. MiB is 1,024 KiB and GiB is 1,024 MiB.
*[batching]: Serving several requests together in the same step, so one read of the weights is used for all of them.
*[throughput]: How many tokens the whole server generates per second, across all users.
*[memory bound]: Limited by how fast data can be read from memory, not by how fast the GPU can do math.
*[compute bound]: Limited by how fast the GPU can do math, not by how fast data can be read.
*[GQA]: Grouped-query attention: several query heads share one key head and one value head.
*[MLA]: Multi-head latent attention, from DeepSeek: the keys and values are stored as a smaller compressed vector.
*[hash]: A short fingerprint computed from some data. The same data always gives the same hash.
*[block table]: The list of blocks that hold one sequence's keys and values, in order.
*[radix tree]: A tree where each path from the root spells out a sequence, so sequences with the same beginning share the same branch.
