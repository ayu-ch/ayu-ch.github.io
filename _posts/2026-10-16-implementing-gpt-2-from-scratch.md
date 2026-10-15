---
title: Implementing GPT-2 from scratch
series: Inference
part: 2
---

In the [previous blog](/2026/10/12/introduction-to-inference-engineering.html), I went over how inference works: tokenization, embeddings, attention, the MLP, logits and generation. In this one, we will build it ourselves. We will take the real GPT-2 (the smallest one, with 124 million parameters), load its weights, and run it using nothing but NumPy. No other library for DL or even tokenizer. At the end, we will port it to PyTorch and run it on a GPU.

I have also made a notebook with all the code, so you can run it and experiment on your own. It works on Colab's free CPU, so you don't need anything installed:

**[Open the notebook in Colab](https://colab.research.google.com/drive/1GpahI9O11nyieCjPA2qjOtVI9zHJ1IUo?usp=sharing)**

I will use the same prompt as last time, **"Michael Jordan played the sport of"**, so you can compare everything with the previous blog.

## Getting the weights

A model is just a collection of named matrices. GPT-2's weights come as a single **safetensors** file, and its format is surprisingly simple:

1. The first 8 bytes tell you the length of a JSON header.
2. The JSON header has, for every tensor, its name, data type, shape and where its bytes are.
3. After that, it's just the raw bytes of all the tensors, one after another.

So we can read it ourselves:

```python
def read_safetensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]  # 8 bytes, little-endian unsigned integer
        header = json.loads(f.read(header_len))
        data = f.read()
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        tensors[name] = np.frombuffer(data[start:end], dtype=np.float32).reshape(info["shape"])
    return tensors
```

Here is what the first layer looks like:

```
wte.weight               (50257, 768)
wpe.weight               (1024, 768)
h.0.ln_1.weight          (768,)
h.0.ln_1.bias            (768,)
h.0.attn.c_attn.weight   (768, 2304)
h.0.attn.c_attn.bias     (2304,)
h.0.attn.c_proj.weight   (768, 768)
h.0.attn.c_proj.bias     (768,)
h.0.ln_2.weight          (768,)
h.0.ln_2.bias            (768,)
h.0.mlp.c_fc.weight      (768, 3072)
h.0.mlp.c_fc.bias        (3072,)
h.0.mlp.c_proj.weight    (3072, 768)
h.0.mlp.c_proj.bias      (768,)
...
ln_f.weight              (768,)
ln_f.bias                (768,)
```

Everything from the previous blog is here. `wte` is the token embedding matrix (one row for each of the 50,257 tokens and consisting of 768 dimensions) and `wpe` is the positional embedding matrix (one row for each of the 1,024 positions and again 768 dimensions ofcourse). `h.0` to `h.11` are the 12 layers, each with two layer norms, the attention weights and the MLP weights. And `ln_f` is the final layer norm.

Two things to notice here:

- `attn.c_attn.weight` is of shape `(768, 2304)`. The query, key and value matrices are stored together in one matrix (3 × 768 = 2304), so all three are computed with a single matrix multiplication.
- GPT-2 stores its weights as `(in, out)`, so we can just do `x @ W`. Many other models store them the other way around.

The file also has some `attn.bias` entries, of shape `(1, 1, 1024, 1024)`. Despite the name, they are not learnt weights. Older versions of GPT-2's code saved the causal mask there, so we can ignore them.

## The tokenizer

I said last time that GPT-2 uses byte pair encoding. Implementing it has three parts.

### Bytes to characters

The text is first converted into UTF-8 bytes, so that any language or emoji can be handled. But many of the 256 possible bytes are invisible (spaces, newlines, control characters), which makes them awkward to store in a vocabulary file. So GPT-2 maps every byte to a visible character. The printable ones stay the same, and the rest are shifted to other characters. This is why you will see `Ġ` all over GPT-2's vocabulary. That's a representation of a space.

### Splitting the text into chunks

Before merging anything, a regular expression splits the text into chunks: words, numbers, punctuation and spaces. This makes sure merges never cross from one word into another.

```python
PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

The ` ?` before each part is what attaches a space to the word after it, which is why the tokens look like `' Jordan'`:

```
"Michael Jordan played the sport of"  →  ['Michael', ' Jordan', ' played', ' the', ' sport', ' of']
"It's 2026, and I'll write    code!"  →  ['It', "'s", ' 2026', ',', ' and', ' I', "'ll", ' write', '   ', ' code', '!']
```

GPT-2's vocabulary is already made, so we don't build it. We just load two files: `vocab.json` (every token and its ID) and `merges.txt` (the learnt merges).

<details markdown="1">
<summary>ow the vocabulary was built</summary>

It's worth seeing how these two files were made. OpenAI did this before training the model, on its training data.

It starts with just the 256 possible bytes. Then it counts every pair of neighbouring pieces in all of the text, and merges the most common pair into a new token. Then it counts again and merges again, 50,000 times. So the vocabulary is 256 bytes + 50,000 merges + 1 special token (`<|endoftext|>`) = **50,257** tokens. The order in which the merges were learnt is their **rank**, and that's exactly the priority we will use while tokenizing.

Here's the same thing on a tiny piece of text, "low lower lowest newer newest wider widest":

```
merge ('w', 'e')    seen 4 times    (lower, lowest, newer, newest)
merge ('l', 'o')    seen 3 times
merge ('s', 't')    seen 3 times
merge ('lo', 'we')  seen 2 times
merge ('n', 'e')    seen 2 times
merge ('ne', 'we')  seen 2 times
```

Notice that after `('w', 'e')` is merged, new pairs appear, like `('lo', 'we')`. That's why the pairs have to be counted again after every merge.

</details>

### Applying the merges

`merges.txt` has the 50,000 merges that were learnt before training, **in the order they were learnt**. The position of a merge in that list is its rank: a lower rank means it was learnt earlier, because that pair was more common.

To tokenize a chunk, we start with its individual characters and keep merging the pair with the lowest rank, until none of the pairs has a learnt merge:

```python
def bpe(chunk):
    pieces = list(chunk)
    while len(pieces) > 1:
        pairs = list(zip(pieces, pieces[1:]))
        best = min(pairs, key=lambda p: merge_rank.get(p, float("inf")))
        if best not in merge_rank:
            break  # no learnt merge applies anymore
        # merge every occurrence of the best pair
        merged, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == best:
                merged.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        pieces = merged
    return pieces
```

It's really nice to watch it work. Here is " inference" being built up, one merge at a time:

```
merge ('i', 'n') (rank 3):         ['Ġ', 'in', 'f', 'e', 'r', 'e', 'n', 'c', 'e']
merge ('r', 'e') (rank 4):         ['Ġ', 'in', 'f', 'e', 're', 'n', 'c', 'e']
merge ('Ġ', 'in') (rank 31):       ['Ġin', 'f', 'e', 're', 'n', 'c', 'e']
merge ('c', 'e') (rank 88):        ['Ġin', 'f', 'e', 're', 'n', 'ce']
merge ('e', 're') (rank 311):      ['Ġin', 'f', 'ere', 'n', 'ce']
merge ('Ġin', 'f') (rank 911):     ['Ġinf', 'ere', 'n', 'ce']
merge ('n', 'ce') (rank 942):      ['Ġinf', 'ere', 'nce']
merge ('ere', 'nce') (rank 1689):  ['Ġinf', 'erence']
merge ('Ġinf', 'erence') (rank 32022):  ['Ġinference']
```

Common pairs like "in" and "re" merge first, and the full word " inference" is a single token in the end. (In the previous blog, "inference" without the space was two tokens, `'in'` and `'ference'`. With the space, it's a different and more common chunk.)

Putting it all together, `encode` splits the text into chunks, converts each chunk's bytes into characters, runs `bpe` on it and looks up the IDs in `vocab.json`. And our prompt gives the same 6 IDs as last time:

```
[13256, 8078, 2826, 262, 6332, 286]
```

I also compared our tokenizer with Hugging Face's on some harder inputs, like Hindi, emojis, long numbers and extra spaces, and it matched on all of them.
The notebook also has a small bonus section that learns merges from a tiny piece of text, which is the training step I described in the previous blog.

## Embeddings

This is the easiest part. Each token ID picks its row from `wte`, each position picks its row from `wpe`, and the two are added:

```python
x = wte[ids] + wpe[np.arange(len(ids))]   # (6, 768)
```

## Layer normalization

For each token's vector, we subtract the mean, divide by the standard deviation, and then scale and shift with the two learnt vectors (gamma and beta, stored as `weight` and `bias`):

```python
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias
```

The small `eps` is there so we never divide by zero.

## Attention

Let's go through the attention of the first layer, step by step, with the shapes at every step.

**Query, key and value.** One matrix multiplication with `c_attn` gives all three, and we split the result into three parts of 768:

```python
qkv = h @ W_attn + b_attn            # (6, 2304)
q, k, v = np.split(qkv, 3, axis=-1)  # (6, 768) each
```

Each of the 12 heads gets its own 64 of the 768 numbers. We reshape `(6, 768)` into `(6, 12, 64)`, and then move the head dimension to the front, giving `(12, 6, 64)`. Now every head is its own `(6, 64)` matrix, and NumPy can compute all 12 heads at once.

```python
def split_heads(t):
    T = t.shape[0]
    return t.reshape(T, 12, 64).transpose(1, 0, 2)  # (12, T, 64)
```

Every query is compared with every key using a dot product, for all heads at once. The scores are divided by √64 = 8.  It's done because the dot product becomes very large.

```python
scores = q @ k.transpose(0, 2, 1) / 64 ** 0.5   # (12, 6, 6): heads × query token × key token
```

<details markdown="1">
<summary>Why √64?</summary>

A dot product of two 64-dimensional vectors is a sum of 64 products. If the numbers in $$q$$ and $$k$$ have a variance of about 1, then each product also has a variance of about 1, and the sum of 64 of them has a variance of 64:

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{64} \text{Var}(q_i k_i) \approx 64
$$

So the scores are typically around ±8, and with bigger heads they would be even bigger. Softmax of large numbers is almost one-hot: nearly all the weight goes to the single highest score, and the model can't spread its attention over several tokens. During training, the gradients through such a softmax are also close to zero, so it learns slowly.

Dividing by $$\sqrt{64} = 8$$ brings the variance back to about 1, whatever the head size is.

</details>

We set a causal mask by setting every score above the diagonal to minus infinity, so that softmax gives it a weight of exactly 0. These are the "future" tokens:

```
[[0 1 1 1 1 1]
 [0 0 1 1 1 1]
 [0 0 0 1 1 1]
 [0 0 0 0 1 1]
 [0 0 0 0 0 1]
 [0 0 0 0 0 0]]
```

Now we use a softmax function so that each row of scores is turned into weights that add up to 1. Here are the weights of the first head:

```
               Michael  Jordan  played   the   sport    of
'Michael'       1.00
' Jordan'       0.92     0.08
' played'       0.56     0.31     0.12
' the'          0.68     0.13     0.15   0.04
' sport'        0.51     0.12     0.08   0.16   0.14
' of'           0.46     0.16     0.12   0.07   0.15   0.04
```

Each row is a token looking back at the ones before it. You can see the causal mask at work: everything above the diagonal is 0. Also notice how much weight goes to the very first token, `'Michael'`. This is a well-known pattern: many heads put a lot of weight on the first token, which is called an **attention sink**. I will come back to it in a future blog.

Then, just like in the previous blog, each token's output is a weighted sum of the values. Every value is multiplied by its attention weight, and the results are added up. A single matrix multiplication, softmax(scores) @ v, does this for all tokens and all 12 heads at once. After that, we join the heads back into one vector of 768 numbers per token, and apply the output projection, `c_proj`:

```python
out = softmax(scores) @ v                          # (12, 6, 64)
out = out.transpose(1, 0, 2).reshape(T, 768)       # (6, 768): join the heads back
out = out @ W_proj + b_proj                        # (6, 768)
```

Putting all of these steps together, this is the formula you will see in the paper (for each head, before the heads are joined and `c_proj` is applied):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

Here $$d_k = 64$$ is the head size, and $$M$$ is the causal mask: 0 where token $$i$$ is allowed to look at token $$j$$, and minus infinity for the future tokens.

## The MLP

Two linear layers with GELU between them, 768 → 3072 → 768. GPT-2 uses an approximation of GELU with `tanh`, so we use the same one:

```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh((2 / np.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

def mlp(x, layer):
    h = gelu(x @ W_fc + b_fc)    # (T, 3072)
    return h @ W_proj + b_proj   # (T, 768)
```

## The whole model

We almost have it working now!! Each layer is attention and the MLP, both after a layer norm and both added to `x` (the residual connection). Then the final layer norm and the logits, using `wte` again:

```python
def gpt2(ids):
    x = wte[ids] + wpe[np.arange(len(ids))]
    for layer in range(12):
        x = x + attention(ln(x, f"h.{layer}.ln_1"), layer)
        x = x + mlp(ln(x, f"h.{layer}.ln_2"), layer)
    return ln(x, "ln_f") @ wte.T   # (T, 50257)
```

## Logits

The model gives us logits for every position, `(6, 50257)`. For the next token, we only need the last row. Applying softmax to it:

```
' basketball'   14.3%
' football'      8.5%
' golf'          4.7%
' hockey'        3.4%
' rugby'         3.4%
```

## Generation

Now we pick a token, append it, and run the model again. With greedy sampling:

```
Michael Jordan played the sport of basketball for the University of Michigan, where he was named the
```

And with some randomness (temperature 0.8, top-p 0.9):

```
Michael Jordan played the sport of rugby for the first time when he played for Great Britain at
Michael Jordan played the sport of golf as a 12-year-old, and when he
Michael Jordan played the sport of basketball for 13 years.
```

It can be really funny at times lmao.

## Is it actually correct?

It's easy to write something that produces reasonable-looking text but is subtly wrong. So I compared our logits with Hugging Face's GPT-2, loaded from the same files. The largest difference across all 6 positions and 50,257 tokens was about **0.00014**, and the top token was the same at every position. Differences this small come from the order in which floating point numbers are added up, which is slightly different between implementations, so this means our implementation is correct.

## How slow is it?

Our `generate` runs the whole sequence through the model for every new token. Every step gets slower, because the input keeps growing and we recompute everything for all the earlier tokens. But the keys and values of the earlier tokens never change! Storing them instead of recomputing them is what's called **KV cache**.

Here's what happens when we generate 60 tokens, without and with a KV cache. The first step is the **prefill**, where the model reads the 6-token prompt. Every step after that is a **decode** step, which produces one more token:

```
                        tokens run through the model       time per step
                        without cache    with cache    without cache   with cache
prefill                       6               6            71 ms          50 ms
decode steps  1-10          7-16              1            84 ms          30 ms
decode steps 11-20         17-26              1           100 ms          33 ms
decode steps 21-30         27-36              1           144 ms          37 ms
decode steps 31-40         37-46              1           152 ms          38 ms
decode steps 41-50         47-56              1           179 ms          34 ms
decode steps 51-59         57-65              1           175 ms          42 ms

all 60 tokens                                             8.2 s          2.1 s
```

These are from my laptop's CPU, averaged over each group of 10 steps and over 5 runs, because single steps on a CPU jump around a lot. The prefill does the same work in both cases, so the difference there is just noise.

Without the cache, every decode step runs the whole text through the model again. With the cache, only the one new token goes through, so every step takes about the same time, no matter how long the text gets, and generating 60 tokens is almost **4× faster**. And the longer the text, the bigger the difference. We will talk about how KV cache and different systems for caching in the next blog!

## The same model in PyTorch

Porting it to PyTorch is **almost** same line by line. Some small changes and you can run on GPU. 

PyTorch also has the whole attention computation as one function, `F.scaled_dot_product_attention`. On a GPU, it uses fused kernels like FlashAttention, and it gives the same result as our step-by-step attention.

And the speed, still without a KV cache:

```
NumPy (CPU):          89.7 ms per token
PyTorch (RTX 3060):    6.5 ms per token
```

These numbers are from my laptop, so yours will be different. On Colab, switch the runtime to a T4 GPU to try the PyTorch part.

## Wrapping up

You just understood how GPT-2 generates tokens now! I was so excited to show this implementation. 

Please try out the [notebook](https://colab.research.google.com/drive/1GpahI9O11nyieCjPA2qjOtVI9zHJ1IUo?usp=sharing) with your own prompts, change the temperature, or print the attention weights of other heads and layers to see what they look at. I am very excited to talk about KV caching and different systems around it in the next blog:)

If you find anything wrong, please let me know!

*[safetensors]: A file format for model weights, made by Hugging Face: a JSON header followed by the raw bytes of every tensor. Unlike older pickle files, loading it can't run any code.
*[UTF-8]: The standard way to store text as bytes. English letters take 1 byte each, while other scripts and emojis take 2 to 4.
*[GELU]: Gaussian Error Linear Unit, a smooth version of ReLU. Large positive inputs pass through, large negative ones become 0, and in between it curves smoothly.
*[residual connection]: Adding a layer's output to its input instead of replacing it, so each layer only learns a change to x. It also helps training, because gradients can flow straight through.
*[logits]: The raw scores the model gives every token in the vocabulary. Softmax turns them into probabilities.
*[greedy]: Always picking the most likely token. The output is the same every time.
*[top-p]: Sampling only from the smallest set of most likely tokens whose probabilities add up to p, so very unlikely tokens are never picked.
*[temperature]: The logits are divided by it before softmax. Below 1 makes the model more confident, above 1 makes it more random.
*[prefill]: Running the whole prompt through the model at once, which also produces the first new token.
*[fused kernels]: A kernel is a function that runs on the GPU. A fused kernel does several steps in one go, so the intermediate results never have to be written to GPU memory.
*[FlashAttention]: A fused attention kernel that works on small tiles in fast on-chip memory, so the full matrix of attention scores is never stored.
