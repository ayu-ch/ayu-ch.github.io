---
series: Inference
part: 1
---

Hello and welcome to my blog. It's been a long time since I wrote a blog. I have been studying inference for some time. While I was doing so, it was difficult for me to find good resources that went deep into the topics I wanted to study. Most of the blogs or posts I found were just surface level knowledge, which is fine for someone wanting to get into the field. Well, it is understandable that there cannot be a single place where you would find everything. In this series, I will try to condense the information I found and understood. I hope that people wanting to get into this field can refer to my blogs and make use of them as much as possible. It will also help me solidify my concepts as I write them down myself. I would like to be corrected if I am wrong somewhere because it would be a bigger loss for the reader to be misinformed.

I will try to be regular, and this series will dive deep into different topics and aspects of inference, GPU architecture, its usage and optimization, and also scaling (I have not decided the order and the contents to be posted but I will try my best to cover as much as I can). Ironically, this particular blog will be an overview and introduction for anyone trying to get into the field of inference engineering and optimization. 


## What is inference?

Before talking about inference, let's see where it lies in a model's timeline. A model like GPT-2 goes through roughly these stages:

1. **Training**: The model is shown huge amounts of text and learns to predict the next token. This is where its <abbr title="The learnt numbers inside the model's matrices. GPT-2 small has about 124 million of them.">weights</abbr> (the numbers inside the model) are learnt. It takes a lot of GPUs, a lot of time and a lot of money, but it happens once.
2. **Fine-tuning**: The trained model is trained a bit more on specific data, for example conversations, so that it behaves like an assistant.
3. **Inference**: The weights are fixed as per the use case and can now be used. When someone sends a prompt, it generates an output now and the process behind this is inference. While training happens only once, inference happens everytime anyone uses the model. This is why making inference faster and cheaper is a whole field on its own.

To explain how inference works, I will use GPT-2 as the reference. It's an old and small model, but the newer models are built on the same ideas, and it is small enough that you can run it (and even write it yourself, trust me) on a laptop. I will be using the smallest version, which has 12 layers.

Let's follow what happens when we give it the prompt **"Michael Jordan played the sport of"**. For the model to complete it correctly, let's see what it does.

## Tokenization

The model cannot read text directly. It only works with numbers, so the first step is to split the input into tokens and convert each token into its mapped token ID.


Now, a token is not exactly a word. It CAN be a full word, a part of a word or a single character. GPT-2 has a <abbr title="The fixed list of all the tokens the model knows. Each one has an ID.">vocabulary</abbr> of 50,257 tokens, and our prompt becomes 6 tokens:

```
'Michael' | ' Jordan' | ' played' | ' the' | ' sport' | ' of'
[13256, 8078, 2826, 262, 6332, 286]
```

Notice that the space is a part of the token: it's `' Jordan'` and not `'Jordan'`. Also, `'Jordan'` without the space is a completely different token with a different ID than `' Jordan'`. Common words get a token of their own, while less common words get split into pieces. For example, "inference" becomes `'in'` and `'ference'`.

So how are these tokens decided? GPT-2 uses **[Byte Pair Encoding (BPE)](https://huggingface.co/learn/llm-course/en/chapter6/5)**. The idea is simple:

1. Start with the text as individual bytes (so any text in any language can be represented).
2. Find the pair of neighbouring pieces that occurs most often in the training data, and merge it into a new piece.
3. Repeat this until the vocabulary reaches the size you want.

This is done once, before training, and the list of merges is saved. While tokenizing a new input, the same merges are applied in the same order. So frequent words like " the" end up as a single token, and rare words are built from smaller pieces.

Take a look at [Tiktokenizer](https://tiktokenizer.vercel.app/?model=gpt2) and play around to see how your text breaks down into tokens.

## Embeddings
Now we have calculated token IDs from our input text, but those don't tell the model much about the token. So each token ID is converted into a vector, called its embedding.

The model has an **embedding matrix** with one row for every token in the vocabulary. In GPT-2, each row has 768 numbers. Converting a token ID into its embedding is just picking that row from the matrix, it's just a lookup and no computation is needed. 

These vectors are learnt during training, and tokens with similar meanings end up closer to similar vectors. Our 6 tokens now become 6 vectors, i.e. a matrix of shape (6, 768).

For reference, a vector is just a list of numbers, where each number is a coordinate along one dimension. A 2-D vector like [3, 2] is a point on a flat sheet, and a 3-D vector like [3, 2, 5] is a point in space. A 768-D vector follows the same idea, but the human mind cannot imagine it.

### Positional embeddings

There is one problem. The embedding of `' Jordan'` is the same no matter where it appears in the sentence. But the order of words matters a lot: "the dog bit the man" and "the man bit the dog" have the same tokens but very different meanings.

So the model also needs to know the **position** of each token. GPT-2 does this using **positional embeddings**: another learnt matrix, with one vector for each position (0, 1, 2, ... up to 1023, which is the longest input GPT-2 can take). The positional embedding is simply added to the token embedding:

```
input to the model = token embedding + positional embedding
```

Now `' Jordan'` at position 1 and `' Jordan'` at position 20 have different vectors.

**Techniques to look out for:** (might talk in some other blog)
- [RoPE (Rotary Position Embedding)](https://krasserm.github.io/2022/12/13/rotary-position-embedding/): This is used by most newer models like Llama and Qwen. Instead of adding a position vector at the start, it rotates the query and key vectors inside attention based on their positions.
- **ALiBi**: Adds a penalty to attention scores based on how far apart two tokens are.



## Attention

This is the most famous part of the transformer. The vectors now go through 12 layers (also called blocks), and each layer has two parts: **attention** and an **MLP**.

Till now, every token's vector only knows about the token itself and its position. Attention intuitively is where tokens look at each other and update their vectors using the context they gather from the input. You can picture each token's vector moving to a new point in the 768-dimensional space, one that better reflects what it means in this sentence.

For example, the embedding of `' Jordan'` alone could mean a country, a river or a person. It's only by looking at `'Michael'` before it that the model can tell which Jordan this is. And for the model to predict what comes after "of", the vector at that position needs to know that the sentence is about Michael Jordan and a sport.

For every token, three vectors are made by multiplying its vector with three learnt matrices/weights:

- **Query (Q)**
- **Key (K)**
- **Value (V)**

Then, for each token:

1. Its query is compared with the keys of all the tokens (using a dot product). A higher score means that token is more relevant.
2. The scores are turned into weights that add up to 1, using <abbr title="Takes a list of scores, raises e to the power of each one, and divides by their sum. The result is positive and adds up to 1.">softmax</abbr> .
3. The values of all the tokens are added up using these weights, and the result is the information this token gathered.

One important rule: a token can only look at itself and the tokens before it. This is called the **causal mask**. Since, there is no meaning for having the knowledge of future words, and this also makes sense when it comes to generating text.

```
x = x + attention(LayerNorm(x))
```
Layer normalization: Before attention and before the MLP, each token's vector is normalized. Its 768 numbers are shifted and scaled so that their mean is 0 and their variance is 1, and then multiplied and shifted again by two learnt vectors (called gamma and beta), so the model can still choose the scale it wants. This is done for every token separately.

GPT-2 actually does this 12 times in parallel in each layer, with smaller vectors of size 64. These are called **heads**, and each head can learn to look for different things. The outputs of all the heads are joined back together and passed through one more matrix.

## MLP

After attention, each token's vector goes through an **MLP** (multi-layer perceptron), also called the feed-forward network and unlike attention, the MLP works on **each token separately**.

In GPT-2, the MLP takes the 768-sized vector, expands it to 3072, applies an activation function (<abbr title="Gaussian Error Linear Unit: a smooth curve that keeps positive values and pushes negative values towards zero.">GELU</abbr>), and brings it back down to 768.

Intuitively, attention is where tokens gather information from each other, and the MLP is where each oken processes that information. A lot of what the model knows (facts, patterns) is believed to be stored in the MLP weights. In fact, the MLP holds about 2/3rd of the parameters in each layer.

Two more things happen in every layer:

- **Layer normalization** before attention and before the MLP, which keeps the numbers in a stable range.
- **Residual connections**: The output of attention and of the MLP is added to the vector, instead of replacing it. So each layer adds its own bit of information on top of what was already there.

```
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

This is repeated for all 12 layers.

## Logits

After the 12 layers, we take the vector of the last token (the one at " of" in our example), apply one final layer normalization, and multiply it with the embedding matrix again. This gives us one score for every token in the vocabulary, i.e. 50,257 scores. These scores are called **logits**.

A higher logit means the model thinks that token is more likely to come next. Applying softmax to the logits turns them into probabilities.

## Generation

Now we have to pick the next token from these probabilities. This is called **sampling**:

- **Greedy**: Always pick the token with the highest probability.
- **Temperature**: It controls how random the output is. It makes the probabilities sharper (lower temperature) or flatter (higher temperature) before picking.
- **Top-k / Top-p**: Only pick from the k most likely tokens, or from the smallest set of tokens whose probabilities add up to p.

The picked token is added to the input, and the whole process runs again to get the next token. And again, and again. This loop continues until the model generates a special end token (`<|endoftext|>` for GPT-2) or until we reach the maximum number of tokens we asked for.

With greedy sampling, GPT-2 continues our prompt as: "Michael Jordan played the sport of **basketball for the University of Michigan**...". It does get the sport right, but only barely: `' basketball'` had a probability of just 14%, with `' football'` close behind at 8%. And then it confidently makes up the rest (he actually played for North Carolina). Small models remember some facts, but not all of them, and they don't know which ones they got wrong. Every one of those tokens was generated one at a time.

So, the model generates **one token at a time**, and every new token needs to go through all the layers again. This is an important point where most problems in inference come from. We will study how to make this process more efficient later.

## Prefill and decode

If you look at the process above, it can be divided into two phases:

- **Prefill**: The whole prompt is processed at once. All the prompt tokens go through all the layers together, in parallel, and this gives us the first generated token.
- **Decode**: After that, every new token is generated one at a time. Each step takes the newly generated token, runs it through all the layers, and gives the next token.

From a user's point of view, prefill decides how long it takes for the answer to **start** appearing (time to first token), and decode decides how fast the rest of the answer **streams** in.

### The problems in each phase

**Prefill** has a lot of work to do at once. Every input token goes through every layer, and in attention, every token is compared with every token before it, so the computation increases very rapidly as the prompt gets longer. With longer prompts, like a document or even a long chat history, the time to the first token increases.

**Decode** has a problem too but its quite the opposite. Each step does very little computation, <abbr title="That's only true because of something called the KV cache, which we'll get to at the end. Without it, every step would have to process the whole sequence again, from the first token.">just one token</abbr>, but to do that work, the GPU has to read **all of the model's weights** from its memory. Then it does it again for the next token, and again. Also, the steps can't run in parallel, since each token depends on the previous one. So during decode, the GPU spends most of its time waiting for data to arrive from memory, rather than actually computing.

## GPU: compute bound and memory bound

A GPU has <abbr title="Highly simplified. We will talk about its architecture in future blogs">two important parts</abbr> for us:

- **Compute**: Thousands of small cores that do the arithmetic (multiplying and adding numbers). This is measured in FLOP/s, i.e. floating point operations per second.
- **Memory**: The GPU's own memory (<abbr title="Video RAM: the GPU's own memory, separate from the computer's RAM. My laptop GPU has 6 GB, data-center GPUs have 80 GB or more.">VRAM</abbr>), where the model's weights live. Before the cores can use the weights, they have to be moved from this memory to the cores. This is measured as memory bandwidth, in bytes per second.

Any piece of work on the GPU is limited by one of these two:

- **Compute bound**: There is a lot of arithmetic to do for every byte of data that is read, so the cores are the bottleneck. Making the cores faster would make it faster.
- **Memory bound**: There is very little arithmetic for every byte read, so the cores keep waiting for data. Here, faster memory would help, and faster cores would not.


So by that logic: 
- **Prefill is usually compute bound.** The weights are read once and used for all the prompt tokens together, so there is a lot of arithmetic for every byte read.
- **Decode is usually memory bound.** All the weights are read just to process one token, so there is very little arithmetic for every byte read, and the cores sit idle most of the time.

This is the reason behind a lot of inference optimizations. For example, if many users are being served at the same time, their tokens can be processed together in one decode step (this is called <abbr title="Serving several requests together in the same step, so one read of the weights is used for all of them.">**batching**</abbr> ). The weights are read once and used for all of them, so the same memory read does a lot more work.

## A little about the KV cache

There is one more thing that makes decode work efficiently. In attention, every new token needs the keys and values of all the tokens before it. Recomputing them for the whole sequence at every step would be a huge waste, because they never change once computed.

So they are computed once and stored in GPU memory. This is called the **KV cache**. With it, each decode step only computes the key and value for the new token and reuses the stored ones.

But the KV cache keeps growing with every token and with every user, and it can take a lot of memory. Managing it well is one of the biggest topics in inference, so I will talk about it in detail in another blog.

## Wrapping up

So that was an overview of how inference works: the text is tokenized, converted into embeddings with positional information, passed through layers of attention and MLPs, turned into logits, and then a token is sampled, one at a time. Prefill processes the prompt and is usually compute bound, while decode generates the tokens one by one and is usually memory bound.

The next blog would be a hands-on gpt-2 implementation. In that, I will go into detail about how all the theoretical stuff which I shared in this blog is implemented. I will also try to talk more about topics that I just skimmed over during this introduction. I am very excited to continue this series and I will be very happy if it helps even one person:) 

Thanks a lot for reading till the end! I know it was a long one, but I hope it gave you a clear picture of what happens inside a model during inference.