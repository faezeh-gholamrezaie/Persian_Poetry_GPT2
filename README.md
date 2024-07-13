# Persian_Poetry_GPT2

<div align="center">
    <img width="30%" src="https://github.com/faezeh-gholamrezaie/Persian_Poetry_GPT2/blob/main/Persian%20Poetry%20Generator%20using%20GPT2.jpg">
</div>

# Understanding Transformers in GPT Models

The Transformer architecture has revolutionized text generation, forming the foundation for GPT (Generative Pre-trained Transformer) models. Its applications extend beyond language tasks, with uses in image classification, generation, and even protein folding! Let's delve into the three key components of a Transformer and how they work together in GPT models.

<div align="center">
    <img width="30%" src="https://github.com/Mahsa33/GPT2/assets/53941450/c77015a1-afa1-4544-bf3e-3227fe80a471">
</div>

# 1. Embeddings: Bridging the Gap Between Text and Numbers

Imagine feeding raw text into a Transformer – it wouldn't understand! Computers thrive on numbers, so we first convert text into a sequence of numbers through tokenization. This process breaks down text into smaller units called tokens, which can be words, parts of words, or even named entity chunks. Each token receives a unique number, making it easier for the Transformer to process.

GPT-style Transformer models, like GPT-35-turbo (ChatGPT) and GPT-4-32k, have a predefined token limit. For example, GPT-35-turbo is capped at 4096 tokens, while GPT-4-32k can handle up to 32768 tokens. This limit considers both the input tokens and all generated output tokens. In simpler terms, it refers to the total number of tokens the model can process, including what's fed in and what it generates.

# 2. Decoding and Generating Text Step-by-Step

Unlike our expectations of generating entire sentences at once, Transformers create text one token at a time. To build a longer sequence, the model employs a loop. Within each loop iteration, it predicts the most likely next token based on the previous ones. This process continues until reaching the predefined token limit, ensuring the generation doesn't spiral infinitely.

Instead of raw tokens, Transformers leverage embeddings. These are like special codes capturing the meaning and relationships between words. Similar words will have similar embeddings, even if spelled differently. This empowers the model to grasp the context and flow of the text more effectively.

Another crucial step is positional encoding. Unlike models like RNNs (Recurrent Neural Networks) or CNNs (Convolutional Neural Networks), Transformers lack an inherent understanding of word order. Positional encoding injects information about each token's position, allowing the model to comprehend the sequence and sentence structure.

Positional encoding utilizes a clever technique involving sine and cosine functions of varying frequencies to create a unique mathematical representation for each token's position in the sequence. The idea behind this approach is to encode positional information into a continuous vector space, allowing the model to learn the relationships between tokens based on their relative positions.

To illustrate this concept, consider a sequence of tokens: "The cat sat on the mat." Each token is assigned a unique position, ranging from 0 (the first token "The") to 6 (the last token "mat"). Positional encoding then generates a vector representation for each token's position.

The positional encoding vectors are constructed using sine and cosine functions of varying frequencies. The frequency of the sine and cosine functions increases as the token's position progresses. This means that tokens closer to the beginning of the sequence have lower frequency representations, while tokens closer to the end have higher frequency representations.

The reason for using sine and cosine functions is their inherent ability to capture relative positions and distances. As the frequency increases, the sine and cosine waves oscillate more rapidly, allowing them to distinguish between tokens that are closer together and those that are farther apart.

By incorporating these positional encoding vectors into the overall token representations, the Transformer model is able to effectively learn the sequential relationships between words. The model can then leverage this positional information to generate more coherent and contextually relevant text.

In essence, positional encoding with sines and cosines provides a powerful and elegant mechanism for Transformers to understand the order and context of words, enabling them to excel in tasks like text generation and machine translation.

# 3. The Decoder: The Engine of Text Generation

The decoder serves as the heart of a Transformer in GPT models. It comprises multiple layers that collaborate to predict the next token. Each layer has two key components:

Attention: This ingenious mechanism allows the model to "focus" on specific parts of the previously generated text, considering their relevance to the token being predicted. It's like the model is reviewing its own writing to decide what comes next.

Feed-forward step: This step takes the information gleaned from the attention mechanism and processes it further to generate the predicted token.

Full Transformers vs. GPT-style Transformers

The original Transformer architecture was designed for machine translation, where it understands one language (encoder) and translates it into another (decoder). GPT-style Transformers differ slightly. They primarily rely on a special type of attention called "masked self-attention," where the model only focuses on previously generated tokens when predicting the next one. This is because GPT models are trained to generate text based on a starting prompt or seed, rather than translating an entire source text.

# Example:

To explain the detailed steps of the Attention Mechanism with the given example and related calculations, let's go through each step in detail:

### 1. Word Embedding
Each word in the sentence is represented by an embedding vector. Let's assume the embedding vectors for the words are as follows:

```
The  -> [1, 0, 0, 1]
Cat  -> [0, 1, 0, 0]
Sat  -> [0, 0, 1, 0]
Mat  -> [1, 0, 1, 0]
```

### 2. Positional Embedding
To preserve the information about the order of the words in the sentence, positional embeddings are added to the word embeddings. Let's assume the positional embeddings for the words are as follows:
Here is the English translation of the provided Persian text:

For each position pos and dimension i:

<div align="center">
    <img width="40%" src="https://github.com/Mahsa33/GPT2/assets/53941450/dba83f3c-6531-4d38-8d0e-d69d2dc38b1b">
</div>

<div align="center">
    <img width="40%" src="https://github.com/Mahsa33/GPT2/assets/53941450/4de3d842-3469-43a9-a389-ffce222c495d">
</div>


### 3. Combining Word and Positional Embeddings
The word embeddings and positional embeddings are added together to form the final vectors, which contain both semantic and positional information:

```
The -> [1 + 0, 0 + 1, 0 + 0, 1 + 1] = [1, 1, 0, 2]
Cat -> [0 + 0.841, 1 + 0.540, 0 + 0.001, 0 + 0.999] = [0.841, 1.540, 0.001, 0.999]
Sat -> [0 + 0.909, 0 + (-0.416), 1 + 0.002, 0 + 0.998] = [0.909, -0.416, 1.002, 0.998]
Mat -> [1 + 0.141, 0 + (-0.990), 1 + 0.003, 0 + 0.997] = [1.141, -0.990, 1.003, 0.997]
```

### 4. Multi-Head Attention Mechanism
When using the multi-head attention mechanism, several attention vectors are generated for each word (in this case, the word "The"). Each attention head independently calculates a set of Query, Key, and Value vectors, and produces the attention scores and corresponding attention vectors. These attention vectors from different heads are then combined.

### Calculating Attention Vectors with Multiple Heads

Let's assume there are 3 attention heads in this example. For each word, each attention head separately calculates its Query, Key, and Value vectors and then computes the attention scores and corresponding attention vectors. The same steps are followed for each head.

#### Head 1:
**Query:**
```
The  -> [1, 0, 0, 1]
Cat  -> [0, 1, 0, 0]
Sat  -> [0, 0, 1, 0]
Mat  -> [1, 0, 1, 0]
```

**Key:**
```
The  -> [1, 0, 1, 0]
Cat  -> [0, 1, 0, 1]
Sat  -> [1, 0, 0, 1]
Mat  -> [0, 1, 1, 0]
```

**Value:**
```
The  -> [1, 0, 0, 1]
Cat  -> [0, 2, 0, 0]
Sat  -> [0, 0, 3, 0]
Mat  -> [4, 0, 0, 0]
```

#### Head 2:
**Query:**
```
The  -> [0, 1, 1, 0]
Cat  -> [1, 0, 0, 1]
Sat  -> [0, 1, 0, 0]
Mat  -> [0, 0, 1, 1]
```

**Key:**
```
The  -> [0, 1, 0, 1]
Cat  -> [1, 0, 1, 0]
Sat  -> [0, 1, 1, 0]
Mat  -> [1, 0, 0, 1]
```

**Value:**
```
The  -> [2, 1, 0, 0]
Cat  -> [1, 0, 2, 0]
Sat  -> [0, 3, 0, 1]
Mat  -> [0, 0, 1, 2]
```

#### Head 3:
**Query:**
```
The  -> [0, 0, 1, 1]
Cat  -> [1, 1, 0, 0]
Sat  -> [1, 0, 0, 1]
Mat  -> [0, 1, 1, 0]
```

**Key:**
```
The  -> [1, 1, 0, 0]
Cat  -> [0, 0, 1, 1]
Sat  -> [1, 0, 1, 0]
Mat  -> [0, 1, 0, 1]
```

**Value:**
```
The  -> [1, 0, 2, 1]
Cat  -> [0, 1, 0, 2]
Sat  -> [3, 0, 1, 0]
Mat  -> [0, 2, 1, 0]
```

### Calculating Attention Scores and Attention Vectors for Each Head

Let's assume we only calculate the attention scores and attention vectors of Head 1 for the word "The":

**Head 1:**

Attention scores for the word "The":
```
(The, The)  -> 1*1 + 0*0 + 0*1 + 1*0 = 1
(The, Cat)  -> 1*0 + 0*1 + 0*0 + 1*1 = 1
(The, Sat)  -> 1*1 + 0*0 + 0*0 + 1*1 = 2
(The, Mat)  -> 1*0 + 0*1 + 0*1 + 1*0 = 0
```

Normalizing attention scores using Softmax:
```
Softmax (Attention scores "The"):
(The, The) -> softmax(1) = 0.24
(The, Cat) -> softmax(1) = 0.24
(The, Sat) -> softmax(2) = 0.52
(The, Mat) -> softmax(0) = 0.10
```

Calculating the attention vector for "The":
```
Output "The":
= 0.24 * Value(The) + 0.24 * Value(Cat) + 0.52 * Value(Sat) + 0.10 * Value(Mat)
= 0.24 * [1, 0, 0, 1] + 0.24 * [0, 2, 0, 0] + 0.52 * [0, 0, 3, 0] + 0.10 * [4, 0, 0, 0]
= [0.24, 0, 0, 0.24] + [0, 0.48, 0, 0] + [0, 0, 1.56, 0] + [0.4, 0, 0, 0]
= [0.64, 0.48, 1.56, 0.24]
```

### Combining Results of Different Heads

Similar calculations are performed for each head, resulting in separate attention vectors. These vectors are then combined. Typically, this combination is done by concatenation or summation.

Assuming the calculated attention vectors for each head are as follows:
```
Head 1: [0.64, 0.48, 1.56, 0.24]
Head 2: [1.00, 0.50, 0.75, 0.25]
Head 3: [0.30, 0.70, 1.00, 0.90]
```

These vectors are combined (for simplicity, let's assume concatenation is used):
```
Final vector "The": [0.64, 0.48, 1.56, 0.24, 1.00, 0.50, 0.75, 0.25, 0.30, 0.70, 1.00, 0.90]
```

### 6. Using the Final Vectors in the Model
The combined final vector is fed into subsequent layers of the model (such as FFN layers) for further processing, ultimately producing the model's final output. This final output can be used for next-word prediction, classification, or any other task for which the model is trained.

reference : https://bea.stollnitz.com/blog/gpt-transformer/


