Jak możecie to tutaj wszystko piszcie. Starajmy się trzymać jakiegoś jednego schematu odnośnie formatowania i pisać wszystko po angielsku.

# Question Answering System
*Authors: Dawid Sitnik, Natalia Jakubiak, Piotr Czajkowski*

## Introduction
The problem that we are going to face during our project is question answering challenge where we are given some piece of the context in which we need to find an answer to the particular question. The answer is always quoted from the context. 

The issue seems to be quite trivial while solving by humans, but upon further observation, it demands a lot of complex tasks for a machine, which should finally understand the contextual meaning of each word from context and question as well. Then using an abstract understanding of the question it should extract the correct section of the context.

For this task, we are going to test different techniques of question answering, point their advantages and disadvantages, and decide which one performs the best. The methods which we are going to inspect are:
* **BiDAF model**
* **BERT model**
* **Classical ML approaches like XGboost, Random Forest, Linear Regression, etc.**

## Dataset
Natalia, jak możesz uzupełnij opis datasetu. Przypominam, że działamy na SQuAD Dataset.

## BiDAF Model
### Introduction
To help us understand the BiDAF model lets us first explain the general structure of a neural network which enables the machine to understand the context as well as questions. 

The first layer of the net is called **Embedding Layer** and it is responsible for converting sentences into words and words into its word embedded representation, using pre-trained vector-like *GloVe*. This type of representation is much better than one hot vector representing each word. How our problem we are going to use 100 dimensional *GloVe* word embeddings.

In the second layer, we are going to use **Encoder Layer**, which used for giving each word knowledge about its predecessors and successors. To implement that layer we will use the LSTM network. The output of this part will be the concatenation of a series of hidden vectors in the forward and backward direction. The same layer is used to create hidden vectors for questions.
<p align="center">
  <img src = "https://imgur.com/eAhLaGD.png"/>
</p>

The third used layer is the so-called **Attention Layer** which is used for getting the final answer from two previous layers. This part aims to point to the fraction of the context that responds to the given question. Lets us start with the explanation of the simplest possible attention layer which is *Dot Product Attention* 
<p align="center">
  <img src = "https://imgur.com/jlY04rn.png"/>
</p>
The dot product attention is a multiplication of each context vector *Ci* by each question vector *Qj* which is *Ei*. Then we use softmax over *Ei* getting *alpha i*. This transformation ensures that the sum of all E is equal to 1. Finally, we calculate *Ai* as the dot product of the attention distribution *alpha i* and the corresponding question vector. It can be described by the above equation:
<p align="center">
  <img src = "https://imgur.com/iFEZkk1.png"/>
</p>

The performance of the model can be enhanced by using **BiDAF Attention Layer** instead of the simple one described before. The main idea behind this layer is that the attention flows both directions - from the context to the question and vice versa. 

Firstly, we compute the similiraty matrix NxM which contains similarity score *Sij* for each pair *(ci, qi)*. Sij = wT sim[ci ; qj ; ci ◦ qj ] ∈ R Here, ci ◦ qj is an elementwise product and wsim ∈ R 6h is a weight vector. Described in equation below: 
<p align="center">
  <img src = "https://imgur.com/nHnVUW4.png"/>
</p>
The next action that is performed is Context to Question Attention (similar to the dot product described above). In this case we take the row-wise softmax of S to obtain attention distributions α i , which we use to take weighted sums of the question hidden states q j , yielding C2Q attention outputs a i .
<p align="center">
  <img src = "https://imgur.com/H5pPylu.png"/>
</p>
Next, we perform Question-to-Context Attention. For each context location i ∈ {1, . . . , N}, we take the max of the corresponding row of the similarity matrix, m i = max j Sij ∈ R. Then we take the softmax over the resulting vector m ∈ R N — this gives us an attention distribution β ∈ R N over context locations. We then use β to take a weighted sum of the context hidden states c i — this is the Q2C attention output c prime:
<p align="center">
  <img src = "https://imgur.com/b0SjDeX.png"/>
</p>
At the end the context position c i is combined with output from C2Q and Q2C attentions as described below:
<p align="center">
  <img src = "https://imgur.com/n9ygwhP.png"/>
</p>

The last layer used in our neural network is **Output Layer** which is a softmax layer that helps to decide what is the start and the end index for the answer span. In that part, the context hidden states are combined with the attention vector from the previous layer to create blended reps. These reps are the input to a fully connected layer which is using softmax and a p_end vector with probability for end index. Because we know that in most cases start and end indexes are spaced from each other for maximally 15 words, we look for start and end indexes that maximize *p_start * p_end*.

In that case, the loss function is the sum of the cross-entropy loss for the start and end locations. It is minimized using Adam Optimizer.

### Data Preprocessing
The used SQuAD dataset consists of 2 files:
- train-v2.0.json
- dev-v2.0.json

The data was in the form of triplets - context, question, and its answer span, which is the answer with its start and end indices. Those files were used to generate four new files containing a tokenized version of the question, context, and answer with its span. The important thing about those files is, that their lines are aligned in triplets. Each line in answer span consists of starting and ending indices of the corresponding context in which the answer can be found. 

To obtain vector representation of the text the GloVe Stanford embedding was used. GloVe performs training on aggregated global word-word co-occurrence statistics from a corpus and the resulting representation showcase interesting linear substructures of the word vector space. A word embeddings with dimensionality d = [50, 100, 200, 300], 6B tokens, and vocabulary of size 400k, pre-trained on Wikipedia, and Gigaword were used. Words that couldn't be found in the GloVe dictionary has been treated as 0 vectors. For the tokenization of the words, the basic tokenizer was used. In the end, the context with the question was converted to token ids indexed against the entire vocabulary. 

### Model Configuration
The model was built and trained using TensorFlow, because of its simplicity and abstraction which enabled creating the network by making only small changes to the existing LSTM layer. It also provides sequence to sequence models. In this case, the BahdamuAttention was used. For the intermediate calculations at each time step, a basic attention wrapper was used. Because of its ability to use the moving average of the parameters, the Adam algorithm was used for controlling the learning rate. For controlling the learning process, the gradient was computed and the loss function minimized. 

### Evaluate Metrics 
For model evaluation, we used described in the initial SQuAD paper ExactMatch metric. It measures the percentage of predictions that match one of the ground truth answers exactly.

### Result 
the final model was trained with 30 epochs of batch-size 32. Training each epoch took about 10 hours which gives almost two weeks of training. The exact match of the model equaled 0.60 which is still far behind the best solutions, but it can be still treated as a satisfying result.

