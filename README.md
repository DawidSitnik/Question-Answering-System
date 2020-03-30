Jak możecie to tutaj wszystko piszcie. Starajmy się trzymać jakiegoś jednego schematu odnośnie formatowania i pisać wszystko po angielsku.

# Question Answering System
*Authors: Dawid Sitnik, Natalia Jakubiak, Piotr Czajkowski*

## Introduction
The problem that we are going to face during our project is question answering challenge where we are given some piece of the context in which we need to find an answer to the particular question. The answer is always quote from the context. 

The issue seems to be quite trivial while solving by humans, but upon further observation, it demands a lot of complex tasks for a machine, which should finally understand the contextual meaning of each word from context and question as well. Than using an abstract understanding of the question it should extract the correct section of the context.

For this task we are going to test different techniques of question answering, point their adventages and disadventages and decide which one performs the best. The methods which we are going to inspect are:
* **BiDAF model**
* **BERT model**
* **Classical ML approaches like xGBoost, Random Forest, Linear Regression etc.**

## Dataset
Natalia, jak możesz uzupełnij opis datasetu. Przypominam, że działamy na SQuAD Dataset.

## BiDAF Model
### Introduction
To help us understanding BiDAF model lets us first eqplain the general structure of neural network which enables the machine to understand the context as well as questions. 

The first layer of the net is called **Embedding Layer** and it is responsible for converting sentences into words and words into its word embeded representation, using pretrained vector like *GloVe*. This type of representation is much better than one hot vector representing each word. How our problem we are going to use 100 dimensional *GloVe* word embedings.

In the second layer we are going to use **Encoder Layer**, which used for giving each word a knowledge about its predecessors and succesors. To implement that layer we will use LSTM network. The output of this part will be the concatination of series of hidden vectors in forward and backward direction. The same layer is used to create hidden vectors for questions.
<p align="center">
  <img src = "https://imgur.com/eAhLaGD.png"/>
</p>

The third used layer is so called **Attention Layer** which is used for getting the final answer from two previous layers. The aim of this part is to point to the praction of the context which responds to the given question. Lets us start with the explantion of the simplest possible attention layer which is *Dot Product Attention* 
<p align="center">
  <img src = "https://imgur.com/jlY04rn.png"/>
</p>
The dot product attention is a multiplication of each context vector *Ci* by each question vector *Qj* which is *Ei*. Then we use softmax over *Ei* getting *alpha i*. This transformation ensures that sum of all E is equal to 1. Finally we calculate *Ai* as the dot product of the attention distribution *alpha i* and the corresponding question vector. It can be described by above equation:
<p align="center">
  <img src = "https://imgur.com/iFEZkk1.png"/>
</p>

The performence of the model can be enchanced by using **BiDAF Attention Layer** instead of the simple one described before. The main idea behind this layer is that the attention flows both directions - from the context to the question and vice versa. 

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

The last layer used in our neural network is **Output Lasyer** which is softmax layer that helps deciding what is the start and the end index for the answer span. In that part the context hidden states are combined with attention vector from the previous layer to create blended reps. These reps are the input to a fully connected layer which is using softmax and a p_end vector with probability for end index. Because we know that in most cases start and end indexes are spaced from each others for maximally 15 words, we look for start and end indexes which maximize *p_start * p_end*.

In that case the loss function is the sum of the cross-entropy loss for the start and end locations. It is minimized using Adam Optimizer.


