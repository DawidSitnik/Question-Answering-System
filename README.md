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

# BiDAF Model
## Introduction
To help us understanding BiDAF model lets us first eqplain the general structure of neural network which enables the machine understanding the context as well as questions. 

The first layer of the net is called **Embedding Layer** and it is responsible for converting sentences into words and words into its word embeded representation, using pretrained vector like *GloVe*. This type of representation is much better than one hot vector representing each word. How our problem we are going to use 100 dimensional *GloVe* word embedings.

In the second layer we are going to use **Encoder Layer**, which used for giving each word a knowledge about its predecessors and succesors. To implement that layer we will use LSTM network. The output of this part will be the concatination of series of hidden vectors in forward and backward direction. The same layer is used to create hidden vectors for questions.
<p align="center">
  <img src = "https://imgur.com/eAhLaGD.png"/>
</p>

