## Index
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
* [Word Embedding (word2vec)](#word-embedding-word2vec)
## Natural Language Processing: Pretraining
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

* Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. 
<img src="https://user-images.githubusercontent.com/12748752/139561324-2b923a98-80bd-49f7-8f74-632563bab76f.png" />

## Word Embedding (word2vec)
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
*  Word vectors are vectors used to represent words of Natural Languages.
* It can also be considered as feature vectors or representations of words.
* **The technique of mapping words to real vectors is called word embedding**.

#### Avoid One-Hot Vectors
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* We used one-hot vectors to represent words (characters are words) .
* Suppose that the number of different words in the dictionary (the dictionary size) is  N
* Each word is represented as a vector of length  N , and it can be used directly by neural networks.
#### Why One-Hot Vectors are bad
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* A main reason is that one-hot word vectors cannot accurately express the similarity between different words, such as the cosine similarity that we often use. 
* Since the cosine similarity between one-hot vectors of any two different words is 0, one-hot vectors cannot encode similarities among words.

### Self-Supervised word2vec
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* **The word2vec tool was proposed to address the issue with using One-Hot Vector.**
* It maps each word to a fixed-length vector, and these vectors can better express the similarity and analogy relationship among different words.

> #### The word2vec tool contains two models, namely-
>> 1) Skip-gram [Mikolov et al]= We predict the context words from the target
>> 
>> 2) Continuous bag of words (CBOW) [Mikolov et al]= We predict the target word from the context.

* For semantically meaningful representations, their training relies on conditional probabilities that can be viewed as predicting some words using some of their surrounding words in corpora. 
> **Since supervision comes from the data without labels, both skip-gram and continuous bag of words are self-supervised models.**

#### 1) The Skip-Gram Model
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* **The skip-gram model assumes that a word can be used to generate its surrounding words in a text sequence.**
> #### Take the text sequence “the”, “man”, “loves”, “his”, “son” as an example. 
> * Let us choose “loves” as the center word and set the context window size to 2. 
> * Given the center word “loves”, the skip-gram model considers the conditional probability for generating the context words:
> * “the”, “man”, “his”, and “son”, which are no more than 2 words away from the center word:

>> <img src="https://latex.codecogs.com/svg.image?P(\textrm{''the''},&space;\textrm{''man''},&space;\textrm{''his''},&space;\textrm{''son''}&space;\mid&space;\textrm{''loves''})" title="P(\textrm{''the''}, \textrm{''man''}, \textrm{''his''}, \textrm{''son''} \mid \textrm{''loves''})" />

* Assume that the context words are independently generated given the center word (i.e., conditional independence). 
* In this case, the above conditional probability can be rewritten as

>> <img src="https://latex.codecogs.com/svg.image?\mathrm{P(\textrm{''the''}\mid\textrm{''loves''})\cdot&space;P(\textrm{''man''}\mid\textrm{''loves''})\cdot&space;P(\textrm{''his''}\mid\textrm{''loves''})\cdot&space;P(\textrm{''son''}\mid\textrm{''loves''}).}" title="\mathrm{P(\textrm{''the''}\mid\textrm{''loves''})\cdot P(\textrm{''man''}\mid\textrm{''loves''})\cdot P(\textrm{''his''}\mid\textrm{''loves''})\cdot P(\textrm{''son''}\mid\textrm{''loves''}).}" />


> ![skip-gram](https://user-images.githubusercontent.com/12748752/139602656-549ebe0a-e0b3-4083-84c0-fa415ac8246b.png)

### 2) The Continuous Bag of Words (CBOW) Model
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* CBOW model is similar to the skip-gram model.
*  The major difference from the skip-gram model is that the continuous bag of words model assumes that a center word is generated based on its surrounding context words in the text sequence.
*   For example, in the same text sequence “the”, “man”, “loves”, “his”, and “son”, with “loves” as the center word and the context window size being 2, the continuous bag of words model considers the conditional probability of generating the center word “loves” based on the context words “the”, “man”, “his” and “son” (as shown in fig_cbow), which is

> <img src="https://latex.codecogs.com/svg.image?P(''loves''|''the'',''man'',''his'',''son'')" title="P(''loves''|''the'',''man'',''his'',''son'')" />


