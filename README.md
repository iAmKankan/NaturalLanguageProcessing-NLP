# Natural Language Processing
* **Natural language processing (NLP)**: It is the branch of artificial intelligence that helps computers understand, interpret and manipulate human language. 
* NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.

## Main approaches in NLP

* In order to comprehend and interpret human language, NLP adopts 3 essential approaches to execute NLP tasks.
* All the 3 approaches are well-recognized and are widely used across numerous segments. 
* The evolutionary approaches have standalone benefits and continue to aid NLP task to deliver best results.

**1. Rule-Based Approach:**
    
   - Regular expressions
   - Context-free grammars

A rule-based approach is perfect to acquire a specific language phenomenon: it efficiently decodes the linguistic relationship between words to translate the sentence.

* It is easily achieved through focus on pattern-matching or parsing.
* It can be counted as the ‘fill in the blanks’ method.
* It offers high performance cases when used specifically but fails to impress when generalized.

Hence, it’s vital to make a good choice for query analysis that’s meant to perform.

**2. Machine Learning or ‘Traditional’ Approach:**

Traditional machine learning approach is described by:

* Training data or annotated corpus – one that has a corpus with mark-up.
* Feature engineering – capitalized, singulars, word type, surrounding words, etc.
* A general ML system of a training set or training a model on defined parameters, followed by fitting on test data.
* Applying model to test data or inference which is characterized by finding most probable words, next word, best category, etc.
* ‘Semantic slot filling’ tag the words or tokens which carry meaning to the sentences or translate utterances to logical form.



* Likelihood maximization
* Linear classifiers




| Rule-based Grammar                         	| Machine Learning algorithm                	|
|:--------------------------------------------	|:-------------------------------------------	|
| **Advantages**                               	| **Advantages**                               	|
| It’s easily adaptable                      	| It’s can scale effortlessly               	|
| Simple to debug                            	| Learnability without clear programming    	|
| Enormous training corpus not needed        	| Quick development if dataset is available 	|
| Comprehends the language                   	| High recall coverage                      	|
| High perfection                            	|                                           	|
|**Disadvantages**                             	| **Disadvantages**                            	|
| Proficient developers & linguists required 	| Training corpus with annotation needed    	|
| Slow parser development                    	| Hard to debug                             	|
| Moderate recall (coverage)                 	| Zero understanding of the language        	|


**3. Deep Learning**

* Recurrent Neural Networks (RNNs)
*  Convolutional Neural Networks (CNNs)

* This is similar to "traditional" machine learning, but with a few differences:

    * feature engineering is generally skipped, as networks will "learn" important features (this is generally one of the claimed big benefits of using neural networks for NLP)
    * instead, streams of raw parameters ("words" -- actually vector representations of words) without engineered features, are fed into neural networks
    * very large training corpus
 
---
NLP Terminology is based on the following factors:

* **Weights and Vectors:**
    - TF-IDF,
    - length(TF-IDF, doc),
    - Word Vectors, 
    - Google Word Vectors

* **Text Structure:**
    - Part-Of-Speech Tagging, 
    - Head of sentence,
    - Named entities

* **Sentiment Analysis:**
    - Sentiment Dictionary,
    - Sentiment Entities,
    - Sentiment Features

* **Text Classification:**
    - Supervised Learning, 
    - Train Set, 
    - Dev(=Validation) Set, 
    - Test Set, 
    - Text Features,
    - LDA.

* **Machine Reading:**
    * Entity Extraction, 
    * Entity Linking,
    * dbpedia, 
    * FRED (lib) / Pikes.

---



* For solving NLP problems we can use two types of approaches **1. Word level**, **2. Character level**.







### Libraries we used for NLP
* We usually use these libraries in NLP, which are:
  * NLTK (Natural language Tool kit)
  * TextBlob
  * CoreNLP
  * Polyglot
  * Gensim
  * SpaCy
  * Scikit-learn
  * And the new one is **Megatron library** launched recently.


### Tokenization
* Tokenisation is the act of breaking a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements called tokens. 
* Tokens can be individual words, phrases or even whole sentences. 
* In the process of tokenisation, some characters like punctuation marks are discarded.

```
'Natural Language Processing'
['Natural', 'Language', 'Processing']
```
### Stemming
* **Stemming**: It is the process of reducing inflexions in words to their root forms such as mapping a group of words to the same stem even if stem itself is not a valid word in the Language.

|    | Word        | StemWord |
|----|-------------|----------|
| 0  | Connect     | Connect  |
| 1  | Connections | Connect  |
| 2  | Connection  | Connect  |
| 3  | Connects    | Connect  |
| 04 | Connected   | Connect  |

### Lemmatisation
* **Lemmatisation**: It is the process of the group together the different inflected forms of the word so that they can be analysed as a single item. It is quite similar to stemming, but it brings context to the words. So it links words with similar kind meaning to one word.

![Lammi](lammi.png)

## Word Embeddings or Word vectorization


### Bag-of-words model
* We need the way to represent text data for the machine learning algorithms(In vector of number), and the bag-of-words model helps us to achieve the task.
* It is the way of extracting features from the text for the use in machine learning algorithms.
* **In this approach, we use the tokenised words for each of observation and find out the frequency of each token.**

---
* Let’s do an example to understand this concept in depth.
```
“It is going to rain today.”
“Today, I am not going outside.”
“I am going to watch the season premiere.”
```
* We treat each sentence as the separate document and we make the list of all words from all the three documents excluding the punctuation.
* We get- 
```
[‘It’, ’is’, ’going’, ‘to’, ‘rain’, ‘today’ ‘I’, ‘am’, ‘not’, ‘outside’, ‘watch’, ‘the’, ‘season’, ‘premiere.’]
```
* The next step is the create vectors. Vectors convert text that can be used by the machine learning algorithm.
* We take the first document — **“It is going to rain today”**, and we check the frequency of words from the ten unique words.
```
“It” = 1
“is” = 1
“going” = 1
“to” = 1
“rain” = 1
“today” = 1
“I” = 0
“am” = 0
“not” = 0
“outside” = 0
```
* Rest of the documents will be:
```
“It is going to rain today” = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0] 
“Today I am not going outside” = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1] 
“I am going to watch the season premiere” = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
```
* In this approach, each word (a token) is called a **“gram”**.
* Creating the vocabulary of two-word pairs is called a bigram model. 
* The process of converting the NLP text into numbers is called **vectorisation** in ML.
* There are different ways to convert text into the vectors :
 - Counting the number of times that each word appears in the document.
 - I am calculating the frequency that each word appears in a document out of all the words in the document.





## spaCy:
* This is completely optimized and highly accurate library widely used in deep learning.






## NLP-NLTK
