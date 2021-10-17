![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

## TF-IDF
![deep](https://user-images.githubusercontent.com/12748752/134754236-8d5549c9-bd05-408d-ba63-0d56ab83c999.png)

* TF-IDF stands for **“Term Frequency — Inverse Document Frequency”**.
* This is a technique to quantify words in a set of documents.
* Raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
* We generally compute a score for each word to signify its importance in the document and corpus. 
* This method is a widely used technique in Information Retrieval and Text Mining.

* **TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)**
### Term Frequency
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)
* This measures the frequency of a word in a document.
* This highly depends on the length of the document and the generality of the word, for example, a very common word such as “was” can appear multiple times in a document. 
* But if we take two documents with 100 words and 10,000 words respectively, there is a high probability that the common word “was” is present more in the 10,000 worded document.
* But we cannot say that the longer document is more important than the shorter document. 
* For this exact reason, we perform normalization on the frequency value, we divide the frequency with the total number of words in the document.

#### Our Corpus:
![light](https://user-images.githubusercontent.com/12748752/134754235-ae8efaf0-a27a-46f0-b439-b114cbb8cf3e.png)

* **Document a='The sky is blue'**     
* **Document b='The sky is not blue'**

* **Step 1**
> <img src="https://latex.codecogs.com/svg.image?\begin{matrix}&space;&&space;\mathbf{TF(a)}&space;&&space;\mathbf{TF(b)}&space;\\the&space;&1&space;&space;&&space;1&space;\\sky&&space;1&space;&1&space;&space;\\is&space;&1&space;&space;&&space;1&space;\\blue&space;&1&space;&space;&1&space;&space;\\not&space;&0&space;&space;&1&space;&space;\\\end{matrix}" title="\begin{matrix} & \mathbf{TF(a)} & \mathbf{TF(b)} \\the &1 & 1 \\sky& 1 &1 \\is &1 & 1 \\blue &1 &1 \\not &0 &1 \\\end{matrix}" />


* **Step 2** On a large document the frequency of the terms will be much higher than the smaller ones. Hence we need to normalize the document based on its size.

> <img src="https://latex.codecogs.com/svg.image?\begin{matrix}&space;&&space;\mathbf{TF(a)}&space;&&space;\mathbf{TF(b)}&&space;\mathbf{N(a)}&&space;\mathbf{N(b)}&space;\\the&space;&1&space;&space;&&space;1&space;&1/4&1/5&space;\\sky&&space;1&space;&1&1/4&1/5&space;&space;\\is&space;&1&space;&space;&&space;1&1/4&1/5&space;\\blue&space;&1&space;&space;&1&1/4&1/5&space;&space;\\not&space;&0&space;&space;&1&&space;0&space;&1/5&space;\\\end{matrix}" title="\begin{matrix} & \mathbf{TF(a)} & \mathbf{TF(b)}& \mathbf{N(a)}& \mathbf{N(b)} \\the &1 & 1 &1/4&1/5 \\sky& 1 &1&1/4&1/5 \\is &1 & 1&1/4&1/5 \\blue &1 &1&1/4&1/5 \\not &0 &1& 0 &1/5 \\\end{matrix}" />


* **Step 3**
 <img src="https://latex.codecogs.com/svg.image?\textbf{IDF}&space;=&space;1\&space;&plus;\&space;\log_e&space;\frac{\textrm{Total&space;Number&space;of&space;Documents}}{&space;\textrm{Number&space;of&space;Documents&space;with&space;term&space;'t'&space;appears}&space;}" title="\textbf{IDF} = 1\ +\ \log_e \frac{\textrm{Total Number of Documents}}{ \textrm{Number of Documents with term 't' appears} }" />

> <img src="https://latex.codecogs.com/svg.image?\begin{matrix}&space;&&space;\mathbf{TF(a)}&space;&&space;\mathbf{TF(b)}&&space;\mathbf{N(a)}&&space;\mathbf{N(b)}&&space;\mathbf{IDF}&space;\\the&space;&1&space;&space;&&space;1&space;&1/4&1/5&&space;1&plus;\log_e\frac{2}{2}&space;\\sky&&space;1&space;&1&1/4&1/5&space;&&space;1&plus;\log_e\frac{2}{2}&space;\\is&space;&1&space;&space;&&space;1&1/4&1/5&&space;1&plus;\log_e\frac{2}{2}&space;\\blue&space;&1&space;&space;&1&1/4&1/5&&space;1&plus;\log_e\frac{2}{2}&space;&space;\\not&space;&0&space;&space;&1&&space;0&space;&1/5&&space;1&plus;\log_e\frac{2}{1}&space;\\\end{matrix}" title="\begin{matrix} & \mathbf{TF(a)} & \mathbf{TF(b)}& \mathbf{N(a)}& \mathbf{N(b)}& \mathbf{IDF} \\the &1 & 1 &1/4&1/5& 1+\log_e\frac{2}{2} \\sky& 1 &1&1/4&1/5 & 1+\log_e\frac{2}{2} \\is &1 & 1&1/4&1/5& 1+\log_e\frac{2}{2} \\blue &1 &1&1/4&1/5& 1+\log_e\frac{2}{2} \\not &0 &1& 0 &1/5& 1+\log_e\frac{2}{1} \\\end{matrix}" />

* **Step 4**

> <img src="https://latex.codecogs.com/svg.image?\begin{matrix}&space;&&space;\mathbf{TF(a)}&space;&&space;\mathbf{TF(b)}&&space;\mathbf{N(a)}&&space;\mathbf{N(b)}&&space;\mathbf{IDF}&&space;\mathbf{TF&space;\&space;*&space;\&space;IDF}&&space;\mathbf{Document(a)}&space;&&space;\mathbf{Document(b)}\\the&space;&1&space;&space;&&space;1&space;&1/4&1/5&&space;1&plus;\log_e\frac{2}{2}&&space;&&space;.25&.2&space;\\sky&&space;1&space;&1&1/4&1/5&space;&&space;1&plus;\log_e\frac{2}{2}&&space;&&space;.25&.2&space;&space;\\is&space;&1&space;&space;&&space;1&1/4&1/5&&space;1&plus;\log_e\frac{2}{2}&&space;&&space;.25&.2&space;&space;\\blue&space;&1&space;&space;&1&1/4&1/5&&space;1&plus;\log_e\frac{2}{2}&&space;&&space;.25&.2&space;&space;&space;\\not&space;&0&space;&space;&1&&space;0&space;&1/5&&space;1&plus;\log_e\frac{2}{1}&&space;&&space;0&.1386294&space;&space;\\\end{matrix}" title="\begin{matrix} & \mathbf{TF(a)} & \mathbf{TF(b)}& \mathbf{N(a)}& \mathbf{N(b)}& \mathbf{IDF}& \mathbf{TF \ * \ IDF}& \mathbf{Document(a)} & \mathbf{Document(b)}\\the &1 & 1 &1/4&1/5& 1+\log_e\frac{2}{2}& & .25&.2 \\sky& 1 &1&1/4&1/5 & 1+\log_e\frac{2}{2}& & .25&.2 \\is &1 & 1&1/4&1/5& 1+\log_e\frac{2}{2}& & .25&.2 \\blue &1 &1&1/4&1/5& 1+\log_e\frac{2}{2}& & .25&.2 \\not &0 &1& 0 &1/5& 1+\log_e\frac{2}{1}& & 0&.1386294 \\\end{matrix}" />

* **The Matrix**
> <img src="https://latex.codecogs.com/svg.image?\begin{matrix}&&space;\mathbf{the}&space;&&space;\mathbf{sky}&&space;\mathbf{is}&space;&\mathbf{blue}&space;&\mathbf{not}&space;\\Document(a)&&space;.25&&space;.25&&space;0&&space;.25&&space;.25\\Document(b)&&space;.2&&space;.2&&space;0.1386294&&space;.2&&space;.2\end{matrix}" title="\begin{matrix}& \mathbf{the} & \mathbf{sky}& \mathbf{is} &\mathbf{blue} &\mathbf{not} \\Document(a)& .25& .25& 0& .25& .25\\Document(b)& .2& .2& 0.1386294& .2& .2\end{matrix}" />
