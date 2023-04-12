# Web predicts stock prices using Long - Short Term Memory algorithm

*Give me some stars please!!!*

<div class="title">Long - Short Term Memory Algorithm</div>

***Link video demo project***  [Click here!!!](https:////youtu.be/dJPBohy9x44)

## Introduction

Long - Short Term Memory (LSTM) is a good algorithm for predicting stock prices. By the way use prices of past we can predict the prices of present and future.



Long - Short Term Memory networks, commonly known as LSTMs, are a special type of RNN that also handle ordered sequence data well, but LSTMs are resistant to vanishing gradients from which to learn dependencies. far.
<image src = "source/a.png" class = "smallimg"></image>

Image 1:  LSTM network 


-	Input: $C t-1 , h t-1 , x t$ . Where x t is the input in the tth state of the model. $C t-1 , h t-1$ are the output of the previous layer.

-	Output: $ C t , h$ t , we call c cell state, h is hidden state. K sign: σ, tanh means that step uses sigmoid, tanh activation function. The multiplication here is element-wise multiplication, the addition is matrix addition. Where: $ f t , i t , O t$ corresponding to forget gate, input gate and output gate.
	-	Forget gate: $f t = σ(U f * x f + W f * h t -1 + b f )$
	-	Input gate: $i t = σ(U i * x t +W i * h t-1 + b i )$
	-	Output gate: $o t = σ (U o * x t + W o * h t-1 + b o )$

## Explanation of the LSTM algorithm

- The LSTM network is an improvement of the traditional regression network, so the model has the following new points:

<image src = "source/c.png" class = "smallimg"></image>

Image 2: Cell state 

Cell state is the horizontal line that runs through the top of the diagram, like a carousel , the memory of an LSTM network . It runs through the entire chain, with only a small linear number of interactions LSTMs are capable of removing or adding information to the cell state, which is carefully regulated by structures called gates.
Portals are an optional way to pass information. They use sigmoid and tanh activation functions. An LSTM has three ports, for protection and control of cell state.

### a.	Forget gate:

<image src = "source/c2.png" class = "smallimg"></image>

Image 3: Forget gate

-	Forget gate : t does not pass through the sigmod layer to make informed decisions about whether to enter the cell state. h value t-1 and x t passing through the sigmod class yields a value between 0 and 1 for each cell state.

-	The sigmoid class outputs numbers from 0 to 1, describing the throughput of each component. A value of 0 means "nothing through", while a value of one means "let everything pass!"

<image src = "source/c3.png" class = "smallimg"></image>

### b.	Input gate: 

<image src = "source/c4.png" class = "smallimg"></image>

Image 4: Input gate

-	Input gate : q determines the new information to be stored in the cell state. Consists of two parts: The sigmod class that decides which values are updated and a tanh class that holds new values that can be added to the cell state .

<image src = "source/c5.png" class = "smallimg"></image>


-	Finally combine the above two to create a new value to update the cell state.

<image src = "source/c6.png" class = "smallimg"></image>


-	Next we update the old cell state C t-1 with C t . Multiply f t forget information to forget and add new values

<image src = "source/c7.png" class = "smallimg"></image>

### c.	Output gate: 

<image src = "source/c8.png" class = "smallimg"></image>

Image 5: Output gate

-	Output gate: q decides what information will be approved. First, we run a sigmoid class, which determines what part of the cell state we should output. Then we set the cell state via tanh function (push the value to range from -1 to 1).

<image src = "source/c9.png" class = "smallimg"></image>


-	Finally multiply by the output of the sigmoid gate to get the necessary information.

<image src = "source/c10.png" class = "smallimg"></image>
							

## Application to stock prediction problem

### 1.	Get data

-	Data is downloaded from yahoo finance
-	From the downloaded dataset we extract the data field that we will use to train
-	After getting the required data, we divide the data into 2 datasets: Data_train and Data_test
	+	Data_train will be used by us in training to create Model
	+	Data_test will be used in evaluating Model

<image src = "source/i.png" class = "smallimg"></image>

Image 1: Data

- The dataset is taken from the finance.yahoo.com package . The information about the stock exchange from March 8, 2010 to October 31, 2021 includes 2795 lines and 7 columns. Data fields:

<image src = "source/ii.png" class = "smallimg"></image>
							

Image 2: Data Netflix stock 

### 2.	Data processing

-	Here, we use the MinMaxScaler function of scikit learn library and scale the data set to numbers in the range (0, 1) to put into the neural network.

### 3. Building LSTM neuron model

- As a first step, we need to instantiate the Sequential class. Sequential is a model where layers are stacked linearly
	The model class of the problem includes the LSTM, Dropout, and Dense classes .
	Above we add 3 consecutive LSTM layers, and every 1 layer is 1 dropout 0.3 . Finally, we pass a Dense layer with 1-dimensional output.

<image src = "source/i1.png" class = "smallimg"></image>
	

Image 3: LSTM neural network 

### 4. Experimental results:

- Accuracy of the model on stock Facebook account:

<image src = "source/i4.png" class = "smallimg"></image>

Image 4: The chart shows the predicted and actual Facebook shares in the period of 2020 - 2021

+	MSE = 47.55213519067378
+	MAE = 5.282921711782391
+	Max = 23.76751708984375
+	M in = 0.061798095703125

The value of Facebook votes tends to increase, from the chart above we can see that the prediction line matches the actual line, the average sum of squares is about 47.55 , the average price difference is low at dollar 5.28 , the price difference is high. approx. 23.7 7 dollar, lowest 0.06 dollar
=> Good predictive model.

## The flow of website

### 1. User interface


<image src = "source/1.png" class = "center"></image>

Image 1: User interface

### __2. Select: __Stocks__, __Field stocks

- To predict price stock you need to input the name of stock and the number of days you want to predict

<image src = "source/2.png" class = "smallimg"></image>

Image 2: Select stocks and field stocks

- At here if we don't have stock you want, we can add it by clicking the button "Another stock" and input file csv of stock.
- Tips: If your stock is new one, you can choose stock already on the market but similar with your stock.

### 3. Choose day to predict: DayBegin, DayEnd

<image src = "source/4.png" atl="choose stock and Field" class = "smallimg"></image>

Image 3: Select stocks and field stocks

### 4. Predict stock price

-	Output: Stock price in future

This train live with data of historical stock prices. So time to predict quite long.

<image src = "source/5.png" class = "smallimg"></image>

Image 4: Predict stock price

<div class ="title">How to install project</div>

## Library required

- [Numpy](https://www.numpy.org/) =  1.21.5
- [Pandas](https://pandas.pydata.org/) = 1.3.5
- [Matplotlib](https://matplotlib.org/) = 3.3.4
- [Streamlit](https://streamlit.io/) = 1.0.0
- [tensorflow](https://www.tensorflow.org/) = 2.7.0
- skikit-learn  = 1.0 
- Yfinance = 0.1.64

<h2> Run tutorial </h2>

<image src="source/6.png" class = "center"> </image>

run web in terminal

<div class="title">Conclusion and comments </div>

- After running the experiment for each stock code, the model gives results with different accuracy. Details for stock code Amazon, Google , T esla : actual price and predicted price are different. Many prediction models are not good, affecting the quality of the transaction, but can predict the up or down trend of stocks. promissory note. As for the code Facebook and Apple , the predicted price is quite similar to the actual price . 

- However, in reality, the stock market depends not only on numbers but also on political factors, domestic and global economic contexts, unexpected shocks (Covid-19 and natural disasters). disaster, crop failure,...), the company's financial performance, etc.

- In addition, the prediction accuracy is still lacking because the LSTM network still has many disadvantages such as: Information must be processed sequentially, only learning information from previous states, cannot learn information. distant due to vanishing gradient.

- Ways to improve and develop direction : add data, add features ( Quarterly profit , Income from service activities, Other operating expenses, General and administrative expenses, ...). Research using algorithms to overcome the disadvantages of LSTM networks .

<div class="title">References </div>


	1.https://en.wikipedia.org/wiki/Average_Average_Number_Available
	Accessed December 20, 2021 Wikipedia
	2. https://ndquy.github.io/posts/cac-phuong-phap-scaling/
	accessed on December 28, 2021 ndquy blog
	3. https://medium.com/analytics-vidhya/long-short-term-memory-networks-23119598b66b
	accessed on December 28, 2021 Author Vinithavn – medium.com
	4. https://medium.com/@asmello/introduction-to-model-evaluation-part-1-regression-and-classification-metrics-e75179d01db
	accessed on December 28, 2021 Author André Mello – medium.com
	5. https://nttuan8.com/bai-14-long-short-term-memory-lstm/
	accessed on 1/12/2021 Author Nguyen Thanh Tuan
	6. https://nttuan8.com/bai-13-recurrent-neural-network/	
	accessed on 1/12/2021 Author Nguyen Thanh Tuan
	7. https://www.kaggle.com/towarddatascience/sample-code?fbclid=IwAR3St8P2IhS6r18Oso18_PYLObLjH03lIUYoFusasFMR0tOKHM4_i0xv2As
	accessed 12/27/2021 – Kaggle
	8. https://viblo.asia/p/optimizer-hieu-sau-ve-cac-thuat-toan-toi-uu-gdsgdadam-Qbq5QQ9E5D8
	accessed on December 27, 2021 – Author Tran Trung Truc
	9. https://streamlit.io
	accessed 12/15/2021 – Streamlit
	10. https://blog.mlreview.com/understanding-lstm-and-its-diagrams-37e2f46f1714
	accessed 12/15/2021 – Author Shi Yan
	11. https://stanford.edu/~shervine/l/en/teaching/cs-230/cheatsheet-recurrent-neural-networks
	accessed 12/15/2021 - Author Shervine Amidi
	12. https://medium.datadriveninvestor.com/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577
	accessed 12/15/2021 - Author Nir Arbel
	13. https://dominhhai.github.io/vi/2018/04/nn-bp
	accessed 11/28/2021- Hai's Blog


# License

Any questions? Feel free to contact me at: vothuongtruongnhon2002@gmail.com

# Author

[Võ Thương Trường Nhơn](https://github.com/truongnhon-hutech)

[Phạm Đức Tài](https://github.com/tai121)

Nguyễn Hồng Thái
