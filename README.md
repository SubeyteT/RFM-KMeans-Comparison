# RFM-KMeans-Comparison
![image](https://user-images.githubusercontent.com/83431435/127213705-93329df8-e45f-4900-aa97-d6e217f2d8e4.png)

RFM and K Means models are applied to Online Retail Dataset II.

**Aim of the project:** Rule-based customer segmentation method RFM and machine learning method K-Means are expected to be compared for customer segmentation.

## Dataset:

The dataset named Online Retail II includes the sales of an UK-based online store between 01/12/2009 - 09/12/2011. (Project is based on 2010-2011 data)
The product catalog of this company includes souvenirs.
The vast majority of the company's customers are corporate customers.

### Variables:
InvoiceNo: Invoice number. The unique number of each transaction, namely the invoice. Aborted operation if it starts with C.
StockCode: Product code. Unique number for each product.
Description: Product name
Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
InvoiceDate: Invoice date and time.
UnitPrice: Product price (in GBP)
CustomerID: Unique customer number
Country: Country name. Country where the customer lives.

## RFM Analysis:

RFM Analysis is a technique used for customer segmentation. It enables customers to be divided into groups based on their purchasing habits and to develop strategies specific to these groups. 

**Metrics:**
R: Recency (innovation): Time from customer's last purchase to date.
F: Frequency: Total number of purchases.
M: Monetary: Customer's total expenditure.

**RFM Scores:**
![image](https://user-images.githubusercontent.com/83431435/127220745-8ac7ec88-24f4-446d-8dba-0249e40ea5d3.png)

Hibernating -> R:[1-2], F:[1-2]
At Risk -> R:[1-2], F:[3-4]
Can't lose -> R:[1-2], F:5
About to Sleep -> R:3 , F:[1-2]
Need Attention -> R:3 , F:3
Loyal Customers -> R:[3-4], F:[4-5]
Promising -> R:4, F: 1
New_Customers -> R:5, F: 1
Potential Loyalists -> R:[4-5], F:[2-3]
Champions-> R:5, F:[4-5]

RFM score facilitates customer tracking and comparability. Marketing strategies can be developed based on each segment. 

## K-Means:

![Untitled](https://user-images.githubusercontent.com/83431435/127221079-53ca80dc-7e4e-41ff-b0bb-bda3154c3c95.png)

K - Means is an unsupervised learning machine learning algorithm. Unsupervised learning is a type of algorithm where ther isn't labeling and finding out relationships of variables. K- Means is a distance based algorithman and its purpose is to separate the observations into clusters according to their similarity to each other. The clusters should be chosen to be heterogeneous with each other and homogeneous within themselves. 

Determination of optimum number of clusters and iteration numbers is important. We can find it with the model and elbow method. But developer has to have an eye on these hyperparameters due to industry knowledge's importance. 





