######################
# Customer Segmentation with K-Means and RFM
######################

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from helpers import *

pd.set_option('display.width', value)
pd.set_option('display.max_rows', value)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', value)


###################
# DATA
###################
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df.head()
df.isnull().sum()
df.info()
df.describe().T


# Data Preprocessing
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df.shape

###################
# RFM METRİCS CREATING
###################

# Recency (innovation): Time from customer's last purchase to date
# Frequency: Total number of purchases.
# Monetary: Customer's total expenditure.

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11) # for recency calculation.

df["TotalPrice"] = df["Quantity"] * df["Price"]

df = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                    "Invoice": lambda Invoice: Invoice.nunique(),
                                    "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
df.columns = ["recency", "frequency", "monetary"]

df = df[df["monetary"] > 0]   # monetary has to be greater than 0

df.head()

###################
# RFM ANALYSIS
###################

##### RFM Scores:

rfm = df.copy()

rfm["RecencyScore"] = pd.qcut(rfm["recency"], q=5, labels=[5,4,3,2,1])
rfm["FrequencyScore"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1,2,3,4,5])
rfm["MonetaryScore"] = pd.qcut(rfm["monetary"], q=5, labels=[5,4,3,2,1])

rfm["RFM_SCORE"] = (rfm["RecencyScore"].astype(str) +
                    rfm["FrequencyScore"].astype(str))

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

##### RFM Segments:
df["Segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
df["Segment"].value_counts()

# Grouping RFM mean and frequency values by segment for dataframe review
df[["Segment", "recency", "frequency", "monetary"]].groupby("Segment").agg(["mean", "count"])

###################
# K - MEANS
###################

km = df.copy()
km = km.drop("Segment", axis=1)

sc = MinMaxScaler((0, 1))
km = sc.fit_transform(km)
km[0:5]

kmeans = KMeans()
k_fit = kmeans.fit(km)

k_fit.get_params()

k_fit.n_clusters # how many clusters? 8
k_fit.cluster_centers_ # centers
k_fit.labels_  # class of each observation belongs.
k_fit.inertia_ ## Total SSE value  = 10.878611159797698

################################
# Determination of Optimum Number of Clusters
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(km)
    ssd.append(kmeans.inertia_)

# errors for each k values
ssd  # sum of squared distances

plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Against Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(km)
elbow.show()

elbow.elbow_value_  # = 6

################################
# Creating Final Clusters
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(km)

# clusters of each observation unit:
clusters = kmeans.labels_

# only indexes:

pd.DataFrame({"Customers": df.index, "Clusters": clusters})

df["cluster_no"] = clusters

df.head()

###################
# COMPARISON OF BOTH MODELS' OUTPUTS
###################

df["cluster_no"] = df["cluster_no"] + 1 # we don't want it to start with 0

df.groupby("cluster_no").agg({"cluster_no": "count"})

for i in df["cluster_no"]:
    df[df["cluster_no"] == i]

df.groupby("cluster_no").agg(np.mean) # A look at descriptive statistics
df.groupby(["cluster_no", "Segment"]).agg({"Segment": ["count", "min", "mean", "max"]})
