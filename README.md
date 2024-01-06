# Traffic Dataset

In our project, we have relied on a traffic prediction dataset to train several models, with the aim of reaching a high accuracy, and observing which of the models perform better. 
**Kaggle link:**
```
https://www.kaggle.com/datasets/hasibullahaman/traffic-prediction-dataset/data
```
## Data description
The traffic prediction dataset we have used contains 5952 records and 9 columns. The information collected is the traffic situation measured each 15 minutes over a period of 2 months. The observation is done based on the total number of cars, bikes, buses and trucks during each 15 minute interval. The total column represents the total number of vehicles for that time interval.  4 traffic situations are distinguished: low, normal, high and heavy. 

<img width="244" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/63e9ad1d-bd7a-401e-b31a-fecbaef89db3">

## Some visualization to better understand the nature of the dataset
| <img width="1000" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/2b9d2058-7e06-4280-9570-90455865d4f0"> | This visualization confirms that the traffic situation is dependent on the number of vehicles. If this number exceeds 180 for each 15 minutes then we deal with heavy traffic. If the number of vehicles does not exceed 120 then it is a favorable situation of either low or normal traffic. |
|---|---|

| <img width="2000" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/6d7cb4ec-cea2-4080-bd22-ddc7631877f2"> | This visualization represents the bus count on all days of the week. The colors represent traffic severity, where the purple color stands for heavy traffic, which as we expect is present when the number of buses is higher. We can observe that on Wednesdays this number is the lowest. |
|---|---|

| <img width="2000" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/6dd57607-1ee8-4adf-ab9d-c9ddee0e7bc5"> | This visualization represents the number of cars for different times, when the time is expressed in seconds. As in the representation above, the purple color corresponds to heavy traffic that occurs when a large number of cars is present. We can see that during the mid-day there are a lot of cars, where the minimum does not drop lower than 18.  |
|---|---|

## Data preprocessing
After understanding of the information contained within the dataset and visualizing it we used these steps as follows: 

1. Splitting the time column into hours and minutes/ splitting the time column into seconds. 
2. Creating a midday column containing the AM/PM which we encoded with 0 for AM and 1 for PM. 
3. Encoding the days of the week with numbers from 1-7 for each day respectively. 
4. Encoding the traffic situation with number 0-3 for each traffic situation respectively. 
5. Removing the target column(traffic situation) from the features.
6. Normalizing the features using: 
7. MinMaxScaler
8. StandardScaler
9. Applying dimensionality reduction using PCA.

## Methodologies
We decided to use both classification (supervised learning) and clustering (unsupervised learning). Classification was possible because we have the target column, which we remove and do not use for clustering.

### Individual Contributions:

#### Anita Mjeshtri
- **Classification Model:**
  - Artificial Neural Networks with Multilayer Perceptron and Backpropagation.
- **Clustering Model:**
  - Agglomerative Clustering.

#### Amara Çela
- **Classification Model:**
  - Support Vector Machines.
- **Clustering Model:**
  - DBSCAN.

#### Ilvio Çumani
- **Classification Model:**
  - Decision Trees.
- **Clustering Model:**
  - K-means.


## Disclaimer
It is important to note that the performance of unsupervised models on this dataset might be suboptimal. Since the dataset was curated with a focus on classification, unsupervised algorithms, which often thrive on inherent data structures, did not yield satisfactory results.

## Neural Network Model
### Model Architecture
The neural network of the model, dataset of which has not experienced dimensionality reduction, consists of two dense layers:
1. Input layers with 16 neurons, 9 inputs (equal to the number of features) and ReLu activation function.
2. Output layer with 4 neurons (equal to the number of Traffic Situation classes) and softmax activation function for multiclass classification.


The neural network of the model, dataset of which has experienced dimensionality reduction, consists of two dense layers:
1. Input layers with 16 neurons, 2 inputs (equal to the number of features after PCA is being applied) and ReLu activation function.
2. Output layer with 4 neurons (equal to the number of Traffic Situation classes) and softmax activation function for multiclass classification.

I have also experimented with the number of hidden neurons as well as activation and optimization functions, but these are shown in more detail below.

| <img width="2500" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/2e218977-f042-4823-822d-74a6b909971f"> | This visualization shows the input, hidden and output layer, demonstrating their relation to each other (16 nodes for hidden layer, no PCA applied). |
|---|---|

### Implementation details and Statistics


For each model I have experimented with a different number of hidden nodes, batch size and activation and optimization function. I also have gathered statistics about accuracy scores for different train-test splits while drawing conclusions that will be listed above.

**StandardScaler normalized data results:**


<img width="585" alt="Screenshot 2024-01-06 203145" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/9b925562-38be-471d-af4e-b2826ae42936">

**The case of Relu, Softmax, Adam with small batch size and large number of hidden nodes:**


<img width="484" alt="Screenshot 2024-01-06 203249" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/78b25a3b-048d-45bd-b497-f14497639aa0">

**MinMaxScaler and PCA normalized data result:**


<img width="583" alt="Screenshot 2024-01-06 203331" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/3c13e7fb-b8ce-454a-9ece-4b9692ac84f7">

### Conclusions regarding the number of hidden nodes:

After I did several experiments with the model, while increasing and decreasing the number of hidden nodes I came to the conclusion that these operations have distinct effects on the model:
- **Fewer hidden nodes result in a simpler model that requires less computational time during training and prediction.**
- **More hidden nodes provide the model with a higher capacity to learn complex patterns in the training data, with the risk of overfitting present, where the model memorizes the training data but fails to generalize unseen data.**
- **A large number of hidden nodes increase computational cost during training and inference.**

### Conclusions regarding the number of input nodes and accuracy results:

As seen above I have tried building models with a dataset in which PCA (dimensionality reduction is applied) and a dataset where PCA is not applied. Results have shown that the dataset we have chosen, Traffic Dataset, works better in the second case and to be realistic our dataset doesn’t really need PCA, since it has a small number of features. However, if we would talk about computational performance results, the cases where we are using the reduced dataset clearly require less computational resources.

### Conclusions regarding the batch size:

Batch size defines the number of samples used in each iteration of training the neural network. From the results of the table we can clearly see that a greater value of batch size leads to better computational performance, as more samples are processed in parallel. However, we should keep in mind that smaller batch sizes lead to better generalization since the model updates its weights more frequently based on fewer examples, even though it requires more iterations for convergence.

##  Agglomerative Clustering

Agglomerative Clustering is a form of unsupervised learning used to group objects in clusters based on their similarity. The algorithm starts by treating each object as a singleton cluster. Next, closest pairs of clusters are successively merged until all clusters have been merged into one big cluster containing all objects. The result is a tree-based representation of the objects, named dendrogram. 

Data has been normalized in the same way I have already mentioned above.

I have used complete and ward linkage. Complete linkage is max-agglomerative clustering, where we merge two closest clusters and choose the max of these clusters for the next step. The silhouette score has shown better results with ward linkage rather than complete linkage. Number of clusters is set to be 4 in both cases, since we have 4 classes of Traffic Situation.

The very low results are justified by the fact that our problem is a classification one, so it’s normal to have low accuracy scores.
| <img width="2500" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/63825563-e4ec-4932-a234-3b753da4683a"> | <img width="2500" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/da20aefa-3717-4292-8a99-67a95ee1ad3e"> |
|---|---|

### Statistics
<img width="507" alt="image" src="https://github.com/IlviCumani/TrafficPrediction/assets/122668935/a931b9c4-7a4b-430e-af4b-f52f2bc505a6">



