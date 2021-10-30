# Car Price Predictions

A Chinese car manufacturer wants to enter the US market. We are required to model the price of cars. This will help management understand how exactly the price varies across different features. They can then accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels.

Slides can be downloaded <a href="https://github.com/ialkamal/project2-machine-learning/blob/master/technical_presentation/Car_Sales_Predictions.pdf" target="_blank">here</a>. 

## Video Presentation

<p align="center">
          <a href="https://www.youtube.com/watch?v=aMH-_jQsHUg" target="_blank"><img alt="Car Price Predictions" src="https://github.com/ialkamal/project2-machine-learning/blob/master/images/Car_Price_Predictions.png"/></a>          
</p>

## Dataset
src: https://www.kaggle.com/hellbuoy/car-price-prediction
- Entries: 205
- Features: 26

|No| Feature | Description | Type |
|-|-|-|-|
|1|**car_ID**| Unique id for each observation| Integer |
|2|**Symboling**| Its assigned insurance risk rating, A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe. | Categorical |
|3|**CarName** | Manufacturer and Model Name | Categorical | 
|4|**fueltype**| Car fuel type i.e gas or diesel | Categorical |
|5|**aspiration**| Aspiration used in a car | Categorical |
|6|**doornumber**| Number of doors in a car | Categorical |
|7|**carbody**| body of car | Categorical |
|8|**drivewheel**| type of drive wheel | Categorical |
|9|**enginelocation**|location of car engine | Categorical |
|10|**wheelbase**|wheelbase of car | Numeric |
|11|**carlength**|length of car | Numeric |
|12|**carwidth**|width of car | Numeric |
|13|**carheight**|height of car | Numeric |
|14|**curbweight**|weight of car without occupants or baggage | Numeric |
|15|**enginetype**|type of car engine | Categorical |
|16|**cylindernumber**|cylinder placed in the car | Categorical |
|17|**enginesize**|size of car | Numeric |
|18|**fuelsystem**|fuel system of car| Categorical |
|19|**boreratio**|boreratio of car | Numeric |
|20|**stroke**|stroke or volume inside the engine | Numeric |
|21|**compressionratio**|compression ratio of car | Numeric |
|22|**horsepower**|horsepower | Numeric |
|23|**peakrpm**|car peak rpm  | Numeric |
|24|**citympg**|mileage on city | Numeric |
|25|**highwaypmpg**|mileage on highway | Numeric |
|26|**price**|price of car | Numeric |

## Observations

##### Q1. What car manufacturers are represented in the data?

There are 22 car manufacturers reapresented in the dataset. Japanese car manufacturers represent about half the dataset.

<img src="/images/Car Manufacturers.png"/>

##### Q2. What are the most important factors affecting price?

The following features are the most correlated to the price [+ being positvely correlated] and [- being negatively correlated]
- Engine Size (+)
- Curb Weight (+)
- Horse Power (+)
- Car Width (+)
- Four Cylinder Type (-)
- Highway MPG (-)
- City MPG (-)
- Car Length (+)
- Rear Wheel Drive (-)
- Forward Wheel Drive (+)

#### Heatmap (Numerical Features)
<img src="/images/corr.png"/>

##### Q3. How does Engine Size affect Price?

Engine size is positvely correlated with Price and is the most important factor in determining car price.

<img src="/images/Engine_Size.png"/>

##### Q4. What is the price range of different car manufacturers?

- Jaguars, Buicks, Porsches and BMWs are the most expensive.
- Hondas, Plymouths, Dodges and Chevrolets are the least expensive.

<img src="/images/price_range.png"/>

###### Code Snippet
```python
# Plotting Price boxplots grouped by Car Manufacturer
ordered_by_mean = df.groupby("Car_Manufacturer")["Price"].mean().sort_values(ascending=False).keys()
plt.figure(figsize=(30,10))
sns.boxplot(data=df,x="Car_Manufacturer",y="Price",order=ordered_by_mean)
plt.title("Price Range by " + "Car_Manufacturer".replace("_", " "),size=20,y=1.03)
plt.xlabel("Car_Manufacturer".replace("_", " "),size=15)
plt.ylabel("Price",size=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
```

## Prediction Models and their Performance

| Model| R<sup>2</sup> | RMSE |
|---|---|---|
|Baseline| -.19% | 8,233.30|
|Linear Regression| 81.78% | 3,510.66|
|Decision Tree| 88.05% | 2,843.92|
|Bagging Trees| 94.01% | 2,012.87|
|Random Forest| 94.01% | 2,013.49|
|Random Forest| 91.80% | 2,354.95|

**Bagging Trees** did the best with an R<sup>2</sup> score of 94.01% and a root mean sqaure error of 2,012.87.
