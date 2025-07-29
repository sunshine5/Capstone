### Real Estate Data Analysis

### Executive Summary
I explored different regression models to find the best model to help buyers, sellers and realtors make more informed real estate price decisions.  The modeling and analysis performed helped with identifying the top features that drive housing prices. The models that I used are Linear regression, KneighborsRegressor, RandomForestRegressor, and the XGBRegressor.  XGBRegressor was the most effcient model with highest R2 score and lowest errors.

#### Problem Statement 
How can real estate market trend analysis be used to predict market demand, property values including buying/selling prices and rental prices?  
- Project goal was to build regression models while fine tuning the model performance and identifying the top features in the datasets that drives the housing price.
- Some initial challenges for this project include getting reliable data and hyper parameter tuning.
- Benefits of my machine learning model are that it gives buyers, sellers, and realtors an accurate look at the features that most significantly impact housing prices to help them make more informed financial decisions.

#### Model Outcomes or Predictions 
The models explored are regression models where expected output is different error metrics, such as mean squared error, root mean squared error.  Supervised learning algorithms were used.

#### Data Acquisition
- I am using a public dataset available on Kaggle called “Housing Price Dataset”. Here is the link: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset 

- The Housing Price Dataset has 21 features: id, date, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15

- The Data analysis and Visualizations:  The following graphs captures the nature of the data and their correlation with the housing price.  This is very helpful as the main objective of the project is to find a model that can predict accurate housing price based on the features of the Dataset.

<img width="474" height="369" alt="House_Price_Distribution" src="https://github.com/user-attachments/assets/a1280105-6b19-4b56-8e40-587567e33406" />

<img width="474" height="369" alt="yr_built_distribution" src="https://github.com/user-attachments/assets/fcc717da-1d4e-445d-8d94-86fe16e42811" />

<img width="479" height="369" alt="yr_renovated_dist" src="https://github.com/user-attachments/assets/d20d2d6f-266c-4d2f-9092-c355185d2665" />

<img width="458" height="352" alt="HouseLatDistribution" src="https://github.com/user-attachments/assets/60e65241-3b7f-4b6f-92ed-e0582797d115" />

<img width="452" height="351" alt="HouseLongDistribution" src="https://github.com/user-attachments/assets/cb3985de-4e16-4f98-bd35-cd5018378d42" />

<img width="1095" height="714" alt="PriceByZipcode" src="https://github.com/user-attachments/assets/cb488bdc-5890-44bc-a0e5-ec2394b747c1" />

<img width="1078" height="713" alt="PriceByBathrooms" src="https://github.com/user-attachments/assets/7cd7cd90-ce26-4804-b82a-30bcaf3b991c" />

<img width="1090" height="712" alt="PriceByBedrooms" src="https://github.com/user-attachments/assets/48c5602f-bad5-4865-9902-fbf2700c48a0" />

<img width="558" height="426" alt="PriceByCondition" src="https://github.com/user-attachments/assets/fc56853d-1623-4f1d-bb17-303c273b58a3" />

<img width="1091" height="713" alt="PriceByFloors" src="https://github.com/user-attachments/assets/603862b2-1b90-4c5e-864e-0ccd39851c6f" />

<img width="577" height="427" alt="PriceBySqftLiving" src="https://github.com/user-attachments/assets/ea376b98-2bb9-4625-beda-a13cfed441a8" />

<img width="1083" height="728" alt="PriceByYr_built" src="https://github.com/user-attachments/assets/b4603157-20fd-4ecb-9ac6-c60e35296bfa" />

<img width="1073" height="710" alt="PriceByView" src="https://github.com/user-attachments/assets/a47fb28b-7ac1-44f6-832a-d1ad8beb8a21" />

<img width="1081" height="710" alt="PriceByGrade" src="https://github.com/user-attachments/assets/d7cd3163-2f6d-4c13-bc15-8bde807ca5d4" />

#####   Heatmap for all features

<img width="957" height="557" alt="HeatMap_All_Features" src="https://github.com/user-attachments/assets/99ac2891-4145-4a35-8f8a-bd359244a0d8" />

#### Data Preprocessing/Preparation: 
- I leveraged different techniques from Pandas python libraries for Data cleaning.
- I checked if there are any null values, missing values and zero values in the dataset.
- I also evaluated the dataset by getting more information, understanding the shape and description, and evaluating unique values for each column.
- I dropped the id and date columns as they were not affecting the house price directly.

- I created X and y datasets, where y is the target variable feature which is Price in our case and X has features other than price.
- This dataset contains all numerical features which was scaled using Standard Scaler utility from sklearn preprocessing.
- The data further was split into training and test sets by keeping 0.2 as test size.

#### Modeling 
- I considered supervised regression models for my problem.  The models that I used include Linear regression, KneighborsRegressor, RandomForestRegressor, and the XGBRegressor.
- The evaluation metrics used are Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and R2 Score.

- I started with Linear regression as the first model.  I used Polynomial features and Sequential feature selection to further optimize Linear regression model.

- I used GridSearhCV technique to fine tune hyperparameters for KneighborsRegressor, RandomForestRegressor, and the XGBRegressor models.  

##### Model Evaluation:

- To decide the most optimal model I choose the model with lowest error metrics - Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and highest R2 Score.

#### Results
- Comparing linear regression along with linear regression run combined with polynomial features and feature selection I found that there is slight improvement in the model performance.

- There was significant improvement with every new model explored in the sequence from LinearRegression, KneighborsRegressor, RandomForestRegressor, and the XGBRegressor.
  
- For all the regression models the optimized run with GridSearchCV tuned hyperparameters yielded higher R2 score and lower error metrics.
  
- The top 7 selected features using from Linear regression using polynomial features and feature selection are: sqft_living, grade, yr_built, lat, sqft_living grade, view^2, yr_built sqft_living15

- The Feature importances by KneighborsRegressor, RandomForestRegressor, and the XGBRegressor based on gain and weight are captured in the graphs below.

<img width="1193" height="779" alt="Feature_Importances_For_RegressionModels" src="https://github.com/user-attachments/assets/298c70f5-89b8-424b-ad84-c5b00feec53e" />

- The evaluation metrics for regression model - Mean Squared Error, Root Mean Squared Error, Mean Absolute Error and R2 Score – are captured both in combined results format and in the following plot doing side-by-side comparison.

<img width="639" height="537" alt="ErrorMetricsSummary_all_models" src="https://github.com/user-attachments/assets/d9202721-5b97-42c1-a3fe-a2e922c65413" />

<img width="1271" height="670" alt="ErrorMetricsByRegressionModels" src="https://github.com/user-attachments/assets/5f299746-711f-4133-9696-c984706277f1" />

#### Next steps
- Exploring other housing data sources can help capture historical sales data, economic indicators, and seasonal patterns.  
- Using other techniques like time series forecasting (e.g., ARIMA) can help evaluate price trends over time.

#### Outline of project
- https://github.com/sunshine5/Capstone/blob/main/prompt.ipynb

#### Contact and Further Information
**Sima Shah** simashah010@gmail.com
