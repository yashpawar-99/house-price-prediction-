import numpy as np
import pandas as pd 
import os 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score



MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(df_feature_num, df_feature_cat):

    num_pipeline = Pipeline([
        ("impute",SimpleImputer()),
        ("scaleing",StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoding",OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,df_feature_num),
        ("cat",cat_pipeline,df_feature_cat)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
 
    #1. load the dataset 

    df = pd.read_csv("../House-Price-Predication/data/housing.csv")

    #2. split a train set and test set

    df["income cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    train_set, test_set = train_test_split(df,
                                        test_size=0.2,
                                        stratify=df["income cat"],
                                        random_state=42
                    )

    for sett in (train_set,test_set):
        sett.drop("income cat",axis=1,inplace=True)


    #3. work with train set 

    df_feature = train_set.drop("median_house_value",axis=1)
    df_label = train_set["median_house_value"]

    #4. spliting a feture into numric and catageric data 

    df_feature_num = df_feature.select_dtypes(include=[np.number]).columns
    df_feature_cat = ["ocean_proximity"]

    #5. creating a pipeline for numric data 

    pipeline = build_pipeline(df_feature_num,df_feature_cat)
    housing_prepard = pipeline.fit_transform(df_feature)

    #print(housing_prepard)

    # train the model :-  we test multiple model and random forest give best
    # rsme score so we choose a that model all model score are avilable in notebooks 

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepard,df_label)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    # Work with test dataset 

    test_feature = test_set.drop("median_house_value",axis=1)
    test_label = test_set["median_house_value"]

    test_prepared = pipeline.transform(test_feature)
    test_prediction = model.predict(test_prepared)

    rsme = root_mean_squared_error(test_label,test_prediction)
    mae = mean_absolute_error(test_label,test_prediction)
    r2 = r2_score(test_label,test_prediction)

    
    print("Model Is Traind ! congrats !!\nThis are the Score : ")
    print("RMSE:", rsme)
    print("MAE:", mae)
    print("R²:", r2)


else:
    #Lets take input the model 

    #load the model and pipeline 
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    #user input 
    longitude = float(input("Enter longitude: "))
    latitude = float(input("Enter latitude: "))
    housing_median_age = float(input("Enter housing median age: "))
    total_rooms = float(input("Enter total rooms: "))
    total_bedrooms = float(input("Enter total bedrooms: "))
    population = float(input("Enter population: "))
    households = float(input("Enter households: "))
    median_income = float(input("Enter median income: "))
    ocean_proximity = input("Enter ocean proximity: ")


    #convrt into dataframe 

    input_data = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
    })

    input_tranceform = pipeline.transform(input_data)
    input_prediction = model.predict(input_tranceform)
    print("------------------------------------------------------")
    print("Predicted House Price:", input_prediction[0])
    print("------------------------------------------------------")