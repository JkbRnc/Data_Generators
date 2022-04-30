import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighbor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate_randomforest(original, generated, target_column, classification=True):
  train_data = generated.drop([target_column], axis=1).to_numpy()
  train_values = generated[target_column].to_numpy()
  test_data = original.drop([target_column],axis=1).to_numpy()
  test_values = original[target_column].to_numpy()

  x_train, x_test, y_train, y_test = train_test_split(test_data, test_values)

  if classification:
    model = RandomForestClassifier()
    test_model = RandomForestClassifier()
  else:
    model = RandomForestRegressor()
    test_model = RandomForestRegressor()
  
  model.fit(train_data, train_values)
  test_model.fit(x_train, y_train)

  if classification:
    model_score = model.score(test_data, test_values)
    test_score = test_model.score(x_test, y_test)
  else:
    model_score = mean_squared_error(
        test_values, model.predict(test_data)
    )
    test_score = mean_squared_error(
        y_test, test_model.predict(x_test)
    )
  return model_score, test_score


def create_statistics(data, target_column, model, categorical_columns = [], k = 100, classification=False):
  model.fit(data, categorical_columns)
  model_score = []
  test_score = []
  for _ in range(k):
    samples = model.sample(10000)
    samples_df = pd.DataFrame(samples, columns=model.columns)
    
    x, y = evaluate_randomforest(data, samples_df, target_column=target_column, classification=classification)
    model_score.append(x)
    test_score.append(y)
  return np.array(model_score), np.array(test_score)


def distance (og, new):
  m = og.to_numpy() - new.to_numpy()
  return np.linalg.norm(m) / m.shape[0]**2