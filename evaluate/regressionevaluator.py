from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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