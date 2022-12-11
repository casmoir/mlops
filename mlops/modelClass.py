class modelClass:
    def __init__(self, input_model, hyperparam1, hyperparam2):

        '''initialize class
        params:
        input_model : str, name of model (Ridge or RandomForestRegressor)
        hyperparam1 = int, hyperparam 1 for model
        hyperparam2 = int, hyperparam 2 for model
        '''

        X_raw, self.y = load_boston(return_X_y=True)
        self.X = pd.DataFrame(X_raw)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=10)

        if input_model == 'Ridge':
            self.model_name = 'Ridge'
            self.model = Ridge(alpha = hyperparam1, max_iter = hyperparam2)
        else:
            self.model_name = 'RandomForestRegressor'
            self.model=RandomForestRegressor(max_depth = hyperparam1, min_samples_leaf = hyperparam2)

    def fit(self):
        '''
        fit model
        '''
        self.model.fit(self.X_train, self.y_train)

    def save_model(self):
        '''
        Serialize model and its columns and save 
        '''
        joblib.dump(self.model, self.model_name +'.pkl')
        print("Model is saved")
        rnd_columns = list(self.X.columns)
        joblib.dump(rnd_columns, self.model_name +'_columns.pkl')
        print("Model Colums are Saved")

    def load_model(self):
        '''
        Load the model
        '''
        model = joblib.load(self.model_name +'.pkl')
        print(model)
        print("Model is loaded")
        return model

    def predict(self, x_test):
        '''
        Make prediction
        params:
        x_test : data frame of features for prediction
        '''
        model = self.load_model()
        y_pred = model.predict(x_test)
        return y_pred