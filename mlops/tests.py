import unittest
from modelClass import *

class TestCase(unittest.TestCase):

    def test_get_data(self):
        x, y = get_data()
        #print('x', ((y.shape)))
        assert x.shape[1] > 1
        assert len(y) > 0
        print('test_get_data is done')

    def test_fit_predict_ridge(self):
        model = modelClass('Ridge', {'alpha': 0.1, 'max_iter': 100})
        model.fit()
        assert len(model.predict(model.X_test))!=0
        print('test_fit_predict_ridge is done')

    def test_fit_predict_RandomForestRegressor(self):
        model = modelClass('RandomForestRegressor', {'n_estimators': 100})
        model.fit()
        assert len(model.predict(model.X_test))!=0
        print('test_fit_predict_RandomForestRegressor is done')
   

if __name__ == '__main__':
    unittest.main()