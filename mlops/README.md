SWAGGER: http://localhost:12345/  

METHODS

Get list of available models   
GET  http://127.0.0.1:12345/getListOfModels (no params)

Learn default model and get prediction  
POST http://127.0.0.1:12345/learnDefaultModel
params: {"model":"Ridge", "0":0.1, "1":100} , where "0" is hyperparam 1, "1" is hyperparam 2 for model  
Ridge(alpha = hyperparam1, max_iter = hyperparam2)  
RandomForestRegressor(max_depth = hyperparam1, min_samples_leaf = hyperparam2)  

Use default model and get prediction on custom data  
POST http://127.0.0.1:12345/learnDefaultModel  
params: [{"model":"Ridge", "0":0.1,"1":100},  
                {"0":0.05479,"1":33.0,"2":2.18,"3":0.0,"4":0.472,  
                "5":6.616,"6":58.1,"7":3.3700, "8":7.0,"9":222.0,  
                "10":18.4,"11":393.36,"12":8.93}] , with name of model, hyperparams and features  
                

Delete model  
DELETE http://127.0.0.1:12345/deleteModel  
params: [{"model":"Ridge"}]  
