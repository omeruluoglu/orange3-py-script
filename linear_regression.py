import Orange
from Orange.regression import linear
from Orange.data import Table
from Orange.evaluation import scoring, testing

#data operations
data = Table("auto-mpg.tab")
#set up linear regressions
##regularizations
Linear= linear.LinearRegressionLearner(preprocessors=None)
Linear.name="No regularization"

Ridge= linear.RidgeRegressionLearner(alpha=0.0001, fit_intercept=True,  normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', preprocessors=None)
Ridge.name="Ridge Regression(L2)"
                 
Lasso= linear.LassoRegressionLearner(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, preprocessors=None)
Lasso.name="Lasso Regression(L1)"

ElasticNet=linear.ElasticNetLearner(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, preprocessors=None)
ElasticNet.name="Elactic net Regression"   
           
learners= [Linear, Ridge, Lasso, ElasticNet]
results= testing.CrossValidation(data, learners, k=10, 
                                                 #random_state=1,
                                                 #store_data=False,
                                                 #store_models=False,
                                                 #preprocessor=None,
                                                 #callback=None,
                                                 #warnings=None
                                                 )
#data information                                                 
print("------Data Information------")
print("File: %a"%data.name,
      "Data Instances: %a"%len(data),
      "Attributes: %a"%len(data.domain.attributes),
      "Class:", data.domain.class_var.name)
#regression part
for i in range(len(learners)):
    print("Regularization: %a"%learners[i].name)
    print("------Testing&Score------")
    print("MSE: %.3f"%scoring.MSE(results)[i],
          "RMSE: %.3f"%scoring.RMSE(results)[i],
          "MAE: %.3f"%scoring.MAE(results)[i],
          "R2: %.3f"%scoring.R2(results)[i])
    print("------Predictions------")
    model = learners[i](data)    #linear model
    for j in range(len(data[0:10])):
        print("Data %a"%data[j], end=' ') #predictor variable "mpg"
        print("Linear Regression: %.4f"%model(data)[j]) #predicted values
