import Orange
from Orange.classification import naive_bayes, tree, knn, svm
from Orange.evaluation import testing, scoring
from Orange.data import Table
from Orange.widgets.evaluate import owconfusionmatrix

#set up learners
NaiveBayes = naive_bayes.NaiveBayesLearner(preprocessors=None)
ClassificationTree = tree.TreeLearner(criterion= "entropy",
                                      #max_leaf_nodes= None,
                                      min_samples_leaf= 2,
                                      #min_samples_split= 5,
                                      #max_features= None,
                                      #random_state= None,
                                      #splitter= 'best',
                                      max_depth= 100)
KNN= knn.KNNLearner(n_neighbors=5,
                    metric="euclidean", 
                    weights="uniform",
                    algorithm='auto',
                    preprocessors=None)
SVM= svm.SVMLearner(C=1.0,
                    kernel='linear',
                    degree=3,
                    gamma=0.0,
                    coef0=0.0,
                    shrinking=True,
                    probability=False,
                    tol=0.001,
                    cache_size=200,
                    max_iter=-1,
                    preprocessors=None)
learners= [NaiveBayes,ClassificationTree,KNN,SVM]
#data operations
data = Table("iris.tab")
#trainingData = data[1:20]
results= testing.CrossValidation(data, learners, k=3, 
                                                 #random_state=1,
                                                 #store_data=False,
                                                 #store_models=False,
                                                 #preprocessor=None,
                                                 #callback=None,
                                                 #warnings=None
                                                 )
#data information
print("-----Data Information-----")
print("File: %a"%data.name,
      "Instances: %a"%len(data),
      "Attributes: %a"%len(data.domain.attributes))

print("-----Evaluating-----")
for i in range(len(learners)):
    print("Learner Name: {}".format(learners[i].name))
    print(
          "AUC: %.4f"%scoring.AUC(results)[i],
          "Accuracy: %.4f"%scoring.CA(results)[i],
		  #"Log loss: %.4f"%scoring.LogLoss(results)[i],
          "F1: %.4f"%scoring.F1(results)[i],
		  #"MSE: %.4f"%scoring.MSE(results)[i],
		  #"RMSE: %.4f"%scoring.RMSE(results)[i],
          "Precision: %.4f"%scoring.Precision(results)[i],
          "Recall: %.4f"%scoring.Recall(results)[i],
          #"MAE: %.4f"%scoring.MAE(results)[i],
          #"R2: %.4f"%scoring.R2(results)[i]
		  )
    #confusion matrix
    print("Confusion Matrix for %a"%learners[i].name)
    conf_matrix = owconfusionmatrix.confusion_matrix(results,i)
    print(conf_matrix, end="\n"*2)
