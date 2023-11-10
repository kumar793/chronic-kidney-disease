import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
class model_creation:
    def __init__(self):
        pass

    def readData(self,path = "cleaned_data.csv"):   #pass the path
        self.kidney = pd.read_csv(path)
        return self.kidney

    def optimisedFeatureSelection(self):
        ind_col=[col for col in self.kidney.columns if col!='class']
        dep_col='class'
        self.X=self.kidney[ind_col]
        self.X.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)

        #Then, drop the column as usual.

        self.X.drop(["a"], axis=1, inplace=True)
        self.y=self.kidney[dep_col]
        imp_features=SelectKBest(score_func=chi2,k=20)
        imp_features=imp_features.fit(self.X,self.y)

        datascore=pd.DataFrame(imp_features.scores_,columns=['Score'])
        dfcols=pd.DataFrame(self.X.columns)
        self.features_rank=pd.concat([dfcols,datascore],axis=1)
        self.features_rank.columns=['features','Score']
        self.selected=self.features_rank.nlargest(11,'Score')['features'].values

        self.X=self.kidney[self.selected[1:]]
        return self.selected

    def featureSelection(self):
        self.X = self.kidney.drop(['class', 'specific gravity', 'appetite', 'red blood cell count', 'packed cell volume', 'haemoglobin', 'sodium'], axis = 1)
        self.X.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)

        # Then, drop the column as usual.

        self.X.drop(["a"], axis=1, inplace=True)
        self.y = self.kidney['class']

    def splitData(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.3, random_state = 85)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def modelCreation(self):

        self.model = RandomForestClassifier(n_estimators = 30)
        self.model.fit(self.X_train, self.y_train)

        with open("kidney_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def modelXGCreation(self):
        from xgboost import XGBClassifier
        from sklearn.model_selection import RandomizedSearchCV


        params={'learning-rate':[0,0.5,0.20,0.25],
                'max_depth':[5,8,10],
                'min_child_weight':[1,3,5,7],
                 'gamma':[0.0,0.1,0.2,0.4],
                  'colsample_bytree':[0.3,0.4,0.7]}

        self.classifier=XGBClassifier()
        random_search=RandomizedSearchCV(self.classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
        random_search.fit(self.X_train,self.y_train)
        est = random_search.best_estimator_
        param = random_search.best_params_

        self.classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='', learning_rate=0.300000012, max_delta_step=0,
              max_depth=5, min_child_weight=1,
              monotone_constraints='()', n_estimators=100, n_jobs=8,
              num_parallel_tree=1, random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)
        self.classifier.fit(self.X_train,self.y_train)

        with open("kidneyX_model.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

    def modelEvaluation(self):
        from sklearn.metrics import confusion_matrix, accuracy_score
        print(confusion_matrix(self.y_test, self.model.predict(self.X_test)))
        print(f"Accuracy is {round(accuracy_score(self.y_test, self.model.predict(self.X_test))*100, 2)}%")

    def ModelEvaluation(self):
      from sklearn.metrics import confusion_matrix, accuracy_score
      print(confusion_matrix(self.y_test, self.classifier.predict(self.X_test)))
      print(f"Accuracy is {round(accuracy_score(self.y_test, self.classifier.predict(self.X_test))*100, 2)}%")

if __name__ == "__main__":
    m = model_creation()
    d = m.readData()
    print(m.optimisedFeatureSelection())
    print(m.X.columns)
    m.splitData()
    m.modelXGCreation()
    m.ModelEvaluation()