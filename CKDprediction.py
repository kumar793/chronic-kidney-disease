import pickle
import numpy as np
class predict:

    #the values list must contain 18 values
    def __init__(self,*values):
        self.values = list(values)

    def predicted(self):
        model = pickle.load(open('kidneyX_model.pkl','rb'))
        val = np.asarray(self.values)
        self.predicted = model.predict(val.reshape(1, -1))[0]
        return True

    def printPrediction(self):
      if predict.predicted(self):
        if self.predicted==1:
          str = "Chance of CKD is high"
        else:
          str = "very low chance negligible"
      return str

if '__name__'=='__main__':
    p = predict([])
    p.printPrediction()