import sys
import json
import pickle
from sklearn.naive_bayes import GaussianNB
model=pickle.load(open('Models\\DiabetesDiagnosis.sav', 'rb'))

def predict(features):
    p=model.predict(features)
    if p[0]==0:
        result = {'result': 'You do not have diabetes'}
    else:
        result = {'result': 'You have diabetes'}
    # print(result)
    print(json.dumps(result))


# The first argument (sys.argv[0]) is the script name, so we skip it
feature_values_json = sys.argv[1:]
# print(feature_values_json[0])

features = [[ int(feature_values_json[0]), int(feature_values_json[1]), int(feature_values_json[2]), float(feature_values_json[3]) ]]

# print(features)
# Call the prediction function with the features
predict(features)
