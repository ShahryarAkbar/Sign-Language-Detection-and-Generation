import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Assuming each hand landmark contributes 2 features (x and y) and you have 21 landmarks
num_landmarks = 21
num_features_per_landmark = 2
expected_num_features = num_landmarks * num_features_per_landmark

# Ensure the data has the expected number of features
if data.shape[1] != expected_num_features:
    raise ValueError(f"Expected {expected_num_features} features, but the data has {data.shape[1]} features.")

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
