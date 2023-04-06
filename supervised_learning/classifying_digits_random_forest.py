import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.metrics import confusion_matrix


from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load the dataset
digits = load_digits()
print(digits.keys())

# Get the features and the target
X, y = digits.data, digits.target

# set up the figure
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom= 0, top=1, hspace=0.05, wspace=0.05)

#plot the digits:each image is 8x8 pixels
for i in range(64):
    ax= fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    #label the image
    ax.text(0, 7, str(digits.target[i]))
plt.show()
# split the data on train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("Xtrain.shape", X_train.shape)
print("Xtest.shape", X_test.shape)

# instantiate the model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)


# get the predictions
y_pred = model.predict(X_test)

# get the ccuracy
score = accuracy_score(y_pred=y_pred, y_true=y_test)
print("score", score)

# Classification report
print(metrics.classification_report(y_pred, y_test))

#plot the confuison matrix
mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
sns.heatmap(mat.T, square = True, annot = True, fmt='d', cbar = False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()