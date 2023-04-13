import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix


data = fetch_20newsgroups()
print(data.target_names)

categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)


print(f"shape of data, {len(train.data)}")
print("Some samples of data")
# print(train.data[5])


#Create the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

#apply the model and make predictions
model.fit(train.data, train.target)
labels = model.predict(test.data)

#Evaluate the model

plt.figure()
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels = train.target_names)
plt.xlabel('True labels')
plt.ylabel('Predicted labels')

plt.show()


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    print(pred)
    return train.target_names[pred[0]]


print(predict_category('sending a payload to the ISS'))
print(predict_category('discussing islam vs atheism'))