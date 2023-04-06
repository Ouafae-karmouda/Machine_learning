
from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from decision_trees import visualize_classifier

X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1)
tree = DecisionTreeClassifier()

bag = BaggingClassifier(tree, n_estimators= 300, max_samples= 0.8, 
                        random_state=1)
bag.fit(X,y)
visualize_classifier(bag, X, y)


model = RandomForestClassifier(n_estimators=100, random_state=1)
visualize_classifier(model, X, y)