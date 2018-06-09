from sklearn import tree

# features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [10, "bumpy"]]
features = [[140, 1], [130, 1], [150, 0], [10, 0]]
# labels = ["apple", "apple", "orange", "orange"]
labels = [0, 0, 1, 1]
#logic using a decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
result = clf.predict([[150, 0]])
print(result)
