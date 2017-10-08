from sklearn import tree

# Examples
# Horsepower Seats Label
# 300       2      sports-car
# 450       2      sports-car
# 200       8      minivan
# 150       9      minivan

features = [[300,1],[450,1],[200,0],[150,0]]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[100,0]]))
