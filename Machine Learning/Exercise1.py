from sklearn import tree
import sys
class Conv:

    def __init__(self,weight,numberOfSeats):
        self.weight=weight
        self.numberOfSeats=numberOfSeats
    def convertEntryToArray(self):
        return  [[self.weight, self.numberOfSeats]]

print("What is the weight of the Vehicle (in tonnes):")
weight = sys.stdin.readline()
print("How many Seats does Car have? :")
numberOfSeats = sys.stdin.readline()
#Horsepower and Seats of Vehicle
features =[[300, 2], [450, 2], [200, 8], [150, 9]]
#class of Vehicle sports-car=1,minivan=2
label = [1, 1, 0, 0]
try:

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, label)
    obj = Conv(weight,numberOfSeats)
    result = clf.predict(obj.convertEntryToArray())
    if result == [1]:
        print("This is a sport-car")
    elif result == [0]:
          print("This is a minivan")
except (RuntimeError,ValueError):
    print("Error %s found,make sure value enter is a number or maybe {}".format(ValueError) % RuntimeError)
