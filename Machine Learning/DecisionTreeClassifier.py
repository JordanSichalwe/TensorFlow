from __future__ import print_function

training_data =[["Yellow", 3, "Apple"],
                ["Green", 4, "Apple"],
                ['Red', 1, 'Grape'],
                ["Red", 2, "Grape"],
                ["Yellow", 3, "Lemon"]]


header = ["color", "size", "label"]

def univalues(rows, col):
    return set(row[col] for row in rows)
def countOfArray(rows):
    count = {}
    for row in rows:
        label = row[-1]
        if label not in count:
            count[label]=0
        count[label] +=1
    return count

def is_numeric(rows):
    return isinstance(rows, int) or isinstance(rows, float)

class Question:
    def __init__(self,column, value):
        self.column = column
        self.value = value
    def __repr__(self):
        condition ="=="
        if is_numeric(self.value):
            condition = ">="
        return ("Is %s %s %s") %(header[self.column], condition, str(self.value))
    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value
def partition(rows, question):
    true_rows,falue_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            falue_rows.append(row)
    return true_rows,falue_rows

def gini(rows):
    count = countOfArray(rows)
    impurity = 1
    for lbl in count:
        prob_lbl = count[lbl]/float(len(rows))
    impurity -= prob_lbl ** 2
    return impurity

def info_gain(left, right, current_uncertaintiy):
    p = float(len(left))/(len(left)+len(right))
    return current_uncertaintiy - p* gini(left)-(1-p)* gini(right)

def find_best_spit(rows):
    best_gain = 0
    best_question = None
    n_features = len(rows[0])-1
    current_uncertainity = gini(rows)
    for col in range(n_features):
        values = set(row[col] for row in rows)
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows)==0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainity)
            if gain >= best_gain:
                    best_gain = gain
                    best_question = question
    return best_gain,best_question
class Leaf:
    def __init__(self, rows):
        self.predictions = countOfArray(rows)
class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question =question
        self.true_branch = true_branch
        self.false_branch = false_branch
def build_tree(rows):
    gain, question = find_best_spit(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return DecisionNode(question, true_branch, false_branch)
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict",node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + " ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + " ")

def classify(rows, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(rows):
        return classify(rows, node.true_branch)
    else:
        return classify(rows, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values())*10
    #Memoization
    prob = {}
    for lbl in counts.keys():
        prob[lbl] = str(int(counts[lbl]/total *100))+"%"
    return prob



if __name__ =="__main__":

    #try:
    my_tree = build_tree(training_data)
    print_tree(my_tree)
    testing_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 4, 'Apple'],
        ['Red', 2, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
         ]
    for row in testing_data:
           print("Actual:%s. Predicted:%s" % (row[-1],classify(row, my_tree)))
    #except(TypeError,RuntimeError):
       # print("Error found")

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for regression

