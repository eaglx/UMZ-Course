import sys
import pandas
from sklearn.tree import DecisionTreeClassifier

if len(sys.argv) < 4:
    print("Not enough args!")
    print("python3 decision_tree_solution.py train.tsv in.tsv out.tsv")
    exit()
# ***************** SETUP *****************
train_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
columns_in_file = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
train_columns =['Survived','PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
columns_to_get_solve = ['PassengerId','Pclass','Sex','Age','SibSp','Fare','Embarked']
# ***************** READ DATA *****************
train_tsv = pandas.read_csv(train_file, sep='\t', header=0, usecols=train_columns)
in_tsv = pandas.read_csv(in_file, sep='\t', header=None, names=columns_in_file)
# ***************** SOLVE *****************
x = train_tsv[columns_to_get_solve]
y = train_tsv['Survived']
x = x.fillna(0)
x['Sex'] = x['Sex'].replace('male',1)
x['Sex'] = x['Sex'].replace('female',0)
x['Embarked'] = x['Embarked'].replace('S',1)
x['Embarked'] = x['Embarked'].replace('Q',2)
x['Embarked'] = x['Embarked'].replace('C',3)

x_in = in_tsv[columns_to_get_solve]
x_in = x_in.fillna(0)
x_in['Sex'] = x_in['Sex'].replace('male',1)
x_in['Sex'] = x_in['Sex'].replace('female',0)
x_in['Embarked'] = x_in['Embarked'].replace('S',1)
x_in['Embarked'] = x_in['Embarked'].replace('Q',2)
x_in['Embarked'] = x_in['Embarked'].replace('C',3)

dt = DecisionTreeClassifier()
dt.fit(x, y)
test_out = dt.predict(x_in)
# ***************** SAVE *****************
outFile = open(out_file, "w")
for l in test_out:
	outFile.write('%01d\n' % l)
outFile.close()
