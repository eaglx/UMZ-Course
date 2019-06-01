import sys
import pandas
from sklearn.neural_network import MLPClassifier

if len(sys.argv) < 4:
    print("Not enough args!")
    print("python3 neural_network_solution.py train.tsv in.tsv out.tsv")
    exit()
# ***************** SETUP *****************
train_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]
columns_in_file = ['Powierzchnia w m2','Liczba pokoi','Miejsce parkingowe','Liczba pięter w budynku','Piętro','Typ zabudowy','Okna','Materiał budynku','Rok budowy','Forma własności','Forma kuchni','Stan','Stan instalacji','Głośność','Droga dojazdowa','Stan łazienki','Powierzchnia działki w m2','opis']
train_columns =['cena','Powierzchnia w m2','Liczba pokoi','Miejsce parkingowe','Liczba pięter w budynku','Piętro','Typ zabudowy','Okna','Materiał budynku','Rok budowy','Forma własności','Forma kuchni','Stan','Stan instalacji','Głośność','Droga dojazdowa','Stan łazienki','Powierzchnia działki w m2','opis']
columns_to_get_solve = ['Liczba pokoi', 'Liczba pięter w budynku', 'Rok budowy', 'Powierzchnia w m2']
# ***************** READ DATA *****************
train_tsv = pandas.read_csv(train_file, sep='\t', header=0, usecols=train_columns)
in_tsv = pandas.read_csv(in_file, sep='\t', header=None, names=columns_in_file)
# ***************** SOLVE *****************
x = train_tsv[columns_to_get_solve]
y = train_tsv['cena']
x = x.fillna(0)

x_in = in_tsv[columns_to_get_solve]
x_in = x_in.fillna(0)

neural = MLPClassifier(solver='lbfgs', max_iter=100000000, random_state=1000, early_stopping=True, learning_rate='adaptive', alpha=0.000001)
neural.fit(x, y)
test_out = neural.predict(x_in)
# ***************** SAVE *****************
outFile = open(out_file, "w")
for l in test_out:
	outFile.write('%01d\n' % l)
outFile.close()
