import pandas

data = pandas.read_csv('Data/train.csv')

display(data['Survived'].to_numpy())