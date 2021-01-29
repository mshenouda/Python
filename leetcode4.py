import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def pandasFunc():

	a = np.arange(2, 12, 2)
	a_series = pd.Series(a)

	myList = a_series.tolist()
	b = np.arange(1, 11, 2)
	b_series = pd.Series(b)

	#print(a_series)
	#print(b_series)

	c_series = np.round(a_series * b_series, 2)
	c_largest = np.argmax(c_series)
	c_largest = c_series[c_largest]

	a = dict()
	a = {'a': 100, 'b': 200, 'c':300, 'd':400}
	b = pd.Series(a)

	a = [0, 10, 20, 30, 40]
	a = pd.Series(a)

	a = pd.Series([100, 200, 'python', 300.12, 400], dtype=object)
	b = pd.to_numeric(a, errors='coerce')

	a = np.arange(0, 5)
	b = np.arange(1, 11, 2)
	df = pd.DataFrame(index=['a', 'b', 'c', 'd', 'e'], columns=['one', 'two'])
	df['one'] = a
	df['two'] = b
	pd.DataFrame()

	c_series = pd.Series(df['one'])


	a = ['Red', 'Green', 'White']
	b = ['Red', 'black']
	c = ['Yellow']
	df = pd.DataFrame(index=['a', 'b', 'c'], data=[a, b, c])

	a = ['100', '200', 'python', '300.12', '400']
	s = pd.Series(a, dtype=object)
	s = sorted(s)

	s = np.arange(0, 10)
	s = pd.DataFrame(s)

	s = s[0:5]
	s = np.arange(0, 5)
	s = pd.Series(index=['A', 'B', 'C', 'D', 'E'], data=s)
	s = s.reindex(index=['B', 'C', 'E', 'A', 'D'])
	#print(s)
	#print(np.round(np.mean(s), 2))
	#print(np.round(np.std(s), 2))

	a = pd.Series([1, 2, 3, 4, 5])
	b = pd.Series([2, 4, 6, 8, 10])
	a = np.random.randint(1, 10, 20)
	a = pd.Series(a)
	b = a.value_counts(normalize=False, sort=True)
	b[1:] = 'Other'

	a = np.random.randint(1, 16, 20)
	a = pd.Series(a)
	b = a % 5 == 0
	b = [0, 4, 6, 8, 10, 15]
	print(a[b])
	c = a.isin(b)
	d = a.where(c)

	myList = ['php', 'python', 'java', 'c#']
	a = [1, 3, 5, 8, 10, 11, 15]
	print(a)

	a = ['01 Jan 2015', '2014/05/06', '20180307', '2019-04-06T11:20']
	b = pd.Series(a, dtype=object)
	c = pd.to_datetime(b)

	a = ['orange', 'Green', 'Red', 'Pink', 'Yellow']
	b = pd.Series(a)

	a = pd.Series([0, 1, 2, 3, 4, 5])
	b = pd.Series([1, 3, 2, 4, 5, 6])


def df():

	myDict = {'X': [75, 78, 76, 80, 81], 'Y': [81, 82, 83, 78, 74]}
	df = pd.DataFrame(myDict)

	exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
	             'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
	             'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
	             'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
	labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
	df = pd.DataFrame(data=exam_data, index=labels)
	selected = df[df['attempts'] >= 2]
	#print(len(df.axes[0]))
	#print(len(df.axes[1]))
	selected = df[(df['score'] >= 15) & (df['score'] <= 20)]
	selected = df
	selected = df[df['score'].isnull()]
	selected = df.loc['d', 'score'] = 11.5
	selected = df.loc['d']
	selected = np.sum(df['attempts'])
	selected = np.round(np.mean(df['score']), 2)
	df.loc['k'] = ["Suresh", 55.5, 2, 'yes']
	df = df.drop('k')

	df.sort_values(by=['score', 'name'], ascending=[False, True])
	selected = df.replace('James', value='Samir')
	df = df.drop(columns='attempts')

	colors = ['Red', 'Green', 'Red', None, None, 'Green', 'Red', 'Green', 'Red', 'Green']
	df.insert(loc=1, column='color', value=colors)
	selected = df.keys().tolist()

	newNames = ['newName', 'newColor', 'newScore', 'newQualify']
	selected = df[df['score'] == 9]


	df = {'city': ['California', 'Georgnia', 'Los Angeles'], 'People':[4, 2, 4]}
	df = pd.DataFrame(df)
	#print(df.groupby(by='city'))
	selected = df.iloc[[0]]
	df = df.replace('Georgnia', np.nan)
	df.fillna(value=0, inplace=True)
	df.reset_index(level=0, inplace=True)
	#print(df.to_string(index=False))
	df.reset_index(level=0, drop=False)

	a = pd.Series([1, 3, 5])
	b = pd.Series([2, 4, 6])
	myList = [[1, 2, 3], [4, 5, 6]]
	df = pd.DataFrame(myList)

	a = pd.Series(['C1', 'C1', 'C2', 'C2', 'C2', 'C3', 'C2'])
	b = pd.Series([1, 2, 3, 4, 5, 6, 7])
	df = pd.DataFrame({'col1':a, 'col2':b})
	df = 

df()