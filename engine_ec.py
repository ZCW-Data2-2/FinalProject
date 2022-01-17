import numpy as np
import pandas as pd
books = pd.read_csv("/Users/cantekinefe/dev/FinalProject/datasets/books/BX-Books.csv", sep=';', encoding="latin-1", error_bad_lines=False)
users = pd.read_csv("/Users/cantekinefe/dev/FinalProject/datasets/books/BX-Users.csv", sep=';', encoding="latin-1", error_bad_lines=False)
ratings = pd.read_csv("/Users/cantekinefe/dev/FinalProject/datasets/books/BX-Book-Ratings.csv", sep=';', encoding="latin-1", error_bad_lines=False)

books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']]
books.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)
users.rename(columns = {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)

ratings['user_id'].value_counts()

x = ratings['user_id'].value_counts() > 200
y = x[x].index  #user_ids
print(y.shape)
ratings = ratings[ratings['user_id'].isin(y)]

rating_with_books = ratings.merge(books, on='ISBN')
rating_with_books.head()

number_rating = rating_with_books.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
final_rating = rating_with_books.merge(number_rating, on='title')
final_rating.shape
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]
final_rating.drop_duplicates(['user_id','title'], inplace=True)

book_pivot = final_rating.pivot_table(columns='user_id', index='title', values="rating")
book_pivot.fillna(0, inplace=True)

from scipy.sparse import csr_matrix
book_sparse = csr_matrix(book_pivot)

from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

distances, suggestions = model.kneighbors(book_pivot.iloc[237, :].values.reshape(1, -1))

for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])