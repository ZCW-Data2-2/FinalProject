import io
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import boto3
import os
from dotenv import load_dotenv, find_dotenv
from io import StringIO
import dotenv
# from .views import fantasysel

# import cgi
# form = cgi.FieldStorage()
# fantasy = form.getvalue('fantasy')
# horror = form.getvalue('horror')
# romance = form.getvalue("romance")
# adventure = form.getvalue("adventure")
# nonfiction = form.getvalue("nonfiction")

# print(fantasy)
# print(horror)

horrorbooks= [
    (1, "Rose Madder", 5),
    (1, "Bag of Bones", 5),
    (1, "Intensity", 193),
    (1, "The Tale of the Body Thief (Vampire Chronicles...)", 193),
    (1, "Prey: A Novel", 191),
]

thrillerbooks= [
    (175510, "The Angel of Darkness", 200),
    (32206, "Cat &amp; Mouse (Alex Cross Novels)", 198),
    (193259, "The Jester", 194),
    (207582, "The Simple Truth", 197),
    (184212, "The Deep End of the Ocean", 196),
]


# rating = pd.read_csv('/Users/laffertythomas/dev/projects/FinalProject/BookRecommender/recommender/data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# user = pd.read_csv('/Users/laffertythomas/dev/projects/FinalProject/BookRecommender/recommender/data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# book = pd.read_csv('/Users/laffertythomas/dev/projects/FinalProject/BookRecommender/recommender/data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_file = os.path.join(BASE_DIR, ".env")
def runEngine(fantasy, horror, romance, adventure, nonfiction):
    if os.path.isfile(dotenv_file):
        dotenv.load_dotenv(dotenv_file)

    load_dotenv(find_dotenv())

    # Creating the low level functional client AWS S3
    client = boto3.client(
        's3',
        aws_access_key_id = os.environ['aws_access_key_id'],
        aws_secret_access_key = os.environ['aws_secret_access_key'],
        region_name = 'us-east-1'
    )
        
    # Creating the high level object oriented interface AWS S3
    resource = boto3.resource(
        's3',
        aws_access_key_id = os.environ['aws_access_key_id'],
        aws_secret_access_key = os.environ['aws_secret_access_key'],
        region_name = 'us-east-1'
    )

    # Load in the data from the CSV files in AWS S3 bucket
# def runEngine(exampleuserid):
    bucket_name = 'bookrecommender-22'

    rating_object_key = ("BX-Book-Ratings.csv")
    user_object_key = ("BX-Users.csv")
    book_object_key = ("BX-Books.csv")

    rating_csv_obj = client.get_object(Bucket=bucket_name, Key=rating_object_key)
    user_csv_obj = client.get_object(Bucket=bucket_name, Key=user_object_key)
    book_csv_obj = client.get_object(Bucket=bucket_name, Key=book_object_key)

    rating_body = rating_csv_obj['Body']
    user_body = user_csv_obj['Body']
    book_body = book_csv_obj['Body']

    # Alternative method
    # rating = pd.read_csv(io.BytesIO(rating_csv_obj['Body'].read()))
    # user = pd.read_csv(io.BytesIO(user_csv_obj['Body'].read()))
    # book = pd.read_csv(io.BytesIO(book_csv_obj['Body'].read()))

    # Removed below line
    # .splitlines(True)
    rating_csv_string = rating_body.read().decode('ISO-8859-1')
    user_csv_string = user_body.read().decode('ISO-8859-1')
    book_csv_string = book_body.read().decode('ISO-8859-1')

    rating = pd.read_csv(StringIO(rating_csv_string), sep=';', error_bad_lines=False)
    
    horrorbooks= [
    [1, "Rose Madder", horror],
    [1, "Bag of Bones", horror],
    [1, "Intensity", horror],
    [1, "The Tale of the Body Thief (Vampire Chronicles...)", horror],
    [1, "Prey: A Novel", horror],
]
    rating.append(horrorbooks)

    # Get into dataframe
    # rating.append(fantasysel)
    
    user = pd.read_csv(StringIO(user_csv_string), sep=';', error_bad_lines=False)
    book = pd.read_csv(StringIO(book_csv_string), sep=';', error_bad_lines=False)

    # Create the S3 object
    # rawrating = client.get_object(
    #     Bucket = 'bookrecommender-22',
    #     Key = 'BX-Book-Ratings.csv'
    # )
    # rawuser = client.get_object(
    #     Bucket = 'bookrecommender-22',
    #     Key = 'BX-Users.csv'
    # )
    # rawbook = client.get_object(
    #     Bucket = 'bookrecommender-22',
    #     Key = 'BX-Books.csv'
    # )

    # Read data from the S3 object
    # rating = pd.read_csv(rawrating['Body'], sep=';', error_bad_lines=False, encoding="latin-1")
    # user = pd.read_csv(rawuser['Body'], sep=';', error_bad_lines=False, encoding="latin-1")
    # book = pd.read_csv(rawbook['Body'], sep=';', error_bad_lines=False, encoding="latin-1")



    book_rating = pd.merge(rating, book, on='ISBN')
    cols = ['Year-Of-Publication', 'Publisher', 'Book-Author', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
    book_rating.drop(cols, axis=1, inplace=True)

    rating_count = (book_rating.
        groupby(by = ['BookTitle'])['BookRating'].
        count().
        reset_index().
        rename(columns = {'BookRating': 'RatingCount_book'})
        [['BookTitle', 'RatingCount_book']]
        )

    threshold = 25
    rating_count = rating_count.query('RatingCount_book >= @threshold')

    user_rating = pd.merge(rating_count, book_rating, left_on='BookTitle', right_on='BookTitle', how='left')
    user_count = (user_rating.
        groupby(by = ['UserID'])['BookRating'].
        count().
        reset_index().
        rename(columns = {'BookRating': 'RatingCount_user'})
        [['UserID', 'RatingCount_user']]
        )


    threshold = 20
    user_count = user_count.query('RatingCount_user >= @threshold')

    combined = user_rating.merge(user_count, left_on = 'UserID', right_on = 'UserID', how = 'inner')

    scaler = MinMaxScaler()
    combined['BookRating'] = combined['BookRating'].values.astype(float)


    # Unique Books 5850
    # Unique Users 3192


    # Tenserflow begins 
    rating_scaled = pd.DataFrame(scaler.fit_transform(combined['BookRating'].values.reshape(-1,1)))
    combined['BookRating'] = rating_scaled

    combined = combined.drop_duplicates(['UserID', 'BookTitle'])
    user_book_matrix = combined.pivot(index='UserID', columns='BookTitle', values='BookRating')
    user_book_matrix.fillna(0, inplace=True)
    users = user_book_matrix.index.tolist()
    books = user_book_matrix.columns.tolist()
    user_book_matrix = user_book_matrix.to_numpy()

    users.append()

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    num_input = combined['BookTitle'].nunique()
    num_hidden_1 = 10
    num_hidden_2 = 5

    X = tf.placeholder(tf.float64, [None, num_input])

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
        'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
        'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
        'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
        'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
        'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
        'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
    }


    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        return layer_2

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        return layer_2

    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)
    y_pred = decoder_op
    y_true = X


    loss = tf.losses.mean_squared_error(y_true, y_pred)
    optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
    eval_x = tf.placeholder(tf.int32, )
    eval_y = tf.placeholder(tf.int32, )
    pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

    init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    pred_data = pd.DataFrame()

    with tf.Session() as session:
        epochs = 10
        batch_size = 35

        session.run(init)
        session.run(local_init)

        num_batches = int(user_book_matrix.shape[0] / batch_size)
        user_book_matrix = np.array_split(user_book_matrix, num_batches)
        
        for i in range(epochs):

            avg_cost = 0
            for batch in user_book_matrix:
                _, l = session.run([optimizer, loss], feed_dict={X: batch})
                avg_cost += l

            avg_cost /= num_batches

            print("epoch: {} Loss: {}".format(i + 1, avg_cost))

        user_book_matrix = np.concatenate(user_book_matrix, axis=0)

        preds = session.run(decoder_op, feed_dict={X: user_book_matrix})

        pred_data = pred_data.append(pd.DataFrame(preds))

        pred_data = pred_data.stack().reset_index(name='BookRating')
        pred_data.columns = ['UserID', 'BookTitle', 'BookRating']
        pred_data['UserID'] = pred_data['UserID'].map(lambda value: users[value])
        pred_data['BookTitle'] = pred_data['BookTitle'].map(lambda value: books[value])
        
        keys = ['UserID', 'BookTitle']
        index_1 = pred_data.set_index(keys).index
        index_2 = combined.set_index(keys).index



        top_ten_ranked = pred_data[~index_1.isin(index_2)]
        top_ten_ranked = top_ten_ranked.sort_values(['UserID', 'BookRating'], ascending=[True, False])
        top_ten_ranked_head = top_ten_ranked.groupby('UserID').head(10)


    prediction = (top_ten_ranked_head.loc[top_ten_ranked['UserID'] == 1])

    #import the relevant sql library 
    from sqlalchemy import create_engine
    from BookRecommender.settings import DATABASES

    # link to your database
    engine = create_engine(DATABASES['default'], echo = False)

    # attach the data frame (df) to the database with a name of the 
    # table; the name can be whatever you like
    top_ten_ranked.to_sql('recommender', engine, if_exists='replace')
    
    # run a quick test 
    # print(engine.execute(“SELECT * FROM recommender”).fetchmany(10))
    
    # return prediction


# runEngine(278582)
