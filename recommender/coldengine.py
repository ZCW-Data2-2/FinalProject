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


# rating = pd.read_csv('/Users/laffertythomas/dev/projects/FinalProject/BookRecommender/recommender/data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# user = pd.read_csv('/Users/laffertythomas/dev/projects/FinalProject/BookRecommender/recommender/data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
# book = pd.read_csv('/Users/laffertythomas/dev/projects/FinalProject/BookRecommender/recommender/data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dotenv_file = os.path.join(BASE_DIR, ".env")
def coldrunEngine(humor, horror, romance, thriller, nonfiction, hp):
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

    rating0 = pd.read_csv(StringIO(rating_csv_string), sep=';', error_bad_lines=False)
    
    # Append user input
    coldbooks= [
        # The Amazing Adventures of Kavalier & Clay
        [1, "0312282990", humor],
        # The Bonfire of the Vanities
        [1, "0553275976", humor],
        # Naked
        [1, "0316777730", humor],
        # Shopaholic Takes Manhattan (Summer Display Opp... 
        [1, "0385335881", humor],
        # Good Omens
        [1, "0441003257", humor],
        # Rose Madder
        [1, "0451186362", horror],
        # Bag of Bones
        [1, "067102423X", horror],
        # Intensity
        [1, "0345384369", horror],
        # The Tale of the Body Thief (Vampire Chronicles...)
        [1, "034538475X", horror],
        # Prey: A Novel
        [1, "0066214122", horror],
        # The Crimson Petal and the White
        [1, "015100692X", romance],
        # Cold Mountain 
        [1, "1400077826", romance],
        # Sea Glass: A Novel
        [1, "0316089699", romance],
        # The Switch
        [1, "0446609943", romance],
        # The Time Traveler's Wife
        [1, "193156146X", romance],
        # The Angel of Darkness
        [1, "0345427637", thriller],
        # Riptide
        [1, "0515130966", thriller],
        # The Jester
        [1, "0316602051", thriller],
        # The Simple Truth
        [1, "0446607711", thriller],
        # The Deep End of the Ocean
        [1, "0670865796", thriller],
        # I Know Why the Caged Bird Sings
        [1, "0553279378", nonfiction],
        # The Diary of a Young Girl: Anne Frank
        [1, "0553296981", nonfiction],
        # Into the Wild
        [1, "0385486804", nonfiction],
        # Rich Dad Poor Dad
        [1, "0446677450", nonfiction],
        # Seabiscuit
        [1, "0449005615", nonfiction],
        # Harry Potter Sorcerers Stone
        [1, "059035342X", hp],
        # Harry Potter CoS
        [1, "0439064872", hp],
        # Harry Potter PoA
        [1, "0439136350", hp],
        # Harry Potter Goblet of Fire
        [1, "0439139597", hp],
        # Harry Potter Order of the Pheonix
        [1, "043935806X", hp],
    ]

    rating = rating0.append(pd.DataFrame(coldbooks, columns=['UserID', 'ISBN', 'BookRating']))

    # print(rating['UserID'].where(rating['UserID'] == 1))
    # print(books_data['BookTitle'].where(books_data[''] == ''))

    user = pd.read_csv(StringIO(user_csv_string), sep=';', error_bad_lines=False)
    book = pd.read_csv(StringIO(book_csv_string), sep=';', error_bad_lines=False)


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

    threshold = 22
    rating_count = rating_count.query('RatingCount_book >= @threshold')

    user_rating = pd.merge(rating_count, book_rating, left_on='BookTitle', right_on='BookTitle', how='left')
    user_count = (user_rating.
        groupby(by = ['UserID'])['BookRating'].
        count().
        reset_index().
        rename(columns = {'BookRating': 'RatingCount_user'})
        [['UserID', 'RatingCount_user']]
        )


    threshold = 13
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

    # print(users['UserID'].where(users['UserID'] == 1))

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
        top_ten_ranked = top_ten_ranked.groupby('UserID').head(20)

    result = top_ten_ranked.loc[top_ten_ranked['UserID'] == 1] #user_id #278582

    result.insert(0, "id",1, allow_duplicates = False)
    print(result)
    print(result['BookTitle'])

    from sqlalchemy import create_engine
    engine = create_engine(os.environ['DATABASE_URL'], echo=False)
    result.to_sql('recommender_recommender_book', engine, if_exists='replace')

