import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def runEngine(user_id):
    rating = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    user = pd.read_csv('data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    book = pd.read_csv('data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")

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
    rating_scaled = pd.DataFrame(scaler.fit_transform(combined['BookRating'].values.reshape(-1,1)))
    combined['BookRating'] = rating_scaled

    combined = combined.drop_duplicates(['UserID', 'BookTitle'])
    user_book_matrix = combined.pivot(index='UserID', columns='BookTitle', values='BookRating')
    user_book_matrix.fillna(0, inplace=True)
    users = user_book_matrix.index.tolist()
    books = user_book_matrix.columns.tolist()
    user_book_matrix = user_book_matrix.to_numpy()

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
        top_ten_ranked = top_ten_ranked.groupby('UserID').head(10)




    result = top_ten_ranked.loc[top_ten_ranked['UserID'] == user_id] #user_id #278582

    result.insert(0, "id",1, allow_duplicates = False)
    print(result)
    print(result['BookTitle'])

    from sqlalchemy import create_engine
    engine = create_engine('sqlite:////Users/roethelchristine/FinalProject/db.sqlite3', echo=False)
    result.to_sql('recommender_recommender_book', engine, if_exists='replace')


#runEngine(278582)

