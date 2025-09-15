import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# Učitavanje podataka iz CSV fajlova
train_data = pd.read_csv('../titanic/train.csv')
test_data = pd.read_csv('../titanic/test.csv')

# Popunjavanje nedostajućih vrednosti
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

# Kombinovanje trening i test podataka
combined_data = pd.concat([train_data[['Sex', 'Embarked']], test_data[['Sex', 'Embarked']]])

# Kodiranje kategorijskih promenljivih
label_encoder = LabelEncoder()
combined_data['Sex'] = label_encoder.fit_transform(combined_data['Sex'])
combined_data['Embarked'] = label_encoder.fit_transform(combined_data['Embarked'])

# Vraćamo kodirane vrednosti nazad u odgovarajuće DataFrame-ove
train_data['Sex'] = combined_data['Sex'][:len(train_data)]
train_data['Embarked'] = combined_data['Embarked'][:len(train_data)]

test_data['Sex'] = combined_data['Sex'][len(train_data):]
test_data['Embarked'] = combined_data['Embarked'][len(train_data):]

# Kreiranje nove funkcije 'FamilySize' kao zbir 'SibSp' i 'Parch'
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Priprema podataka za obuku modela
train_x = train_data[['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked']].values
train_y = train_data['Survived'].values

test_x = test_data[['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'Embarked']].values
test_y = test_data['Survived'].values

nb_train = len(train_y)

# Normalizacija
train_mean = np.mean(train_x, axis=0)
train_std = np.std(train_x, axis=0)

train_x = (train_x - train_mean) / train_std
test_x = (test_x - train_mean) / train_std

# Parametri mreze
learning_rate = 0.001
nb_epochs = 16
batch_size = 128

# Parametri arhitekture
nb_input = train_x.shape[1]
nb_hidden1 = 256
nb_hidden2 = 256
nb_classes = 2

# Sama mreza
w = {
    '1': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_input, nb_hidden1], dtype=tf.float64)),
    '2': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden1, nb_hidden2], dtype=tf.float64)),
    'out': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden2, nb_classes], dtype=tf.float64))
}

b = {
    '1': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden1], dtype=tf.float64)),
    '2': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_hidden2], dtype=tf.float64)),
    'out': tf.Variable(tf.keras.initializers.GlorotUniform()([nb_classes], dtype=tf.float64))
}

f = {
    '1': tf.nn.relu,
    '2': tf.nn.relu,
    'out': tf.nn.softmax
}

def runNN(x):
    z1 = tf.add(tf.matmul(x, w['1']), b['1'])
    a1 = f['1'](z1)
    z2 = tf.add(tf.matmul(a1, w['2']), b['2'])
    a2 = f['2'](z2)
    z_out = tf.add(tf.matmul(a2, w['out']), b['out']) # a2 ovde!
    out = f['out'](z_out)

    pred = tf.argmax(out, 1)

    return pred, z_out

# Optimizator
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Trening!
for epoch in range(nb_epochs):
    epoch_loss = 0
    nb_batches = int(nb_train / batch_size)
    for i in range(nb_batches):
        x = train_x[i*batch_size : (i+1)*batch_size, :]
        y = train_y[i*batch_size : (i+1)*batch_size]
        y_onehot = tf.one_hot(y, nb_classes)

        with tf.GradientTape() as tape:
            _, z_out = runNN(x)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_out, labels=y_onehot))

        w1_g, w2_g, wout_g, b1_g, b2_g, bout_g = tape.gradient(loss, [w['1'], w['2'], w['out'], b['1'], b['2'], b['out']])

        opt.apply_gradients(zip([w1_g, w2_g, wout_g, b1_g, b2_g, bout_g], [w['1'], w['2'], w['out'], b['1'], b['2'], b['out']]))

        epoch_loss += loss

    # U svakoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_train

    print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

# Test!
x = test_x
y = test_y

pred, _ = runNN(x)
pred_correct = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(pred_correct, tf.float32))

print(f'Test acc: {accuracy:.3f}')