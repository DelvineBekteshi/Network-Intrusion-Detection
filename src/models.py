from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def train_logistic_regression(X_train, y_train, C=1.0, penalty='l2', solver='lbfgs'):
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=C,
        penalty=penalty,
        solver=solver
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(
    X_train,
    y_train,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1
):
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5, weights='uniform', p=2):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def build_neural_network(input_dim, layers_config, learning_rate=0.001, dropout_rate=0.3):
    model = Sequential()

    model.add(Dense(layers_config[0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))

    for units in layers_config[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_neural_network(
    X_train,
    y_train,
    layers_config=[64, 32],
    learning_rate=0.001,
    dropout_rate=0.3,
    epochs=50,
    batch_size=64,
    validation_split=0.2
):
    model = build_neural_network(
        input_dim=X_train.shape[1],
        layers_config=layers_config,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )

    return model, history