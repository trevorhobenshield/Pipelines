import os
import warnings
from shutil import rmtree
from tempfile import mkdtemp
from typing import Dict, Iterable, Any

import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.models import Sequential
## sklearn KerasRegressor may be Deprecated
from scikeras.wrappers import KerasRegressor
from sklearn import set_config
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
## [sklearn.experimental] Do Not optimize imports ##
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# set_config(display="text")
set_config(display="diagram")
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU


def view(df_):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        display(HTML(df_.to_html()))


df = pd.read_csv('data/bmw.csv')
df = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType', 'year', 'engineSize'])
X = df.drop('price', 1)
y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=37)


def nn(layer_neurons: Iterable[int], meta: Dict[str, Any], compile_kwargs: Dict[str, Any]):
    model = Sequential([
        Input(shape=(meta["n_features_in_"])),
        *[Dense(neurons, kernel_initializer="uniform", activation="relu") for neurons in layer_neurons],
        Dense(1, activation='relu')
    ])
    model.compile(loss='mse', optimizer=compile_kwargs["optimizer"])
    return model


preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=[np.number])),
    (SimpleImputer(strategy="mean"), make_column_selector(dtype_include=[np.number])),
    (OneHotEncoder(sparse=False, handle_unknown='ignore'), make_column_selector(dtype_exclude=[np.number]))
)
cachedir = mkdtemp()
model = StackingRegressor(
    estimators=[
        ("elastic", make_pipeline(preprocessor, ElasticNet(), memory=cachedir)),
        ("lasso", make_pipeline(preprocessor, LassoCV(), memory=cachedir)),
        ("huber", make_pipeline(preprocessor, HuberRegressor(), memory=cachedir)),
        ("ensemble", Pipeline([
            ("preprocessor", preprocessor),
            ('votereg', VotingRegressor([
                ('lasso', LassoCV()),
                ('elast', ElasticNet())
            ]))
        ], memory=cachedir))],
    final_estimator=KerasRegressor(
        model=nn,
        layer_neurons=(16, 32, 64),  # todo: ajust
        batch_size=32,  # todo: ajust
        epochs=100,  # todo: ajust
        optimizer="adam",
        optimizer__learning_rate=0.001,  # todo: ajust
        verbose=0,
        random_state=0,
        callbacks=[
            EarlyStopping(
                monitor='loss',
                mode='min',
                verbose=0,
                patience=200,  # todo: ajust
                min_delta=1e-5  # todo: ajust
            )
        ]
    ),
    verbose=1
)
grid = HalvingGridSearchCV(estimator=model,
                           # todo: ajust
                           param_grid={
                               'ensemble__votereg__lasso__eps': [1 / (10 ** n) for n in range(4, -1, -1)],
                               # 'ensemble__votereg__lasso__tol': [1 / (10 ** n) for n in range(4, -1, -1)],
                               # 'ensemble__votereg__elast__alpha': [1 / (10 ** n) for n in range(4, -1, -1)],
                               # 'ensemble__votereg__elast__l1_ratio': np.arange(0, 0.5, 0.1)
                           },
                           n_jobs=-1,
                           cv=2,  # todo: ajust
                           min_resources="exhaust",
                           factor=3,  # todo: ajust
                           verbose=0)

predictions = grid.fit(X_train, y_train).predict(X_test)
RMSE = mean_squared_error(y_test, predictions, squared=False)
print(f'{grid.best_score_ = }\n{grid.best_params_ = }\n{RMSE = }')
rmtree(cachedir)
