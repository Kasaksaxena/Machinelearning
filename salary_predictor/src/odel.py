from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def train_linear_model(X,y):
    model=LinearRegression()
    model.fit(X,y)
    return model

def train_polynomial_model(X,y,degree=2):
    poly_model=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    poly_model.fit(X,y)
    return poly_model