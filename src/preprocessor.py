import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

class TrendProcessor:
    """
    Polynomial detrending pre-processor.
    Fits a surface of a given order (1=Linear, 2=Quadratic, 3=Cubic) to spatial data.
    """
    def __init__(self, order=1):
        self.order = order
        self.model = make_pipeline(PolynomialFeatures(order, include_bias=False), LinearRegression())
        
    def fit(self, X, Y, Z):
        """Fit polynomial surface to coordinates (X, Y) and values Z."""
        coords = np.column_stack((X, Y))
        self.model.fit(coords, Z)
        return self
        
    def get_trend(self, X, Y):
        """Calculate trend values at coordinates (X, Y)."""
        coords = np.column_stack((X, Y))
        return self.model.predict(coords)
        
    def detrend(self, X, Y, Z):
        """Remove trend from Z values."""
        trend = self.get_trend(X, Y)
        return Z - trend
        
    def retrend(self, X, Y, Z_res):
        """Add trend back to residuals."""
        trend = self.get_trend(X, Y)
        return Z_res + trend
        
    def get_params(self):
        """Return parameters of the fitted linear regression model."""
        lr = self.model.named_steps['linearregression']
        return {
            'order': self.order,
            'intercept': lr.intercept_,
            'coefficients': lr.coef_.tolist()
        }
