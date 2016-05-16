class TimeSeriesValidation:    
    def __init__(self, n, step):
        """Cross validation for time-series data."""
        self.leap = 50000; self.k = 800000; self.h = 314099
        
        self.n=n
        self.step=step
    
    def __iter__(self):    
        indices = np.arange(self.n)
        for i in range(self.step):
            yield indices[:self.k], indices[self.k:self.k+self.h]
            self.k += self.leap
    
    def __len__(self):
        return self.k

from sklearn.feature_selection import RFECV

rfecv=RFECV(LinearRegression(), step=1, cv=TimeSeriesValidation(X_train.shape[0], 1),verbose=1); rfecv
rfecv.fit(X_train, y_train)
