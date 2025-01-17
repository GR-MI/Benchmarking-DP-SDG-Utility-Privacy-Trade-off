import numpy as np
import pandas as pd
import json
from datasets.domain import Domain

class Dataset:
    def __init__(self, df, domain):
        """ Create a Dataset object with a dataframe and domain """
        assert set(domain.attrs) <= set(df.columns), 'data must contain domain attributes'
        self.domain = domain
        self.df = df.loc[: ,domain.attrs]

    @staticmethod
    def synthetic(domain, N):
        """ Generate synthetic data conforming to the given domain """
        arr = [np.random.randint(low=0, high=n, size=N) for n in domain.shape]
        values = np.array(arr).T
        df = pd.DataFrame(values, columns=domain.attrs)
        return Dataset(df, domain)

    @staticmethod
    def load(path, domain_path):
        """ Load data into a dataset object """
        df = pd.read_csv(path)
        config = json.load(open(domain_path))
        domain = Domain(config.keys(), config.values())
        return Dataset(df, domain)

    def project(self, cols):
        """ Project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:, cols]
        domain = self.domain.project(cols)
        return Dataset(data, domain)

    def drop(self, cols):
        proj = [c for c in self.domain if c not in cols]
        return self.project(proj)

    def datavector(self, flatten=True, weights=None):
        """ Return the database in vector-of-counts form """
        bins = [range(n + 1) for n in self.domain.shape]
        ans = np.histogramdd(self.df.values, bins, weights=weights)[0]
        return ans.flatten() if flatten else ans
