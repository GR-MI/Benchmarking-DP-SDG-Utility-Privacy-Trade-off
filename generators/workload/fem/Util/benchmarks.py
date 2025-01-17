import numpy as np
from datasets.dataset import Dataset
import itertools

def randomKway(dataset_path, domain_path, number, marginal, seed=0):
    data = Dataset.load(dataset_path, domain_path)
    return data, randomKwayData(data, number, marginal, seed)

def randomKwayData(data, number, marginal, seed=0):
    prng = np.random.RandomState(seed)
    total = data.df.shape[0]
    dom = data.domain
    proj = [p for p in itertools.combinations(data.domain.attrs, marginal) if dom.size(p) <= total]
    if len(proj) > number:
        proj = [proj[i] for i in prng.choice(len(proj), number, replace=False)]
    return proj
