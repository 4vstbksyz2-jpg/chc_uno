import os
import numpy as np
from tqdm import tqdm

def prepare_features_labels(data, n_ahead=1, gap=1):
    X = np.array(data[:-n_ahead, :, :])
    X = X[::gap, :, :]
    y = np.array(data[n_ahead:, :, :])
    y = y[::gap, :, :]
    return X, y


class PhaseOrdering:
    def __init__(self, l, dt, start, end, off=0.0, dtype: np.dtype = np.float64):
        self.l = l
        self.dt = dt
        self.start = start
        self.end = end
        self.dtype = dtype
        self.start_ = int(start / dt)
        self.end_ = int(end / dt)
        self.off = off
        self.ic = None
    
    def set_ic(self, seed):
        np.random.seed(seed)
        ic = np.random.uniform(-0.1, 0.1, size=(self.l, self.l)).astype(self.dtype)
        ic = ic - ic.mean() + self.off
        self.ic = ic
    
    def _del_sq(self, psi):
        psi_a = np.roll(psi,1,axis = 0)
        psi_b = np.roll(psi,-1,axis = 0)
        psi_c = np.roll(psi,1,axis = 1)
        psi_d = np.roll(psi,-1,axis = 1)
        dx = 1
        grad = (psi_a + psi_b + psi_c + psi_d - 4*psi)/dx**2
        return grad
    
    def chc(self, psi):
        data = np.zeros((self.end_, len(psi), len(psi))).astype(self.dtype)
        for i in range(self.end_):
            phi = psi**3 - psi - self._del_sq(psi)
            psi += self._del_sq(phi)*self.dt
            data[i, :, :] = psi
        return data[self.start_:, :, :]
    
    def tdgl(self, psi):
        data = np.zeros((self.end_, len(psi), len(psi))).astype(self.dtype)
        for i in range(self.end_):
            psi += (psi - psi**3 + self._del_sq(psi))*self.dt
            data[i, :, :] = psi
        return data[self.start_:, :, :]
    
    def generate_dataset(self, nens, n_ahead, gap, off_values=None, fname=None):
        """
        off: list of off values or None
        """
        np.random.seed(42)

        train_ens = int(0.8*nens)
        val_ens = nens - train_ens

        train_seeds = np.random.randint(0, 10000, train_ens)
        val_seeds = np.random.randint(10000, 20000, val_ens)

        if off_values is not None:
            n_off = len(off_values)
            train_seeds = np.array_split(train_seeds, n_off)
            val_seeds = np.array_split(val_seeds, n_off)

            train_X = []
            train_y = []

            for i, off_seeds in enumerate(train_seeds):
                off_value = off_values[i]
                if off_seeds.shape[0]>0:
                    for seed in tqdm(off_seeds, desc=f"Train | off = {off_value} | {i+1}/{n_off}"):
                        self.off = off_value
                        self.set_ic(seed)
                        X, y = prepare_features_labels(self.chc(self.ic), n_ahead, gap)
                        train_X.append(X)
                        train_y.append(y)
                

            train_X = np.array(train_X)
            train_y = np.array(train_y)

            val_X = []
            val_y = []

            for i, off_seeds in enumerate(val_seeds):
                if off_seeds.shape[0]>0:
                    off_value = off_values[i]
                    for seed in tqdm(off_seeds, desc=f"Validation | off = {off_value} | {i+1}/{n_off}"):
                        self.off = off_value
                        self.set_ic(seed)
                        X, y = prepare_features_labels(self.chc(self.ic), n_ahead, gap)
                        val_X.append(X)
                        val_y.append(y)

            val_X = np.array(val_X)
            val_y = np.array(val_y)

        else:
            train_X = []
            train_y = []

            for seed in tqdm(train_seeds, desc="Training Ensemble"):
                self.set_ic(seed)
                X, y = prepare_features_labels(self.chc(self.ic), n_ahead, gap)
                train_X.append(X)
                train_y.append(y)
            
            train_X = np.array(train_X)
            train_y = np.array(train_y)

            val_X = []
            val_y = []

            for seed in tqdm(val_seeds, desc="Validation Ensemble"):
                self.set_ic(seed)
                X, y = prepare_features_labels(self.chc(self.ic), n_ahead, gap)
                val_X.append(X)
                val_y.append(y)
            
            val_X = np.array(val_X)
            val_y = np.array(val_y)

        if fname is not None:
            np.savez(fname, train_X=train_X, train_y=train_y, val_X = val_X, val_y=val_y)
            return 

        return train_X, train_y, val_X, val_y

if __name__ == "__main__":
    train_nens = 25
    l = 64
    start = 0
    end = 300
    dt = 0.01
    n_ahead = 100
    gap = 10

    off_values = [-0.4, -0.2, 0.0, 0.2, 0.4]

    if off_values is not None:
        fname = f"td_{l}_{int(n_ahead)}_{int(start)}_{int(end)}_off.npz"
    else:
        fname = f"td_{l}_{int(n_ahead)}_{int(start)}_{int(end)}.npz"

    fname="test.npz"

    if fname not in os.listdir():
        system = PhaseOrdering(l=l, dt=dt, start=start, end=end)
        system.generate_dataset(nens=train_nens, n_ahead=n_ahead, gap=gap, off_values=off_values, fname=fname)