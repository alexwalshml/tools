import numpy as np


class CsvPCA:
    # class to perform principal component analysis on data stored in a .csv file
    # data is read and written out to new file internally

    def __init__(self,
                 infile_path: str,
                 outfile_path: str,
                 feature_dim: int = None,
                 variance_capture: float = None,
                 start_index: int = 0,
                 end_index: int = -1,
                 large: bool = False,
                 printouts: bool = False):
        self.infile_path = infile_path  # path to the data csv, must use "," as delimiter
        self.outfile_path = outfile_path  # path to the output csv
        self.feature_dim = feature_dim  # number of output dimensions
        self.variance_capture = variance_capture  # fraction of variance to capture, overrides feature_dim if not None
        self.start_index = start_index  # index of the beginning of the data to analyze
        self.end_index = end_index  # index of the end of the data to analyze
        self.large = large  # true if the file is too large to load into memory, slower than false
        self.printouts = printouts  # true if progress printouts are wanted, slower than false

        self.mean = None
        self.sigma = None

        if (feature_dim is None and variance_capture is None)\
                or (feature_dim is not None and variance_capture is not None):
            raise ValueError("Exactly one of feature_dim or variance_capture must be specified")
        if variance_capture is not None and (variance_capture <= 0 or variance_capture >= 1):
            raise ValueError("variance_capture must be a value between 0 and 1")

    def covariance_matrix(self):
        m_total = 0
        if self.printouts:
            m_total = sum(1 for _ in open(self.infile_path, "r"))
        if self.large:
            with open(self.infile_path, "r") as f:
                for m, line in enumerate(f):
                    if self.printouts:
                        print(f"Computing statistical quantities for covariance {m}/{m_total}", end="\r")
                    items = np.array([float(x) for x in line.strip("\n").split(",")][self.start_index: self.end_index])
                    items_vec = np.matrix(items)
                    if m == 0:
                        cov = np.zeros((len(items), len(items)))
                        x_sum = np.zeros(len(items))
                        x2_sum = np.zeros(len(items))
                    cov += items_vec.T @ items_vec
                    x_sum += items
                    x2_sum += items ** 2

            if self.printouts:
                print(f"Computing statistical quantities for covariance {m}/{m_total} Done!")
                print(f"Computing covariance matrix...", end="\r")

            mean = x_sum / (m + 1)
            self.mean = mean.flatten()
            x2_mean = x2_sum / (m + 1)

            sigma = np.sqrt(x2_mean - mean ** 2)
            self.sigma = sigma
            sigma = sigma.reshape((1, -1))
            sigma_matrix = sigma.T @ sigma

            x_sum = x_sum.reshape((1, -1))
            mean = mean.reshape((1, -1))

            cov -= mean.T @ x_sum
            cov -= x_sum.T @ mean
            cov += (m + 1) * mean.T @ mean
            cov /= (m + 1)
            cov = np.divide(cov, sigma_matrix, out=cov, where=sigma_matrix != 0)

            if self.printouts:
                print(f"Computing covariance matrix... Done!")

        else:
            data = np.empty((0, self.end_index - self.start_index))
            with open(self.infile_path, "r") as f:
                for m, line in enumerate(f):
                    items = np.array([float(x) for x in line.strip("\n").split(",")][self.start_index: self.end_index])
                    items = items.reshape((1, -1))
                    data = np.append(data, items, axis=0)

            mean = np.mean(data, axis=0)
            sigma = np.std(data, axis=0)

            self.mean = mean
            self.sigma = sigma

            numerator = data - mean
            standard_data = np.divide(numerator, sigma, out=numerator, where=sigma != 0)

            cov = np.cov(standard_data, rowvar=False, bias=True)

        return cov

    def principal_components(self):
        cov = self.covariance_matrix()
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        eigenvalue_mags = eigenvalues * eigenvalues.conj()
        normal_eigenvectors = np.array([e / np.linalg.norm(e) for e in eigenvectors])

        sorting = np.argsort(eigenvalue_mags)[::-1]

        sorted_eigenvalues = eigenvalues[sorting]
        sorted_eigenvectors = normal_eigenvectors[:, sorting]

        return sorted_eigenvalues, sorted_eigenvectors

    def feature_dim_fractional(self, val):
        val_num = len(val)
        cumulative_variance = np.array([sum(np.abs(val[i]) for i in range(j + 1)) for j in range(val_num)])
        cumulative_variance_fraction = cumulative_variance / sum(np.abs(val))

        feature_dim = 0
        for i, c in enumerate(cumulative_variance_fraction):
            if c > self.variance_capture:
                feature_dim = i
                break

        return feature_dim

    def feature_matrix(self):
        feature_dim = self.feature_dim
        val, vec = self.principal_components()
        
        if self.variance_capture is not None:
            feature_dim = self.feature_dim_fractional(val)

        matrix = vec[:, :feature_dim]

        return matrix
    
    def pca(self):
        feature_matrix = self.feature_matrix()
        with open(self.infile_path, "r") as f, open(self.outfile_path, "w+") as o:
            if self.large:
                for m, line in enumerate(f):
                    items = np.array([float(x) for x in line.strip("\n").split(",")][self.start_index: self.end_index])
                    numerator = items - self.mean
                    items = np.divide(numerator, self.sigma, out=numerator, where=self.sigma != 0)
                    items = items.reshape((-1, 1))
                    transformed = feature_matrix.T @ items
                    transformed_str = ""
                    for t in transformed.flatten():
                        transformed_str += str(t) + ","
                    transformed_str = transformed_str.strip(",") + "\n"
                    o.write(transformed_str)
            else:
                data = np.empty((0, self.end_index - self.start_index))
                for m, line in enumerate(f):
                    items = np.array([float(x) for x in line.strip("\n").split(",")][self.start_index: self.end_index])
                    numerator = items - self.mean
                    items = np.divide(numerator, self.sigma, out=numerator, where=self.sigma != 0)
                    items = items.reshape((1, -1))
                    data = np.append(data, items, axis=0)
                transformed_data = feature_matrix.T @ data.T
                transformed_data = transformed_data.T
                for td in transformed_data:
                    transformed_str = ""
                    for t in td:
                        transformed_str += str(t) + ","
                    transformed_str = transformed_str.strip(",") + "\n"
                    o.write(transformed_str)

        return
