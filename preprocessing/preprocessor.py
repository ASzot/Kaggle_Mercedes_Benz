from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
import pandas as pd


class Preprocessor(object):
    def __init__(self, magicFeature, keepID = False):
        self.N_COMP = 12
        self.magicFeature = magicFeature
        self.keepID = keepID
        pass


    def transform(self, train, test):
        print('Converting categorical data')
        # Convert categorical data
        for c in train.columns:
            if train[c].dtype == 'object':
                lbl = LabelEncoder()
                lbl.fit(list(train[c].values) + list(test[c].values))
                train[c] = lbl.transform(list(train[c].values))
                test[c] = lbl.transform(list(test[c].values))

        # Remove the outlier
        print('Removing outlier')
        train = train[train['y'] < 250]

        col = list(test.columns)
        if not self.keepID:
            col.remove('ID')

        # tSVD
        print('Generating tSVD components')
        tsvd = TruncatedSVD(n_components=self.N_COMP)
        tsvd_results_train = tsvd.fit_transform(train[col])
        tsvd_results_test = tsvd.transform(test[col])

        # PCA
        print('Generating PCA components')
        pca = PCA(n_components=self.N_COMP)
        pca_results_train = pca.fit_transform(train[col])
        pca_results_test = pca.transform(test[col])

        # ICA
        print('Generating ICA components')
        ica = FastICA(n_components=self.N_COMP)
        ica_results_train = ica.fit_transform(train[col])
        ica_results_test = ica.transform(test[col])

        # GRP
        print('Generating GRP components')
        grp = GaussianRandomProjection(n_components=self.N_COMP, eps=0.1)
        grp_results_train = grp.fit_transform(train[col])
        grp_results_test = grp.transform(test[col])

        # SRP
        print('Generating SRP components')
        srp = SparseRandomProjection(n_components=self.N_COMP, dense_output=True)
        srp_results_train = srp.fit_transform(train[col])
        srp_results_test = srp.transform(test[col])

        print('Appending generated components')
        for i in range(1, self.N_COMP + 1):
            train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
            test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

            train['pca_' + str(i)] = pca_results_train[:, i - 1]
            test['pca_' + str(i)] = pca_results_test[:, i - 1]

            train['ica_' + str(i)] = ica_results_train[:, i - 1]
            test['ica_' + str(i)] = ica_results_test[:, i - 1]

            train['grp_' + str(i)] = grp_results_train[:, i - 1]
            test['grp_' + str(i)] = grp_results_test[:, i - 1]

            train['srp_' + str(i)] = srp_results_train[:, i - 1]
            test['srp_' + str(i)] = srp_results_test[:, i - 1]


        print('Appending magic features')
        if self.magicFeature:
            magic_mat = train[['ID', 'X0', 'y']]
            magic_mat = magic_mat.groupby(['X0'])['y'].mean()
            magic_mat = pd.DataFrame({
                'X0': magic_mat.index,
                'magic': list(magic_mat)
                })
            mean_magic = magic_mat['magic'].mean()
            train = train.merge(magic_mat, on='X0', how='left')
            test = test.merge(magic_mat, on='X0', how='left')
            test['magic'] = test['magic'].fillna(mean_magic)

        # Shuffle the data
        print('Shuffling data')
        train = train.sample(frac=1)

        return train, test
