# import matplotlib.image as img
# import matplotlib.pyplot as plt
from PIL import Image
from numpy import linalg as numLib
from sklearn.decomposition import PCA, IncrementalPCA
import imageio
import numpy as np


def recoverData(Z, U, K):
    # Compute the approximation of the data by projecting back onto
    # the original space using the top K eigenvectors in U.
    # Z: projected data

    new_U = U[:, :K]
    # We can use transpose instead of inverse because U is orthogonal.
    return Z.dot(new_U.T)


image = Image.open(r'./problem/69.jpg')
array = np.array(image)
print(array[0])
eigen_values, eigen_vectors = numLib.eig(array*(array.T))
# for k in [10, 50, 100, 150, 200, 250]:
#     ipca = IncrementalPCA(n_components=k)
#     img_recon = ipca.inverse_transform(ipca.fit_transform(image))
#     imageio.imwrite("./output/pca_output_"+str(k)+".jpg", img_recon)
np.abs(eigen_vectors)
new = recoverData(array, eigen_vectors, 10)
imageio.imwrite("./output/normal.jpg", new)