#Submission by: Mandala Tejesh - CS19B062
import matplotlib.pyplot as plt
from PIL import Image
from numpy import linalg as numLib
import imageio
import numpy as np
import sys


def complex_sort(lam):
    A = abs(lam)
    A = np.argsort(A)[::-1]
    return A

def evd(A):
    eVals, eVecs = numLib.eig(A) # eigen decomposition
    order = complex_sort(eVals) # get the order of elements if sorted
    V = eVecs[:, order] # sort eigen vectors
    sorted_eig_values = eVals[order] 

    D = np.diag(sorted_eig_values) # diagonal matrix
    V_inverse = numLib.inv(V) # Inverse of eigen vector matrix
    norms = [] # keeps track of frobenius norms for various values of k

    for k in range(0,256):
        # reconstruct the image using top k eigen values
        recon = abs(V[:, :k] @ D[:k, :k] @ V_inverse[:k, :])
        # imageio.imwrite(f"./output/evd/evd_{k}.jpg", recon)
        frNorm = numLib.norm(abs(recon-A))
        norms.append(frNorm)

        ## comparitive plot for particular k, add sufficient condition and uncomment the following lines
        # plt.clf()
        # fig, axes = plt.subplots(1, 2, constrained_layout=True)
        # axes[1].imshow(abs(recon-A), cmap='gray')
        # axes[1].axis('off')
        # axes[1].title.set_text("error image")
        # axes[0].imshow(recon, cmap='gray')
        # axes[0].axis('off')
        # axes[0].title.set_text("reconstructed")
        # plt.savefig(f'./plots/evd/evd_{k}.png')
    return norms

def svd(A):
    # A = U x Sigma x V.T
    eVals, eVecs = numLib.eig(A.T @ A) # eigen decomposition
    order = complex_sort(eVals)
    V = eVecs[:, order] # sort eigen vectors

    sorted_eig_values = eVals[order] 

    singular_values = np.sqrt(sorted_eig_values)
   
    r = len(singular_values)
    U = A @  V[:,:r] / singular_values
    VT = V.T
    Sigma = np.diag(singular_values) # diagonal matrix with singular values on diagonal
    norms = [] # to store error for various values of k

    for k in range(0,256):
        # reconstruct the image from top k singular values
        recon = abs(U[:, :k] @ Sigma[:k,:k] @ VT[:k, :])
        # calculate the frobenius norm and store it
        frNorm = numLib.norm(abs(recon-A))
        norms.append(frNorm)
        
        # imageio.imwrite(f"./output/svd/svd_{k}.jpg", recon)
        # fig, axes = plt.subplots(1,2, constrained_layout=True)
        # axes[1].imshow(abs(recon-A), cmap='gray')
        # axes[1].axis('off')
        # axes[1].title.set_text("error image")
        # axes[0].imshow(recon, cmap='gray')
        # axes[0].axis('off')
        # axes[0].title.set_text("reconstructed")
        # plt.savefig(f'./plots/svd/svd_{k}.png')


    return norms

def main():
    A = plt.imread('./problem/69.jpg').astype(np.int64)
    if len(sys.argv) > 1:
        job = sys.argv[1]
    else:
    	job = 'all'
    plot_norm_evd = []
    plot_norm_svd = []
    if job == 'evd' or job == 'all':
        plot_norm_evd = evd(A)
    if job=='svd' or job=='all':
        plot_norm_svd = svd(A)

    
    axis = [i for i in range(0,max(len(plot_norm_evd), len(plot_norm_svd)))]
    ## plot the comparitive error graph
    plt.clf()
    arr = []
    if job=='all' or job=='evd':
        plt.plot(axis, plot_norm_evd, 'r')
        arr.append('evd')
    if job=='all' or job=='svd':
        plt.plot(axis, plot_norm_svd, 'b')
        arr.append('svd')
    plt.ylabel('Frobenius norm')
    plt.xlabel('value of k')
    plt.title('Image Reconstruction using top k eigen values/singular values')
    plt.legend(arr)
    plt.savefig('./plots/comparitivePlot.jpg')

    

main()
