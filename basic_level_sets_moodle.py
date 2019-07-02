# Modified from Agustinus Kristiadi's Blog
# https://wiseodd.github.io/techblog/2016/11/20/levelset-segmentation/
import math
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import cv2 as cv



def grad(x):
    return np.array(np.gradient(x))


def norm(x, axis=0):
    return np.sqrt(np.sum(np.square(x), axis=axis))


def stopping_fun(x):
    return 1. / (1. + norm(grad(x))**2)

############
def div(x):
   dy = np.gradient(x[0], axis = 0)
   dx = np.gradient(x[1], axis = 1)
   return abs(dx + dy)
############

# Funciton for generatign the initial level set function phi
def initialize_phi(x, initshape ='rectangle'):
    phi = -1.0*np.ones(x.shape[:2])
    if initshape == 'rectangle':
    	b = 10
    	phi[b:-b, b:-b] = 1.
    else: # Circle
        r = 48
        M, N = phi.shape
        a = M/2
        b = N/2
        y,x = np.ogrid[-a:M-a, -b:N-b]
        mask = x*x + y*y <= r*r
        phi[mask] = 1.    
    return phi  

# Original image
imgo = cv.imread('two_obj_rr.png', cv.IMREAD_GRAYSCALE)
img = imgo - np.mean(imgo)

img_smooth = cv.GaussianBlur(img,(5,5),0)

# An inverse measure of smoothed gradient as the stopping fuction
g = stopping_fun(img_smooth)#F in the blog
# Forcing the stopping function to be zero at high-gradient regions (as our V is simple here)
g[g < 0.1] = 0.0 # I canont justify the reason fo this. 

# Initial phi (computed usign the function phi)
phi = initialize_phi(img_smooth, 'rectangle')

# Plotting the image, F and initial phi
fig1 = plt.figure()
ax1 = fig1.add_subplot(131)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Image')
ax2 = fig1.add_subplot(132)
ax2.imshow(g, cmap=cm.coolwarm)
ax2.title.set_text('F')
ax3 = fig1.add_subplot(133)
ax3.imshow(phi, cmap=cm.coolwarm)
ax3.title.set_text(r"$\phi$")
plt.pause(1.)


# Plotting the (clipped) level-set function phi and the image with level curve
fig2 = plt.figure()
ax3 = fig2.add_subplot(121, projection='3d')
ax3.view_init(elev=30., azim=-210)
M, N = phi.shape
X = np.arange(0, N, 1)
Y = np.arange(0, M, 1)
X, Y = np.meshgrid(X, Y)
ax4 = fig2.add_subplot(122)
ax4.imshow(imgo, cmap='gray')

# Time step
dt = 1.
# No. of iterations
n_iter = 1000
ims = []
for i in range(n_iter):
    
    grad_phi = grad(phi)
    grad_phi_norm = norm(grad_phi)

    #phi_t = - g * grad_phi_norm

    ###########################

    phi_t =  - grad_phi_norm - (g * grad_phi_norm)
    #phi_t = - div(np.nan_to_num(grad_phi / grad_phi_norm)) * grad_phi_norm
    #phi_t = - div(g * (np.nan_to_num(grad_phi / grad_phi_norm))) * grad_phi_norm
    #phi_t =  - g * div(np.nan_to_num(grad_phi / grad_phi_norm)) * grad_phi_norm

    ##########################
    phi = phi + dt * phi_t

    # Plottign the level set function phi and the zero-level curve on the image
    ax3.cla()
    surf = ax3.plot_surface(X, Y, np.clip(phi, -3.0, 3.0), cmap=cm.bwr,
    linewidth=0, antialiased=False) # I clipped phi fro visualization as it grows large in negative direction
    plt.pause(0.000001)
    for c in ax4.collections:
        c.remove()
    ax4.contour(phi, levels=[0], colors=['red'])
    fig2.suptitle("Iterations {:d}".format(i))

# Plot when iterations are done
surf = ax3.plot_surface(X, Y, phi, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax4.contour(phi, levels=[0], colors=['green'])
plt.show()