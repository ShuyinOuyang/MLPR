import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# tutorial1

def rbf_tut1_q3(xx, kk, hh):
    """Evaluate RBF kk with bandwidth hh on points xx (shape N,)"""
    c_k = (kk - 51) * hh / np.sqrt(2)
    phi = np.exp(-(xx - c_k) ** 2 / hh ** 2)
    return phi  # shape (N,)

# plotting code
def plotting():
    N = 70
    h = 0.2
    X = np.linspace(-1, 1, num=N, endpoint=True)
    K = 101
    plt.figure()
    for k in range(1, K+1):
        plt.plot(X, rbf_tut1_q3(X, k, h))
    plt.axis([-1, 1, 0, 1])
    plt.show()





def sigmoid():
    v = np.array([1,2])
    b = 5
    # x = np.linspace(-10, 10, 100, endpoint=True)
    x = np.array([[0.01*i, 0.01*i] for i in range(-500, 100)])
    sigma = 1/(1 + np.exp(-(x @ v.T + b)))
    plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.plot3D(x[:, 0], x[:, 1], sigma)
    plt.show()
    plt.plot(x[:, 0], sigma)
    plt.show()
    plt.plot(x[:, 1], sigma)
    plt.show()
def plot3d():
    fig = plt.figure()  #定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')

    #定义三维数据
    xx = np.arange(-5,5,0.5)
    yy = np.arange(-5,5,0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(X)+np.cos(Y)
    # 作图
    ax3.plot_surface(X,Y,Z,cmap='rainbow')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()


def fit():
    N = 70
    h = 0.2
    X = np.linspace(-1, 1, num=N, endpoint=True)
    yy = np.random.rand(N) * 2 - 1
    K = 101
    phi = np.array([[rbf_tut1_q3(x, k, h) for k in range(1, K + 1)] for x in X])
    phi_tilde = np.append(phi, [[np.sqrt(0.1) for i in range(1, K + 1)]], axis=0)
    Y_tilde = np.append(yy, [0])
    w_fit = np.linalg.lstsq(phi_tilde, Y_tilde, rcond=None)[0]
    print(w_fit)

    plt.figure()
    plt.scatter(X, yy, marker='x', color='orange')
    plt.plot(X, (phi_tilde @ w_fit)[:-1])
    plt.legend(['curve', 'data'])
    plt.show()

    h = 1
    phi = np.array([[rbf_tut1_q3(x, k, h) for k in range(1, K + 1)] for x in X])
    phi_tilde = np.append(phi, [[np.sqrt(0.1) for i in range(1, K + 1)]], axis=0)
    Y_tilde = np.append(yy, [0])
    w_fit = np.linalg.lstsq(phi_tilde, Y_tilde, rcond=None)[0]
    print(w_fit)

    plt.figure()
    plt.scatter(X, yy, marker='x', color='orange')
    plt.plot(X, (phi_tilde @ w_fit)[:-1])
    plt.legend(['curve', 'data'])
    plt.show()


if __name__ == "__main__":
    plotting()
    # sigmoid()
    # fit()
