import numpy as np
def lr_normal_equation(X,y):
    ones=np.ones(X.shape[0])
    X=np.c_[ones,X]
    print(X)
    w=np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
    return w

if __name__=='__main__':
    X=np.array([[ 0.04613557,  0.72334161],
       [ 0.33225315,  0.4250724 ],
       [-1.14747663,  0.33225003],
       [-1.58457724, -0.41830152],
       [ 0.49995133,  0.05056171],
       [ 0.59857517, -0.64770677],
       [-0.08798693,  0.61866969],
       [ 0.69359851, -0.99590893]])
    y=np.array([49.08463373,  58.48557211,  38.7655718,   5.50049909,

        11.24478596, -19.4285193,   9.06755436, -68.63086905])
    w=lr_normal_equation(X,y)
    print(w)
    print("intercept_:", w[0])
    print("coeffs_:", w[1:])





