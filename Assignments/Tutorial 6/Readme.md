Dear Student,
Please find the Gaussian Process lecture slides in the attachments.
For you assignment please use the following code for generating data set (__ is used for indentation). Rest of the code is given in the slides.

def dataSet_2():

__X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
__Y_train = np.sin(X_train)
__X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)
__return X_train,Y_train,X_test

For plotting variance and mean use the following code:

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):

__X = X.ravel()
__mu = mu.ravel()
__uncertainty = 1.96 * np.sqrt(np.diag(cov))

__plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
__plt.plot(X, mu, label='Mean')
__for i, sample in enumerate(samples):
____plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
__if X_train is not None:
____plt.plot(X_train, Y_train, 'rx')
__plt.legend()

Best Wishes.
