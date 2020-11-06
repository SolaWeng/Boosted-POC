import numpy as np
import cvxpy as cp
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.elementwise.sqrt import sqrt
from cvxpy.atoms.norm import norm
from scipy.linalg import sqrtm
from scipy.stats import norm as stdnormal
from scipy import optimize
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from .utils import mean, cov, p_n_split, normalize_weights

class opt_linear():
    '''
    parent class that includes common methods between different optimization problems.
    '''
    def __init__(self):
        self.weights = None
        
    def predict(self, X):
        '''
        Generate predictions based on the optimal linear classifier stored in the object
        
        Parameters:
        X(np.array): 2D matrix containing the whole dataset, rows are samples, columns are variables
        '''
        return np.sign(X@self.a_opt - self.b_opt).flatten().astype(int)
    
    def prepare(self, X, y, weights=None):
        '''
        Computes some dataset related values for setting up optimization later on. 
        All variable names align with the ones in the paper where p means postive, n means negative.
        
        Parameters:
        X(np.array): 2D matrix containing the whole dataset, rows are samples, columns are variables
        y(np.array): 1D array containing the labels of the whole dataset
        weights(np.array): 1D array of the sample weights or observation rates
        '''
        
        if self.weights is None:
            if weights is None:
                self.weights = np.ones(X.shape[0])/X.shape[0]
            else:
                if len(weights) != X.shape[0]:
                    raise dimensionError('weights dimension is inconsistant with input_dim')
                self.weights = weights
        # always normalize weights (sum to 1)
        self.weights = normalize_weights(self.weights)
        
        # compute necessary statistical parameters for later optimization use
        self.input_dim = X.shape[1]
        self.a_opt = np.zeros(self.input_dim)
        self.b_opt = 0
        X_p, y_p, w_p, X_n, y_n, w_n = p_n_split(X, y, self.weights)
        self.μ_p = mean(X_p, weights=w_p)
        self.μ_n = mean(X_n, weights=w_n)
        self.Σ_p = cov(X_p, weights=w_p)
        self.Σ_n = cov(X_n, weights=w_n)
        self.Σ_p_sqrt = sqrtm(self.Σ_p)
        self.Σ_n_sqrt = sqrtm(self.Σ_n)
        # set up varaibles for cvxpy
        self.a = cp.Variable(shape=(self.input_dim, 1))
        return X_p, y_p, w_p, X_n, y_n, w_n

class opt_linear_AtoB(opt_linear):
    '''
    class for the optimal linear classifier with tp and tn rate > 0.5.
    Refer to figure 1 in the paper.
    
    This problem is formulated as a convex optimization problem. Thus, it can be solved very efficiently using cvxpy
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __obj(self, λ):
        '''
        defines the objective function for section between A and B during the Trade-off analysis

        parameters:
        λ(float): a constant greater than 0
        '''

        if not ((isinstance(λ, float) or isinstance(λ, int)) and λ>0):
            raise ValueError('λ must be a float and is > 0')

        self.objFunc = norm(self.Σ_p_sqrt*self.a, 2) + λ*norm(self.Σ_n_sqrt*self.a, 2)
        self.obj = cp.Minimize(self.objFunc)

    def __constraint(self, sparse_ratio=None, seed=None):
        '''
        define the constraint set
        
        Parameters:
        sparse_ratio(float): persentage of weight parameters a to be 0
        '''
        
        if sparse_ratio is not None and 0<=sparse_ratio<=1:
            np.random.seed(seed = seed)
            self.sparsity = np.random.choice([0, 1], size=self.input_dim, p=[1-sparse_ratio, sparse_ratio])
            self.constraints = [transpose(self.a) * (self.μ_p - self.μ_n) == 1,
                                transpose(self.a) * np.diag(self.sparsity) == np.zeros(self.input_dim)[None]]
        elif sparse_ratio is None:
            self.constraints = [transpose(self.a) * (self.μ_p - self.μ_n) == 1]
        else:
            raise ValueError('sparse_ratio is not [0,1]')
        
    def compile(self, λ=1, sparse_ratio=None, seed=None):
        '''
        compile a problem by defining objectives and constraints
        
        Parameters:
        λ(float): a constant greater than 0
        '''
        self.__obj(λ)
        self.__constraint(sparse_ratio=sparse_ratio, seed=seed)
        self.prob = cp.Problem(self.obj, self.constraints)
    
    def solve(self, verbose=True, solver=cp.SCS, *args, **kwargs): # cp.ECOS is faster, but fails sometimes
        '''
        solve the optimal a and b for the linear classifier
        
        Parameters:
        verbose(bool): switch to the monitor of the solver
        solver: one of the cvxpy solvers
        '''
        self.prob.solve(verbose=verbose, solver=solver, *args, **kwargs)
        self.a_opt = self.a.value.flatten()
        self.b_opt = self.μ_p@self.a.value - 1/self.obj.value*np.sqrt((self.a.value.T@self.Σ_p@self.a.value))
        return self.a_opt, self.b_opt
    
class opt_linear_01toA(opt_linear):
    '''
    class for the optimal linear classifier with tn rate < 0.5 and tp rate > 0.5.
    Refer to figure 1 in the paper.
    
    This problem is formulated in the form of general constrained optimization problem since the definition of the problem is not convex and cannot be solved using cvxpy.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        
    def obj(self, var):
        '''
        defines the objective function for section between [0,1] and A during the Trade-off analysis

        parameters:
        var(np.array): concatenated array including vector a and bias b as the last element
        '''
        if len(var) != self.input_dim+1:
            raise ValueError('input dimension is not correct') 
        a = var[0:self.input_dim]
        b = var[-1]
        return -(a@self.μ_p - b)/np.sqrt(a@self.Σ_p@a)

    def constraint(self, var):
        '''
        defines the constraint function for section between [0,1] and A during the Trade-off analysis

        parameters:
        var(np.array): concatenated array including vector a and bias b as the last element
        '''
        if len(var) != self.input_dim+1:
            raise ValueError('input dimension is not correct')
        a = var[0:self.input_dim]
        b = var[-1]
        return stdnormal.cdf((b-a@self.μ_n)/np.sqrt(a@self.Σ_n@a))-self.α
        
    def compile(self, α=0.3):
        '''
        trivial but only to keep the same format as other classes

        parameters:
        α(float): target tn rate when maximizing the tp rate
        '''
        self.α = α
    
    def solve(self, initial_value=None, method="SLSQP"):
        '''
        solve the optimal a and b for the linear classifier
        
        Parameters:
        initial_value(np.array): initial values of the concatenated array of a and b, if None, random normal
        method(string): scipy optimize method string
        '''
        if initial_value == None:
            initial_value=np.random.rand(self.input_dim+1)
        solution = optimize.minimize(self.obj, initial_value, method=method, 
                                     constraints={"fun": self.constraint, "type": "eq"}, 
                                     options={'maxiter':100000})
        self.a_opt = solution.x[0:self.input_dim]
        self.b_opt = solution.x[-1]
        return solution
    
class opt_linear_Bto10(opt_linear):
    '''
    class for the optimal linear classifier with tn rate > 0.5 and tp rate < 0.5.
    Refer to figure 1 in the paper.
    
    This problem is formulated in the form of general constrained optimization problem since the definition of the problem is not convex and cannot be solved using cvxpy.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        
    def obj(self, var):
        '''
        defines the objective function for section between B and [1,0] during the Trade-off analysis

        parameters:
        var(np.array): concatenated array including vector a and bias b as the last element
        '''
        if len(var) != self.input_dim+1:
            raise ValueError('input dimension is not correct') 
        a = var[0:self.input_dim]
        b = var[-1]
        return -(b-a@self.μ_n)/np.sqrt(a@self.Σ_n@a)

    def constraint(self, var):
        '''
        defines the constraint function for section between B and [1,0] during the Trade-off analysis

        parameters:
        var(np.array): concatenated array including vector a and bias b as the last element
        '''
        if len(var) != self.input_dim+1:
            raise ValueError('input dimension is not correct')
        a = var[0:self.input_dim]
        b = var[-1]
        return stdnormal.cdf((a@self.μ_p-b)/np.sqrt(a@self.Σ_p@a))-self.β
        
    def compile(self, β=0.3):
        '''
        trivial but only to keep the same format as other classes

        parameters:
        β(float): target tp rate when maximizing the tn rate
        '''
        self.β = β
    
    def solve(self, initial_value=None, method="SLSQP"):
        '''
        solve the optimal a and b for the linear classifier
        
        Parameters:
        initial_value(np.array): initial values of the concatenated array of a and b, if None, random normal
        method(string): scipy optimize method string
        '''
        if initial_value == None:
            initial_value=np.random.rand(self.input_dim+1)
        solution = optimize.minimize(self.obj, initial_value, method=method, 
                                     constraints={"fun": self.constraint, "type": "eq"}, 
                                     options={'maxiter':1000})
        self.a_opt = solution.x[0:self.input_dim]
        self.b_opt = solution.x[-1]
        return solution
     
class opt_linear_kernel(opt_linear):
    '''
    class for the optimal rbf kernel linear classifier with tn rate > 0.5 and tp rate > 0.5.
    Refer to figure 1 in the paper.
    
    This problem is formulated as a convex optimization problem. Thus, it can be solved very efficiently using cvxpy
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare(self, X, y, gamma=0.7, δ_p = 1e-6, δ_n = 1e-6):
        '''
        compute extra information for gaussian kernel method
        
        Parameters:
        X(np.array): 2D matrix containing the whole dataset, rows are samples, columns are variables
        y(np.array): 1D array containing the labels of the whole dataset
        kernel(bool): a switch that determines if parameters for kenerl method will be computed 
        gamma(float): a constant used in the rbf kernel, check sklearn.metrics.pairwise.rbf_kernel
        δ_p/δ_n (float): regularization terms that ensures the S++ of the kernel covaiance matrix F
        '''
        
        X_p, y_p, w_p, X_n, y_n, w_n = super().prepare(X, y)

        self.gamma = gamma
        self.m_p = len(y_p)
        self.m_n = len(y_n)
        self.J_p = np.zeros((self.m_p+self.m_n, self.m_p+self.m_n))
        self.J_p[0:self.m_p, 0:self.m_p] = 1/np.sqrt(self.m_p)*(np.identity(self.m_p)-1/self.m_p*np.ones((self.m_p,self.m_p)))
        self.J_n = np.zeros((self.m_p+self.m_n, self.m_p+self.m_n))
        self.J_n[self.m_p:, self.m_p:] = 1/np.sqrt(self.m_n)*(np.identity(self.m_n)-1/self.m_n*np.ones((self.m_n,self.m_n)))
        self.g_p = np.zeros(self.m_p+self.m_n)
        self.g_p[0:self.m_p] = 1/self.m_p*np.ones(self.m_p)
        self.g_n = np.zeros(self.m_p+self.m_n)
        self.g_n[self.m_p:] = 1/self.m_n*np.ones(self.m_n)
        self.X_combine = np.append(X_p, X_n, axis=0)
        self.G = rbf_kernel(X=self.X_combine, Y=self.X_combine, gamma=self.gamma)
        self.F_p = self.G@self.J_p@self.J_p.T@self.G + δ_p*self.G
        self.F_n = self.G@self.J_n@self.J_n.T@self.G + δ_n*self.G
        self.F_p_sqrt = np.real(sqrtm(self.F_p))
        self.F_n_sqrt = np.real(sqrtm(self.F_n))
        # set up varaibles for cvxpy
        self.α_kernel = cp.Variable(shape=(len(self.G), 1))
        
    def __obj(self, λ):
        '''
        defines the objective function for section between A and B during the Trade-off analysis

        parameters:
        λ(float): a constant greater than 0
        '''

        if not ((isinstance(λ, float) or isinstance(λ, int)) and λ>0):
            raise ValueError('λ must be a float and is > 0')

        self.objFunc = norm(self.F_p_sqrt*self.α_kernel, 2) + λ*norm(self.F_n_sqrt*self.α_kernel, 2)
        self.obj = cp.Minimize(self.objFunc)

    def __constraint(self):
        '''
        define the constraint set
        '''
        self.constraints = [transpose(self.α_kernel) * (self.G@(self.g_p - self.g_n)) == 1]
        
    def compile(self, λ=1, seed=None):
        '''
        compile a problem by defining objectives and constraints
        
        Parameters:
        λ(float): a constant greater than 0
        '''
        self.__obj(λ)
        self.__constraint()
        self.prob = cp.Problem(self.obj, self.constraints)
    
    def solve(self, verbose=True, solver=cp.SCS):
        '''
        solve the optimal α_kernel and b_kernel for the kernel linear classifier
        
        Parameters:
        verbose(bool): switch to the monitor of the solver
        solver: one of the cvxpy solver
        '''
        self.prob.solve(verbose=verbose, solver=solver)
        self.α_kernel_opt = self.α_kernel.value.flatten()
        self.b_kernel_opt = self.α_kernel_opt@self.G@self.g_p-1/self.obj.value*np.sqrt(self.α_kernel_opt@self.F_p@self.α_kernel_opt)
        return self.α_kernel_opt, self.b_kernel_opt
    
    def predict(self, X_test):
        kernel_mat = rbf_kernel(self.X_combine, Y=X_test, gamma=self.gamma)
        y_pred = np.sign(self.α_kernel_opt@kernel_mat- self.b_kernel_opt).flatten().astype(int)
        return y_pred
    
class bootstrap_opt_linear():
    '''
    This class is a bootstrapping wrapper that is compatible with individual optimization classes.
    Note: current only tested with opt_linear_AtoB and opt_linear_kernel
    '''
    def __init__(self, X, y, solverClass, T=1, *args, **kwargs):
        '''
        Parameters:
        X(np.array): training data
        y(np.array): training label
        solverClass: choose opt_linear_AtoB or opt_linear_kernel (opt_linear_01toA and opt_linear_Bto10 are not tested)
        T(int): number of weak learners 
        '''
        self.X = X
        self.y = y
        self.classes = [solverClass(*args, **kwargs) for i in range(T)]
    
    def splitData(self, size, state):
        '''
        Splits data into sub sections for bootstrapping
        
        Parameters:
        size(float): a number between 0 and 1 indicating percentage of data to be selected
        state(int): random seed for sample selection
        '''
        _, self.X_sub, _, self.y_sub = train_test_split(self.X, self.y, test_size=size, random_state=state)
        
    def prepare(self, size=0.3):
        '''
        Precompute values in method 'prepare' in individual classes
        
        Parameters:
        size(float): a number between 0 and 1 indicating percentage of data to be selected
        '''
        for t in range(len(self.classes)):
            self.splitData(size=size, state=t)
            self.classes[t].prepare(self.X_sub, self.y_sub)
    
    def compile(self, *args, **kwargs):
        '''
        compute method 'compile' in individual classes
        '''
        for t in range(len(self.classes)):
            self.classes[t].compile(seed=t, *args, **kwargs)
    
    def solve(self, *args, **kwargs):
        '''
        compute method 'solve' in individual classes
        '''
        for t in range(len(self.classes)):
            self.classes[t].solve(*args, **kwargs)
        
    def predict(self, X_test):
        '''
        generate prediction
        
        Parameters:
        X_test(np.array): test set
        '''
        y_pred = []
        for t in range(len(self.classes)):
            y_pred.append(self.classes[t].predict(X_test))
        
        y_pred_bootstrap = np.sign(np.sum(y_pred, axis=0)-0.001).astype(int)
        return y_pred_bootstrap
    
    def release_mem(self):
        del self.X, self.y
        
class boosted_opt_linear():
    '''
    This is a Adaboost wrapper to the class opt_linear_AtoB.
    Note: This class is only compatible with opt_linear_AtoB
    '''
    def __init__(self, X, y, solverClass=opt_linear_AtoB, T=1, sparse_ratio=0.7, lr=0.7, λ=1, *args, **kwargs):
        '''
        Parameters:
        X(np.array): training data
        y(np.array): training label
        solverClass: choose opt_linear_AtoB (opt_linear_kernel, opt_linear_01toA and opt_linear_Bto10 are NOT compatible yet)
        T(int): number of weak learners 
        sparse_ratio(int): a number between 0 and 1 showing Percentage of linear classifier weights to be 0
        lr(int): a value between 0 and 1 showing the learning rate of Adaboost algorithm
        λ(float): check opt_linear_AtoB documentation
        '''
        self.X = X
        self.y = y
        self.T = T
        self.classes = [solverClass(*args, **kwargs) for i in range(T)]
        self.sparse_ratio = sparse_ratio
        self.lr = lr
        self.λ = λ
        self.weights = np.ones(self.X.shape[0])/self.X.shape[0]
        
    def __prepare(self, solverClass, weights):
        solverClass.prepare(self.X, self.y, weights=weights)
        
    def __compile(self, solverClass, *args, **kwargs):
        solverClass.compile(*args, **kwargs)
        
    def __solve(self, solverClass, *args, **kwargs):
        solverClass.solve(*args, **kwargs)
    
    def weakLearning(self, solverClass, seed):
        self.__prepare(solverClass, weights=self.weights)
        self.__compile(solverClass, λ=self.λ, sparse_ratio=self.sparse_ratio, seed = seed)
        self.__solve(solverClass, verbose=False)
        
    def update_weights(self, solverClass):
        y_hat = solverClass.predict(self.X)
        self.ε = self.weights@abs(y_hat-self.y)/2
        ######################### self.β is occupied in opt_linear_Bto10(current not compatible), modify in the future
        solverClass.β = self.lr*np.log((1-self.ε)/self.ε)
        indicator = (y_hat-self.y)/2
        indicator = (indicator<0)*(1)/(1+self.λ)+(indicator>0)*(self.λ)/(1+self.λ)
        self.weights *= np.exp(solverClass.β*indicator)
        
    def solve(self):
        for t in range(self.T):
            self.weakLearning(self.classes[t], seed = t)
            self.update_weights(self.classes[t])
        
    def predict(self, X_test):
        y_pred = np.zeros(len(X_test))
        for cls in self.classes:
            y_pred += cls.β*cls.predict(X_test)
        return np.sign(y_pred)
    
    def release_mem(self):
        del self.X, self.y