import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yaml
#np.random.seed(100)

config=yaml.load(open("config.yml"),Loader=yaml.SafeLoader)

## generate data
## randn(n,sd) generates n normal data points
## add [x,y] to shift clusters along the axes ensuring they form around (x,y)
classA=np.concatenate((np.random.randn(10,2)*0.2+[0,1],np.random.randn(10,2)*0.2+[1,2]))
classB=np.concatenate((np.random.randn(20,2)*0.2+[1,1],np.random.randn(20,2)*0.2+[2,0]))

inputs=np.concatenate((classA,classB))
targets=np.concatenate((np.ones(classA.shape[0]),-np.ones(classB.shape[0])))

N=inputs.shape[0]
permute=list(range(N))
random.shuffle(permute)
inputs=inputs[permute,:]
targets=targets[permute]

P=np.zeros((N,N))


def kernel(x1,x2,type=config['kernel']):
    if type=='linear':
        return np.dot(x1,x2)
    
    elif type=='poly':
        p=config['kernel_params']
        return (np.dot(x1,x2)+1)**p

    elif type=='rbf':
        sigma=config['kernel_params']
        exponent=np.dot((x1-x2),(x1-x2))/(2*(sigma**2))
        return math.exp(-exponent)

def objective(alphas):
    """
    use global values for target labels and kernel function of x1 and x2
    """
    
    return 0.5*np.dot(alphas,np.dot(alphas,P))-np.sum(alphas)

def zerofun(alphas):
    """
    to be constrained to 0, i.e. sum(alpha-i * target-i) for all i
    """
    return np.dot(alphas,targets)

def threshold(support_vectors,alphas):
    res=0
    sv_target_sum=0
    sv=list(support_vectors.keys())[0]
    #for sv in support_vectors:
    for i in range(len(inputs)):
        res+=kernel(inputs[sv],inputs[i])*targets[i]*alphas[i]

    sv_target_sum+=support_vectors[sv]['target']
    print(support_vectors[sv])
    return res-sv_target_sum

def indicator(x,offset,alphas):
    transformed_data=np.array([kernel(x,input) for input in inputs])
    return np.sum(transformed_data*np.array(alphas)*np.array(targets))-offset


## populating P matrix
for i in range(len(inputs)):
    for j in range(len(inputs)):
        P[i][j]=targets[i]*targets[j]*kernel(inputs[i],inputs[j])

## selecting constraints
XC={
    'type':'eq',
    'fun':zerofun
}

## initializing weights
start=np.zeros(N)

## selecting config
C=config['C']
B=[(0,C)]*len(inputs)

## learning and returning weights
ret = minimize(objective, start, bounds=B, constraints=XC )
alpha = ret['x']
print(ret['success'])

## identofying support vectors
support_vectors={}
for i,v in enumerate(alpha):
    if (v>=math.pow(10,-5)) and (v<C if C!=None else True):
        support_vectors[i]={'alpha':v,'target':targets[i]}

## computing bias / offset 
thresh=threshold(support_vectors,alpha)
# print(f"############# THRESHOLD IS {thresh} ####################")
# for data,label in zip(inputs,targets):
#     print(data,label>=0)
#     print(f"prediction is {indicator(data,thresh,alpha)>=0}")

    
    
plt.plot( [ p [ 0 ] for p in classA ] , [ p [ 1 ] for p in classA ] , 'b.' )
plt.plot( [ p [ 0 ] for p in classB ] , [ p [ 1 ] for p in classB ] , 'r.' )
plt.axis( 'equal' ) # Force same s c a l e on both axes
xgrid=np.linspace(-5, 5)
ygrid=np.linspace(-4, 4)
grid=np.array([[indicator([x,y], thresh, alpha) for x in xgrid ] for y in ygrid])
kernel_used=config['kernel']
kernel_param=config['kernel_params'] if kernel_used!='linear' else 'None'
c_used=config['C']
plt.contour(xgrid , ygrid , grid , ( -1.0, 0.0, 1.0 ), colors=('red','black','blue'),linewidths =(1 , 3 , 1))
plt.title(f"Kernel: {kernel_used}\nHyperparameter value: {kernel_param}\nC value: {c_used}")
#plt.savefig( f'SVM_{kernel_used}kernel_{kernel_param}param_{c_used}C.pdf' ) # Save a copy in a f i l e
plt.show() # Show the p l o t on the screen