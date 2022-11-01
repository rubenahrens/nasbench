from pyexpat import model
from ioh import problem, OptimizationType, get_problem,logger
import numpy as np
from nasbench import api
import os

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
_options = {1:CONV1X1,2:CONV3X3,0:MAXPOOL3X3}

NASBENCH_TFRECORD = './nasbench_only108.tfrecord'
# Load the data from file (this will take some time)
nasbench = api.NASBench(NASBENCH_TFRECORD)

def is_valid(x):
  matrix = np.empty((7,7),dtype=int)
  matrix = np.triu(matrix, 1)
  index = 0
  for i in range(7):
    for j in range(i+1,7):
      if not x[index] in [0,1]:
        return False
      matrix[i][j] = x[index]
      index += 1
  
  ops = []
  ops.append(INPUT)
  for i in range(5):
    if not x[index +i] in [0,1,2]:
      return False
    ops.append(_options[x[index + i]])
  ops.append(OUTPUT)

  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=matrix,   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=ops)
    
  if not nasbench.is_valid(model_spec):
    return False
  return True

def nas_ioh(x):
  matrix = np.empty((7,7),dtype=int)
  matrix = np.triu(matrix, 1)
  index = 0
  for i in range(7):
    for j in range(i+1,7):
      if not x[index] in [0,1]:
        return 0
      matrix[i][j] = x[index]
      index += 1
  
  ops = []
  ops.append(INPUT)
  for i in range(5):
    if not x[index +i] in [0,1,2]:
      return 0
    ops.append(_options[x[index + i]])
  ops.append(OUTPUT)

  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=matrix,   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=ops)
    
  if not nasbench.is_valid(model_spec):
    return 0
  
  tmp = nasbench.get_metrics_from_spec(model_spec)
  epoch = tmp[1][108]
  result = 0
  for _ in epoch:
    result += _["final_validation_accuracy"]
  result = result / 3.0
  return result

problem.wrap_integer_problem(nas_ioh, "nas101",  optimization_type=OptimizationType.MAX)

#Call get_problem to instantiate a version of this problem
f = get_problem('nas101', instance=0, dimension=26, problem_type="Integer")
logger = logger.Analyzer(
    root=os.getcwd(),                  # Store data in the current working directory
    folder_name="sstudentnumber1_sstudentnumber2",       # in a folder named: 'sstudentnumber1_sstudentnumber2'
    algorithm_name="random-search",    # meta-data for the algorithm used to generate these results. Please use the same name 'sstudentnumber1_sstudentnumber2' for the assignment
    store_positions=True               # store x-variables in the logged files
)
f.attach_logger(logger)

budget = 5000
runs = 20