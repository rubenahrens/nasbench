from ioh import problem, OptimizationType, get_problem
import numpy as np
from nasbench import api

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
_options = {1:CONV1X1,2:CONV3X3,0:MAXPOOL3X3}

NASBENCH_TFRECORD = './nasbench_only108.tfrecord'
nasbench = api.NASBench(NASBENCH_TFRECORD)

def nas_ioh(x):
  matrix = np.empty((7,7),dtype=int)
  index = 0
  for i in range(7):
    for j in range(7):
      matrix[i][j] = x[index]
      index += 1
  
  ops = []
  ops.append(INPUT)
  for i in range(5):
    ops.append(_options[x[index + i]])
  ops.append(OUTPUT)

  model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=matrix,   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=ops)
    
  if not nasbench.is_valid(model_spec):
    return 0
  return nasbench.query(model_spec)["validation_accuracy"]

problem.wrap_real_problem(nas_ioh, "nas101",  optimization_type=OptimizationType.Maximization)

#Call get_problem to instantiate a version of this problem
f = get_problem('nas101', instance=0, dimension=54)
