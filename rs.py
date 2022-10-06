
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

def random_sampling_model(bench):
  while True:
    matrix = np.random.choice([0, 1], size=(7, 7))
    matrix = np.triu(matrix, 1)

    operations = [CONV1X1,CONV3X3,MAXPOOL3X3]
    ops = np.random.choice(operations,7)
    ops[0] = INPUT
    ops[6] = OUTPUT
    model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=matrix,
      # Operations at the vertices of the module, matches order of matrix
      ops=list(ops))
    # check if the model is valid
    if bench.is_valid(model_spec):
      break
  
  x = np.empty(26,dtype=int)
  index = 0
  for i in range(7):
    for j in range(i+1,7):
      x[index] = matrix[i][j]
      index += 1
  for i in range (1,6):
    if ops[i] == CONV1X1:
      x[index] = 1
      index+=1
    elif ops[i] == CONV3X3:
      x[index] = 2
      index+=1
    elif ops[i] == MAXPOOL3X3:
      x[index] = 0
      index+=1
  return x

def main(argv):
  del argv  # Unused
  for r in range(nas_ioh.runs): # we execute the algorithm with 20 independent runs.
    f_best = sys.float_info.min
    for _ in range(nas_ioh.budget): # budget as 5000
      x = random_sampling_model(nas_ioh.nasbench) # sample a valid module randomly
      y = nas_ioh.f(x) # evaluate the performance of the module x. The logging will be executed in this function automatically.
      if y > f_best:
        f_best = y
        x_best = x
    print("run ", r, ", best x:", x_best,", f :",f_best)
    nas_ioh.f.reset() # Note that you must run the code after each independent run.


# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
  start = time.time()
  app.run(main)
  end = time.time()
  print("The program takes %s seconds" % (end-start))
