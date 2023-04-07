import numpy as np
import matplotlib.pyplot as plt
from time import time
from bipls_table import biplstable
from sub_bipls_limit import sub_bipls_limit
from sub_bipls_vector_limit import sub_bipls_vector_limit

#import numpy as np
from sort_ipls import sort_ipls
from sub_iplsreverse import sub_iplsreverse, sizer

from sub_pls_val import sub_pls_val
#import numpy as np
#from sub_iplsreverse import sizer

#import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

#import numpy as np
#import pandas as pd
from scipy.spatial.distance import cdist
#from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

#from sklearn.cross_decomposition import PLSRegression
#import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
#import matplotlib.pyplot as plt

import matplotlib.collections as collections
from sys import stdout

from sub_pls_pre import sub_pls_pre
from sub_pls import sub_pls
from plsonecomp import plsonecomp