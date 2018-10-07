import numpy as np
from scipy.stats import uniform as sp_rand


param_mars_dist = {"max_terms": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "max_degree": [ 2 ],
                   "penalty": sp_rand(),
                   "endspan_alpha": sp_rand(),
                   "endspan": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "minspan_alpha": sp_rand(),
                   "minspan": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "thresh": sp_rand(),
                   "zero_tol": sp_rand(),
                   "allow_missing": [ True, False ],
                   "min_search_points": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "check_every": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "allow_linear": [ True, False ],
                   "use_fast": [ True, False ],
                   "fast_K": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "fast_h": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ],
                   "smooth": [ True, False ],
                   "enable_pruning": [ True ],
                   "feature_importance_type": [ 'gcv', 'rss', 'nb_subsets' ],
                   "verbose": [ int(x) for x in np.linspace(start=1, stop=1000, num=100) ]
                   }