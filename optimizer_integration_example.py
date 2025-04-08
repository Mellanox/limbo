import numpy as np
from optimization.env import MultiDimensionParabolic
from optimization.optimizer import LimboOptimizer
from optimization.opt_analysis import create_optimizer_analyzer
from optimizer_binding import StateOptimizerBinding

dim = 5
minimize = True
optimal_point = np.arange(5)*3
use_analysis = True
env = MultiDimensionParabolic(dim=dim,a0=optimal_point)

optimization_params = {
        'parameters' : [{
                "name": f'x{i}',
                "type": "range",
                "bounds": [-100.0, 100.0],
                    } for i in range(dim)],
        'minimize' : minimize,
        'min_reward_range' : 0,
        'max_reward_range' : 500,
}


optimizer = LimboOptimizer(optimization_params,env,figure_save_path='figures') if not use_analysis else create_optimizer_analyzer(LimboOptimizer)(optimization_params,env,figure_save_path='figures')

optimizer.optimize(150)
optimizer.save_figures('png')
optimizer.save_figures('html')
