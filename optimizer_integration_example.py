import numpy as np
from optimization.env import MultiDimensionParabolic
from optimizer_binding import StateOptimizerBinding
env = MultiDimensionParabolic(dim=5,a0=np.ones(5)/5) #Optimal point at (0.2,...,0.2)
optimizer = StateOptimizerBinding()
for i in range(100):
    example_state = np.array([1.0, 2.0, 100.0, 50.0], dtype=np.float64) 
    action = optimizer.act(example_state)
    _,reward,_,_,_= env.step(action)
    print(reward)
    reward = np.array([-reward], dtype=np.float64)
    optimizer.update(action, reward)