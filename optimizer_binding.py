'''
Python binding for the Limbo StateBOptimizer using ctypes.
'''

import ctypes
import numpy as np
import os

# --- Configuration ---
# TODO: Adjust this path to the actual location of your compiled shared library
# Common locations might be 'build/src/liblimbo.so' or similar
# Ensure the library name matches what Meson produces (e.g., .so on Linux, .dll on Windows)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), 'build', 'liblimbo.so') 
# --- End Configuration ---

# --- ctypes Definitions ---

# Define the CVector structure mirror in Python
class CVector(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)),
                ("size", ctypes.c_int)]

# Load the shared library
try:
    lib = ctypes.CDLL(LIBRARY_PATH)
except OSError as e:
    print(f"Error loading library at {LIBRARY_PATH}: {e}")
    print("Please ensure the library exists and the LIBRARY_PATH variable is set correctly.")
    exit(1)

# --- Function Prototypes ---

# void* create_optimizer();
lib.create_optimizer.restype = ctypes.c_void_p
lib.create_optimizer.argtypes = []

# void destroy_optimizer(void* optimizer_handle);
lib.destroy_optimizer.restype = None
lib.destroy_optimizer.argtypes = [ctypes.c_void_p]

# CVector optimizer_act(void* optimizer_handle, const double* state_data, int state_size);
lib.optimizer_act.restype = CVector
lib.optimizer_act.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]

# void optimizer_update(void* optimizer_handle, const double* sample_data, int sample_size, 
#                       const double* observation_data, int observation_size);
lib.optimizer_update.restype = None
lib.optimizer_update.argtypes = [
    ctypes.c_void_p, 
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, 
    ctypes.POINTER(ctypes.c_double), ctypes.c_int
]

# CVector optimizer_best_arm_prediction(void* optimizer_handle, const double* state_data, int state_size);
lib.optimizer_best_arm_prediction.restype = CVector
lib.optimizer_best_arm_prediction.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int]


# void free_cvector_data(CVector vec);
lib.free_cvector_data.restype = None
lib.free_cvector_data.argtypes = [CVector]

# --- Helper Function for Vector Conversion ---
def _cvector_to_numpy(c_vec):
    '''Converts a CVector to a NumPy array and frees C memory.'''
    if not c_vec.data or c_vec.size == 0:
        lib.free_cvector_data(c_vec) # Free even if data is null, just in case
        return np.array([]) # Return empty array
    
    # Create a NumPy array viewing the C data (no copy yet)
    # Use ctypeslib.as_array for potentially better integration
    try:
        # Create a copy of the data from the C pointer
        numpy_array = np.ctypeslib.as_array(c_vec.data, shape=(c_vec.size,)).copy()
    except Exception as e:
        print(f"Error converting CVector data to NumPy array: {e}")
        numpy_array = np.array([]) # Return empty on error
    finally:
        # Always free the C-allocated memory
        lib.free_cvector_data(c_vec)
        
    return numpy_array

# --- Python Wrapper Class ---

class StateOptimizerBinding:
    '''
    Python wrapper for the C++ StateBOptimizer.
    Manages the lifecycle and provides methods to interact with the optimizer.
    Methods expect and return NumPy arrays.
    '''
    def __init__(self):
        '''Initializes the optimizer by calling the C++ create_optimizer.'''
        self._handle = lib.create_optimizer()
        if not self._handle:
            raise RuntimeError("Failed to create C++ optimizer instance.")
        print("Optimizer instance created.")

    def __del__(self):
        '''Cleans up the optimizer instance when the Python object is garbage collected.'''
        if hasattr(self, '_handle') and self._handle:
            print("Destroying optimizer instance...")
            lib.destroy_optimizer(self._handle)
            self._handle = None # Prevent double free

    def act(self, state: np.ndarray) -> np.ndarray:
        '''Calls the optimizer's act method.'''
        if not self._handle:
            raise RuntimeError("Optimizer instance is not valid.")
        state = np.ascontiguousarray(state, dtype=np.float64)
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result_cvector = lib.optimizer_act(self._handle, state_ptr, state.size)
        return _cvector_to_numpy(result_cvector)

    def update(self, sample: np.ndarray, observation: np.ndarray) -> None:
        '''Calls the optimizer's update method.'''
        if not self._handle:
            raise RuntimeError("Optimizer instance is not valid.")
        sample = np.ascontiguousarray(sample, dtype=np.float64)
        observation = np.ascontiguousarray(observation, dtype=np.float64)
        sample_ptr = sample.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        observation_ptr = observation.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.optimizer_update(self._handle, sample_ptr, sample.size, observation_ptr, observation.size)

    def best_arm_prediction(self, state: np.ndarray) -> np.ndarray:
        '''Calls the optimizer's best_arm_prediction method.'''
        if not self._handle:
            raise RuntimeError("Optimizer instance is not valid.")
        state = np.ascontiguousarray(state, dtype=np.float64)
        state_ptr = state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        result_cvector = lib.optimizer_best_arm_prediction(self._handle, state_ptr, state.size)
        return _cvector_to_numpy(result_cvector)

# --- Example Usage ---
if __name__ == "__main__":
    print("Creating optimizer...")
    optimizer = StateOptimizerBinding()

    # Example state vector (adjust size based on your actual state dimension)
    # The SnapStateBOptimizer::get_state() uses size 4
    example_state = np.array([1.0, 2.0, 100.0, 50.0], dtype=np.float64) 

    print(f"\nCalling act with state: {example_state}")
    action = optimizer.act(example_state)
    print(f" -> Received action: {action}")

    # Example sample and observation vectors (adjust sizes based on your needs)
    # act() returns the prediction, which is the sample for the next update
    # observation would typically be the measured outcome (e.g., reward/IOPS)
    example_sample = action # Use the action as the sample
    example_observation = np.array([15000.0], dtype=np.float64) # Dummy observation
    print(f"\nCalling update with sample: {example_sample}, observation: {example_observation}")
    optimizer.update(example_sample, example_observation)
    print(" -> Update called.")

    print(f"\nCalling best_arm_prediction with state: {example_state}")
    arm_pred = optimizer.best_arm_prediction(example_state)
    print(f" -> Received arm prediction: {arm_pred}")

    print("\nDestroying optimizer...")
    # Explicitly delete or let it go out of scope for __del__ to trigger
    del optimizer 
    print("Optimizer destroyed.") 