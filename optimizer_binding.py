'''
Python binding for the Limbo StateBOptimizer using ctypes.
'''

import ctypes
import numpy as np
import os
import platform # Added for library extension

# --- Configuration ---
# TODO: Adjust this path to the actual location of your compiled shared library
# Common locations might be 'build/src/liblimbo.so' or similar
# Ensure the library name matches what Meson produces (e.g., .so on Linux, .dll on Windows)
LIB_SUFFIX = '.dll' if platform.system() == "Windows" else '.so'
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), 'build', f'liblimbo_interface{LIB_SUFFIX}')
# --- End Configuration ---

# --- ctypes Definitions ---

# Define the CVector structure mirror in Python
class CVector(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)),
                ("size", ctypes.c_int)]

# Define the function pointer type for the optimizer factory
# Updated to match C definition: void* (*)(int, int)
OptimizerFactoryFunc = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int, ctypes.c_int)

# Load the shared library
try:
    lib = ctypes.CDLL(LIBRARY_PATH)
except OSError as e:
    print(f"Error loading library at {LIBRARY_PATH}: {e}")
    print("Please ensure the library exists and the LIBRARY_PATH variable is set correctly.")
    print("Common causes: ")
    print("  - Project not built (run 'meson setup build && meson compile -C build')")
    print(f"  - Incorrect LIBRARY_PATH (current: {LIBRARY_PATH})")
    print("  - Architecture mismatch (e.g., 32-bit Python vs 64-bit library)")
    exit(1)

# --- Function Prototypes ---

# void* create_optimizer_instance(int dim_in, int dim_out); // Factory function exported from C++
# This should be the name of the factory function exported from C++
# (both in snap_interface.cpp and dummy_interface.cpp).
# It MUST now accept dim_in and dim_out.
FACTORY_SYMBOL_NAME = 'create_optimizer_instance' # Assuming this name is used consistently
try:
    # Check for the factory symbol's existence
    getattr(lib, FACTORY_SYMBOL_NAME)
except AttributeError:
    print(f"Error: Could not find symbol '{FACTORY_SYMBOL_NAME}' in the library.")
    print("Please ensure the factory function is correctly defined, exported (using extern \"C\"), and named in both snap and dummy C++ code.")
    print("The factory function must now accept (int dim_in, int dim_out).")
    exit(1)


# void* create_optimizer(OptimizerFactoryFunc factory_func, int dim_in, int dim_out);
lib.create_optimizer.restype = ctypes.c_void_p
# Updated argtypes
lib.create_optimizer.argtypes = [OptimizerFactoryFunc, ctypes.c_int, ctypes.c_int]

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
    # Updated __init__ to accept dimensions
    def __init__(self, dim_in: int = 5, dim_out: int = 1):
        '''Initializes the optimizer by calling the C++ create_optimizer with the factory.'''
        if not isinstance(dim_in, int) or dim_in <= 0:
            raise ValueError("dim_in must be a positive integer")
        if not isinstance(dim_out, int) or dim_out <= 0:
             raise ValueError("dim_out must be a positive integer")
            
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Get the factory function pointer from the loaded library
        try:
            factory_func_ptr = getattr(lib, FACTORY_SYMBOL_NAME)
            # Define argtypes/restype for the *specific* factory function pointer
            factory_func_ptr.restype = ctypes.c_void_p
            # Factory now takes dim_in, dim_out
            factory_func_ptr.argtypes = [ctypes.c_int, ctypes.c_int] 
            # Create the ctypes function pointer object for the factory type
            c_factory_func = OptimizerFactoryFunc(factory_func_ptr)
            
            # Call the C++ create_optimizer, passing the factory function AND dimensions
            self._handle = lib.create_optimizer(c_factory_func, self.dim_in, self.dim_out)
        except AttributeError:
             raise RuntimeError(f"Failed to find C++ factory function '{FACTORY_SYMBOL_NAME}'. Ensure it's exported correctly and accepts (int, int).")
        except Exception as e:
            raise RuntimeError(f"Error during optimizer creation: {e}")

        if not self._handle:
            raise RuntimeError("Failed to create C++ optimizer instance via factory (returned NULL).")
        print(f"Optimizer instance created with dim_in={self.dim_in}, dim_out={self.dim_out}.")

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
    # Example dimensions (MUST match what the optimizer expects)
    # dim_in often corresponds to the number of parameters being optimized
    # dim_out is often 1 (for a single reward/objective signal)
    example_dim_in = 5 
    example_dim_out = 1
    optimizer = StateOptimizerBinding(dim_in=example_dim_in, dim_out=example_dim_out)

    # Example state vector (adjust size based on your actual state dimension)
    # SnapStateBOptimizer::get_state() seems to use size 4 - this might be unrelated
    # to dim_in/dim_out but rather a fixed state representation.
    # Let's assume state size is fixed at 4 for the example.
    example_state = np.array([1.0, 2.0, 100.0, 50.0], dtype=np.float64)

    print(f"\nCalling act with state: {example_state}")
    action = optimizer.act(example_state)
    print(f" -> Received action: {action}")

    # Example sample and observation vectors (adjust sizes based on your needs)
    # act() returns the prediction, which is the sample for the next update
    # observation would typically be the measured outcome (e.g., reward/IOPS)
    # The size of the observation vector MUST match dim_out.
    example_sample = action # Use the action as the sample (size should match dim_in)
    example_observation = np.array([15000.0] * example_dim_out, dtype=np.float64) # size=dim_out
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