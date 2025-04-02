#ifndef LIMBO_INTERFACE_H
#define LIMBO_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the Limbo Bayesian optimizer
 *
 * This function initializes the optimizer by:
 * - Creating the SNAP RPC interface
 * - Setting up the Bayesian optimization framework
 *
 * @return 0 on success, negative value on error
 */
int cpp_optimizer_init(void);

/**
 * @brief Perform one optimization iteration
 *
 * This function performs one optimization step:
 * - Get initial reward using SNAP performance metrics
 * - Wait 50ms
 * - Get reward again
 * - Calculate new parameter values based on optimization
 * - Set parameters in the SNAP system
 *
 * The optimizer attempts to maximize the reward value over time
 * by intelligently exploring the parameter space.
 *
 * @return 0 on success, negative value on error
 */
int cpp_optimizer_iteration(void);

/**
 * @brief Clean up resources used by the optimizer
 *
 * This function performs cleanup by:
 * - Freeing memory associated with the optimizer
 * - Closing the SNAP RPC connection
 *
 * @return 0 on success, negative value on error
 */
int cpp_optimizer_cleanup(void);

// C-compatible struct to return vector data
struct CVector {
  double *data;
  int size;
};

/**
 * @brief Create a new SnapStateBOptimizer instance
 *
 * @return Opaque pointer (handle) to the optimizer instance, or NULL on error.
 */
void *create_optimizer();

/**
 * @brief Destroy a SnapStateBOptimizer instance
 *
 * @param optimizer_handle Handle returned by create_optimizer()
 */
void destroy_optimizer(void *optimizer_handle);

/**
 * @brief Call the act method of the optimizer
 *
 * @param optimizer_handle Handle to the optimizer instance
 * @param state_data Pointer to the input state vector data
 * @param state_size Size of the input state vector
 * @return CVector struct containing pointer to the result vector data and its
 * size. The caller is responsible for freeing the data using
 * free_cvector_data().
 */
CVector optimizer_act(void *optimizer_handle, const double *state_data,
                      int state_size);

/**
 * @brief Call the update method of the optimizer
 *
 * @param optimizer_handle Handle to the optimizer instance
 * @param sample_data Pointer to the input sample vector data
 * @param sample_size Size of the input sample vector
 * @param observation_data Pointer to the input observation vector data
 * @param observation_size Size of the input observation vector
 */
void optimizer_update(void *optimizer_handle, const double *sample_data,
                      int sample_size, const double *observation_data,
                      int observation_size);

/**
 * @brief Call the best_arm_prediction method of the optimizer
 *
 * @param optimizer_handle Handle to the optimizer instance
 * @param state_data Pointer to the input state vector data
 * @param state_size Size of the input state vector
 * @return CVector struct containing pointer to the result vector data and its
 * size. The caller is responsible for freeing the data using
 * free_cvector_data().
 */
CVector optimizer_best_arm_prediction(void *optimizer_handle,
                                      const double *state_data, int state_size);

/**
 * @brief Call the best_bo_prediction method of the optimizer
 *
 * @param optimizer_handle Handle to the optimizer instance
 * @param state_data Pointer to the input state vector data
 * @param state_size Size of the input state vector
 * @return CVector struct containing pointer to the result vector data and its
 * size. The caller is responsible for freeing the data using
 * free_cvector_data().
 */
CVector optimizer_best_bo_prediction(void *optimizer_handle,
                                     const double *state_data, int state_size);

/**
 * @brief Free memory allocated for a CVector's data
 *
 * @param vec The CVector whose data needs freeing.
 */
void free_cvector_data(CVector vec);

#ifdef __cplusplus
}
#endif

#endif /* LIMBO_INTERFACE_H */
