#ifndef LIMBO_INTERFACE_H
#define LIMBO_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

struct CVector {
  double *data;
  int size;
};

/**
 * @brief Function pointer type for a factory function that creates an optimizer instance.
 * The factory function should return an opaque pointer (handle) to the
 * concrete optimizer instance, or NULL on error.
 */
typedef void* (*OptimizerFactoryFunc)(void);

/**
 * @brief Create a new optimizer instance using a provided factory function.
 *
 * @param factory_func A function pointer that creates and returns the optimizer.
 * @return Opaque pointer (handle) to the optimizer instance, or NULL on error.
 */
void *create_optimizer(OptimizerFactoryFunc factory_func);

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
