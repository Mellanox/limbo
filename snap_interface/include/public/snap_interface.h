extern "C" {
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
}