#include "snap_interface.h"
#include "snap_state_optimizer.hpp"
#include <public/interface.hpp>
#include <Eigen/Core>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <limits>
#include <tuple>
#include <snap_optimizer_interface.h>
#include <cmath>

#define CYCLE_TIME std::chrono::seconds(12)
#define STATS_PRINT_INTERVAL 1000
#define DIRTY_WINDOW_MIN_LENGTH std::chrono::milliseconds(50)

#define TIME_BLOCK(CODE_TO_RUN, RESULT_VECTOR) \
    do { \
        auto _macro_start_time = std::chrono::high_resolution_clock::now(); \
        { \
            CODE_TO_RUN; \
        } \
        auto _macro_end_time = std::chrono::high_resolution_clock::now(); \
        RESULT_VECTOR.push_back(_macro_end_time - _macro_start_time); \
    } while (0)


static SnapStateBOptimizer* snap_optimizer = nullptr;

enum class OptState {
    ACTING,
    UPDATING,
    WAITING_FOR_PREDICTION,
    PREDICTING,
    WAITING_FOR_CYCLE
};

static OptState current_op_state = OptState::ACTING;
static Eigen::VectorXd last_sample;
static std::chrono::high_resolution_clock::time_point last_update_time;
static std::chrono::high_resolution_clock::time_point cycle_start_time;
static Eigen::VectorXd current_best_arm;

// Store raw durations instead of double counts
static std::vector<std::chrono::high_resolution_clock::duration> act_times;
static std::vector<std::chrono::high_resolution_clock::duration> update_times;
static std::vector<std::chrono::high_resolution_clock::duration> predict_times;
static int iteration_count = 0;

// Add variable for tracking state entry time
static std::chrono::high_resolution_clock::time_point state_entry_time;

// Add variable for reward calculation
static uint64_t performance_metric_before = 0;
// Add variable to store reward for next update cycle
static double last_calculated_reward = 0.0;

// Helper function to convert from normalized 0-1 to exact power of 2
// Based on provided SnapParams reference
static int from_power2_space(double normalized_value, int min_power, int max_power) {
    // Convert normalized value to power of 2 exponent
    double power = min_power + normalized_value * (max_power - min_power);
    return 1 << static_cast<int>(std::round(power));
}

// Helper function to scale, clamp, and apply parameters
static void apply_scaled_parameters(const Eigen::VectorXd& params_vector) {
    int poll_size = 1, max_inflights = 1, max_iog_batch = 1, max_new_ios = 1;
    double poll_ratio = 0.5; // Default

    if (params_vector.size() >= 5) {
        // Scale parameters using logic from SnapParams::from_optimization_space
        poll_size     = from_power2_space(params_vector[0], 0, 8);  // 1-256
        poll_ratio    = params_vector[1];                          // 0-1
        max_inflights = from_power2_space(params_vector[2], 0, 16); // 1-65536
        max_iog_batch = from_power2_space(params_vector[3], 0, 12); // 1-4096
        max_new_ios   = from_power2_space(params_vector[4], 0, 12); // 1-4096
        
        // Clamp results just in case normalized value was slightly outside [0,1]
        poll_size = std::max(1, std::min(poll_size, 256));
        poll_ratio = std::max(0.0, std::min(poll_ratio, 1.0));
        max_inflights = std::max(1, std::min(max_inflights, 65535)); // Clamp to actual max 65535
        max_iog_batch = std::max(1, std::min(max_iog_batch, 4096));
        max_new_ios = std::max(1, std::min(max_new_ios, 4096));
    } else {
        std::cerr << "Warning: Parameter vector dimension unexpected (" << params_vector.size() << "), using default params for set_system_params." << std::endl;
    }

    snap_optimizer_set_system_params(
        poll_size,
        poll_ratio,
        max_inflights,
        max_iog_batch,
        max_new_ios
    );
}

void print_stats() {
    // Lambda to calculate stats on duration vectors
    auto calculate_stats = [](const std::vector<std::chrono::high_resolution_clock::duration>& times) {
        using Duration = std::chrono::high_resolution_clock::duration;
        if (times.empty()) {
            // Return tuple of zero durations
            return std::make_tuple(Duration::zero(), Duration::zero(), Duration::zero());
        }
        // Accumulate durations
        Duration sum = std::accumulate(times.begin(), times.end(), Duration::zero());
        // Calculate mean duration
        Duration mean = sum / times.size();
        // Find min/max durations
        Duration min_val = *std::min_element(times.begin(), times.end());
        Duration max_val = *std::max_element(times.begin(), times.end());
        return std::make_tuple(min_val, max_val, mean);
    };

    // Get stats as durations
    std::chrono::high_resolution_clock::duration min_act_d, max_act_d, avg_act_d;
    std::tie(min_act_d, max_act_d, avg_act_d) = calculate_stats(act_times);

    std::chrono::high_resolution_clock::duration min_update_d, max_update_d, avg_update_d;
    std::tie(min_update_d, max_update_d, avg_update_d) = calculate_stats(update_times);

    std::chrono::high_resolution_clock::duration min_predict_d, max_predict_d, avg_predict_d;
    std::tie(min_predict_d, max_predict_d, avg_predict_d) = calculate_stats(predict_times);

    // Convert to milliseconds only for printing
    auto to_ms = [](std::chrono::high_resolution_clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- Aggregated Stats (last " << STATS_PRINT_INTERVAL << " cycles) --- \n";
    std::cout << "--- Op Timings --- \n";
    // Print counts from original vectors for call count
    std::cout << "Act     (ms): min=" << to_ms(min_act_d) << ", max=" << to_ms(max_act_d) << ", avg=" << to_ms(avg_act_d) << " (" << act_times.size() << " calls)\n";
    std::cout << "Update  (ms): min=" << to_ms(min_update_d) << ", max=" << to_ms(max_update_d) << ", avg=" << to_ms(avg_update_d) << " (" << update_times.size() << " calls)\n";
    std::cout << "Predict (ms): min=" << to_ms(min_predict_d) << ", max=" << to_ms(max_predict_d) << ", avg=" << to_ms(avg_predict_d) << " (" << predict_times.size() << " calls)\n";
    std::cout << "--------------------------------------------\n";

    // Clear duration vectors
    act_times.clear();
    update_times.clear();
    predict_times.clear();
    // Initialize state entry time
    state_entry_time = std::chrono::high_resolution_clock::now();
    // Get initial performance metric
    performance_metric_before = snap_optimizer_get_performance_metric();
    return;
}

// Factory function now takes dimensions
void* snap_optimizer_factory(int dim_in, int dim_out) {
    // Pass dimensions to the constructor
    return new SnapStateBOptimizer(dim_in, dim_out);
}
extern "C" {
int cpp_optimizer_init(void) {
    void* handle = create_optimizer(&snap_optimizer_factory, 5, 1);
    snap_optimizer = static_cast<SnapStateBOptimizer*>(handle);

    if (!snap_optimizer) {
        std::cerr << "Error: Failed to create optimizer instance." << std::endl;
        return -1;
    }
    current_op_state = OptState::ACTING;
    iteration_count = 0;
    act_times.clear();
    update_times.clear();
    predict_times.clear();
    return 0;
}

int cpp_optimizer_iteration(void) {
    if (!snap_optimizer) {
        std::cerr << "Error: Optimizer not initialized." << std::endl;
        return -1;
    }

    auto now = std::chrono::high_resolution_clock::now();

    switch (current_op_state) {
        case OptState::ACTING: {
            // Use the getter method for dimension
            Eigen::VectorXd current_state = Eigen::VectorXd::Zero(snap_optimizer->get_dim_in());

            
            TIME_BLOCK({
                last_sample = snap_optimizer->act(current_state); 
            }, act_times);

            auto act_end_time = std::chrono::high_resolution_clock::now();
            auto duration_in_state = act_end_time - state_entry_time;
            std::cout << "State: ACTING -> UPDATING. Sample: " << last_sample.transpose()
                      << " (Spent " << std::chrono::duration<double, std::milli>(duration_in_state).count() << "ms in ACTING)" << std::endl;
            state_entry_time = act_end_time;

            current_op_state = OptState::UPDATING;
            break;
        }

        case OptState::UPDATING: {
            // Create observation from the reward calculated in the previous cycle
            Eigen::VectorXd observation = Eigen::VectorXd::Constant(1, last_calculated_reward);
            
            // Capture metric BEFORE the conditional check
            performance_metric_before = snap_optimizer_get_performance_metric();

            // Conditionally perform update and set_params based on reward
            if (last_calculated_reward == 0.0) {
                std::cout << "Skipping optimizer update and set_params due to zero reward." << std::endl;
                 // NOTE: update_times vector will not get an entry for this cycle
            } else {
                // Only run block if reward is non-zero
                TIME_BLOCK({
                    snap_optimizer->update(last_sample, observation);
                    // Call helper function to apply scaled params
                    apply_scaled_parameters(last_sample); 
                }, update_times);
            }
            
            // Proceed regardless of whether update was skipped
            last_update_time = std::chrono::high_resolution_clock::now();
            
            auto duration_in_state = last_update_time - state_entry_time;
            std::cout << "State: UPDATING -> WAITING_FOR_PREDICTION. Observation: " << observation.transpose()
                      << " (Spent " << std::chrono::duration<double, std::milli>(duration_in_state).count() << "ms in UPDATING)" << std::endl;
            state_entry_time = last_update_time;

            current_op_state = OptState::WAITING_FOR_PREDICTION;
            break;
        }

        case OptState::WAITING_FOR_PREDICTION: {
            auto elapsed = now - last_update_time;
            if (elapsed >= DIRTY_WINDOW_MIN_LENGTH) {
                // Calculate duration in state
                auto duration_in_state = now - state_entry_time;
                // Convert duration to microseconds as a double
                double duration_us = std::chrono::duration<double, std::micro>(duration_in_state).count();

                // Get metric after wait and calculate reward delta
                uint64_t performance_metric_after = snap_optimizer_get_performance_metric();
                uint64_t reward_delta = (performance_metric_after >= performance_metric_before) ?
                                            (performance_metric_after - performance_metric_before) : 0; // Handle wrap/error

                // Calculate reward per microsecond, handle division by zero
                if (duration_us > 0.0) {
                    last_calculated_reward = static_cast<double>(reward_delta) / duration_us;
                } else {
                    last_calculated_reward = 0.0; // Avoid division by zero, set reward to 0
                }

                // Update log message
                std::cout << "State: WAITING_FOR_PREDICTION -> PREDICTING (" << DIRTY_WINDOW_MIN_LENGTH.count() << "ms elapsed)"
                          << " (Perf Metric Delta: " << reward_delta << ")"
                          << " (Duration (us): " << duration_us << ")"
                          << " (Reward/us: " << last_calculated_reward << ")"
                          << " (Spent " << std::chrono::duration<double, std::milli>(duration_in_state).count() << "ms in WAITING_FOR_PREDICTION)" << std::endl;
                state_entry_time = now;

                current_op_state = OptState::PREDICTING;
            } else {
                break;
            }
            /* fall through */
        }

        case OptState::PREDICTING: {
            // Use the getter method for dimension
             Eigen::VectorXd current_state = Eigen::VectorXd::Zero(snap_optimizer->get_dim_in());

            TIME_BLOCK({
                current_best_arm = snap_optimizer->best_arm_prediction(current_state);
                // Call helper function to apply scaled params
                apply_scaled_parameters(current_best_arm);
            }, predict_times);

            // Capture end time after timing for consistency
            auto predict_end_time = std::chrono::high_resolution_clock::now();
            cycle_start_time = predict_end_time; // Mark cycle start time

            // Calculate and print time in state, then reset timer
            auto duration_in_state = predict_end_time - state_entry_time;
            std::cout << "State: PREDICTING -> WAITING_FOR_CYCLE. Best arm: " << current_best_arm.transpose()
                      << " (Spent " << std::chrono::duration<double, std::milli>(duration_in_state).count() << "ms in PREDICTING)" << std::endl;
            state_entry_time = predict_end_time;

            current_op_state = OptState::WAITING_FOR_CYCLE;
            break;
        }

        case OptState::WAITING_FOR_CYCLE: {
            auto elapsed = now - cycle_start_time;
            if (elapsed >= CYCLE_TIME) {
                // Calculate and print time in state, then reset timer
                auto duration_in_state = now - state_entry_time;
                std::cout << "State: WAITING_FOR_CYCLE -> ACTING (Cycle time elapsed)"
                          << " (Spent " << std::chrono::duration<double, std::milli>(duration_in_state).count() << "ms in WAITING_FOR_CYCLE)" << std::endl;
                state_entry_time = now;

                current_op_state = OptState::ACTING;
            } else {
                break;
            }
            /* fall through */
        }
    }

    iteration_count++;
    if (iteration_count >= STATS_PRINT_INTERVAL) {
        print_stats();
        iteration_count = 0;
    }

    return 0;
}

int cpp_optimizer_cleanup(void) {
    destroy_optimizer(snap_optimizer);
    return 0;
}
}