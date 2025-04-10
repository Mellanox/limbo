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
#include <sstream>
#include <string>
#include <functional>
#include <map>
#include <memory>
#include <typeinfo>

#define TIME_BLOCK(CODE_TO_RUN, DURATION_VECTOR) \
    do { \
        auto _macro_start_time = std::chrono::high_resolution_clock::now(); \
        { \
            CODE_TO_RUN; \
        } \
        auto _macro_end_time = std::chrono::high_resolution_clock::now(); \
        DURATION_VECTOR.push_back(_macro_end_time - _macro_start_time); \
    } while (0)

#define STATS_PRINT_INTERVAL 1000
#define INITIAL_SAMPLE_LIMIT (Params::init_randomsampling::samples())

struct CycleTimings {
    std::chrono::duration<double> random_phase_cycle{0.5};
    std::chrono::seconds optimization_phase_cycle{12};
    std::chrono::seconds hyperparam_wait{1};
    std::chrono::milliseconds reward_calc_wait{50};
};
static CycleTimings cycle_timings;
static SnapStateBOptimizer* snap_optimizer = nullptr;

enum class OptState {
    ACTING,
    REWARD_CALCULATION_BO,
    RESTORE_BEST_ARM,
    UPDATING,
    OPTIMIZE_HYPERPARAMS
};

static OptState current_op_state;
static Eigen::VectorXd last_sample;
static std::chrono::high_resolution_clock::time_point wait_start_time;
static Eigen::VectorXd current_best_arm;

static std::map<OptState, std::vector<std::chrono::high_resolution_clock::duration>> state_timings;

static int iteration_count = 0;
static uint64_t performance_metric_before = 0;
static double last_calculated_reward = 0.0;

static double temp_duration_us = 0.0;
static uint64_t temp_reward_delta = 0;

struct StateInfo {
    OptState state_enum;
    OptState next_state;
    std::string name;
    std::function<std::chrono::high_resolution_clock::duration()> get_wait_duration;
    std::function<void()> action;
    std::function<std::string()> get_log_message;
};

static std::map<OptState, StateInfo> state_machine_config;

static int from_power2_space(double normalized_value, int min_power, int max_power) {
    double power = min_power + normalized_value * (max_power - min_power);
    return 1 << static_cast<int>(std::round(power));
}

static void apply_scaled_parameters(const Eigen::VectorXd& params_vector) {
    int poll_size = 1, max_inflights = 1, max_iog_batch = 1, max_new_ios = 1;
    double poll_ratio = 0.5;

    if (params_vector.size() >= 5) {
        poll_size     = from_power2_space(params_vector[0], 0, 8);
        poll_ratio    = params_vector[1];
        max_inflights = from_power2_space(params_vector[2], 0, 16);
        max_iog_batch = from_power2_space(params_vector[3], 0, 12);
        max_new_ios   = from_power2_space(params_vector[4], 0, 12);

        poll_size = std::max(1, std::min(poll_size, 256));
        poll_ratio = std::max(0.0, std::min(poll_ratio, 1.0));
        max_inflights = std::max(1, std::min(max_inflights, 65535));
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
    auto calculate_stats = [](const std::vector<std::chrono::high_resolution_clock::duration>& times) {
        using Duration = std::chrono::high_resolution_clock::duration;
        if (times.empty()) {
            return std::make_tuple(Duration::zero(), Duration::zero(), Duration::zero());
        }
        Duration sum = std::accumulate(times.begin(), times.end(), Duration::zero());
        Duration mean = sum / times.size();
        Duration min_val = *std::min_element(times.begin(), times.end());
        Duration max_val = *std::max_element(times.begin(), times.end());
        return std::make_tuple(min_val, max_val, mean);
    };

    auto to_ms = [](std::chrono::high_resolution_clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "--- Aggregated Stats (last " << STATS_PRINT_INTERVAL << " cycles) --- \n";
    std::cout << "--- Op Timings (Core Logic Duration) --- \n";

    for(const auto& pair : state_machine_config) {
        OptState state = pair.first;
        if (state_timings.count(state) && !state_timings[state].empty()) {
             std::chrono::high_resolution_clock::duration min_d, max_d, avg_d;
             std::tie(min_d, max_d, avg_d) = calculate_stats(state_timings[state]);
             std::cout << pair.second.name << " (ms): min=" << to_ms(min_d)
                       << ", max=" << to_ms(max_d) << ", avg=" << to_ms(avg_d)
                       << " (" << state_timings[state].size() << " calls)\n";
        }
    }

    std::cout << "--------------------------------------------\n";

    for(auto& pair : state_timings) {
        pair.second.clear();
    }

    performance_metric_before = snap_optimizer_get_performance_metric();
    return;
}

void* snap_optimizer_factory(int dim_in, int dim_out) {
    return new SnapStateBOptimizer(dim_in, dim_out);
}

static void transition_to(OptState next_state) {
    current_op_state = next_state;
}

template <typename DurationType>
static bool check_wait_time(const DurationType& required_wait) {
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = now - wait_start_time;
    return elapsed >= required_wait;
}

extern "C" {
int cpp_optimizer_init(void) {
    void* handle = create_optimizer(&snap_optimizer_factory, 5, 1);
    snap_optimizer = static_cast<SnapStateBOptimizer*>(handle);

    if (!snap_optimizer) {
        std::cerr << "Error: Failed to create optimizer instance." << std::endl;
        return -1;
    }

    state_timings[OptState::ACTING] = {};
    state_timings[OptState::REWARD_CALCULATION_BO] = {};
    state_timings[OptState::RESTORE_BEST_ARM] = {};
    state_timings[OptState::UPDATING] = {};
    state_timings[OptState::OPTIMIZE_HYPERPARAMS] = {};

    state_machine_config.emplace(OptState::ACTING, StateInfo{
        /*state_enum=*/ OptState::ACTING,
        /*next_state=*/ OptState::REWARD_CALCULATION_BO,
        /*name=*/ "Act",
        /*get_wait_duration=*/ []() {
            bool is_random_phase = (snap_optimizer->current_iteration() < INITIAL_SAMPLE_LIMIT);
            if (is_random_phase) {
                return std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(cycle_timings.random_phase_cycle);
            } else {
                return std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(cycle_timings.optimization_phase_cycle);
            }
         },
        /*action=*/ []() {
            Eigen::VectorXd current_state = Eigen::VectorXd::Zero(snap_optimizer->get_dim_in());
            last_sample = snap_optimizer->act(current_state);
            apply_scaled_parameters(last_sample);
            performance_metric_before = snap_optimizer_get_performance_metric();
        },
        /*get_log_message=*/ []() {
             std::stringstream ss;
             ss << ". Sample: " << last_sample.transpose();
             return ss.str();
         }
    });

    state_machine_config.emplace(OptState::REWARD_CALCULATION_BO, StateInfo{
        /*state_enum=*/ OptState::REWARD_CALCULATION_BO,
        /*next_state=*/ OptState::RESTORE_BEST_ARM,
        /*name=*/ "WaitPrd",
        /*get_wait_duration=*/ []() { return cycle_timings.reward_calc_wait; },
        /*action=*/ []() {
             auto now = std::chrono::high_resolution_clock::now();
             auto duration_since_wait_start = now - wait_start_time;
             temp_duration_us = std::chrono::duration<double, std::micro>(duration_since_wait_start).count();
             uint64_t performance_metric_after = snap_optimizer_get_performance_metric();
             temp_reward_delta = (performance_metric_after >= performance_metric_before) ?
                                         (performance_metric_after - performance_metric_before) : 0;
             last_calculated_reward = (temp_duration_us > 0.0) ? static_cast<double>(temp_reward_delta) / temp_duration_us : 0.0;

        },
        /*get_log_message=*/ []() {
             std::stringstream ss;
             ss << "(Wait Window: " << cycle_timings.reward_calc_wait.count() << "ms) "
                << "(Perf Metric Delta: " << temp_reward_delta << ") "
                << "(Actual Wait (us): " << std::fixed << std::setprecision(1) << temp_duration_us << ") "
                << "(Reward/us: " << std::fixed << std::setprecision(5) << last_calculated_reward << ")";
             return ss.str();
         }
    });
    state_machine_config.emplace(OptState::RESTORE_BEST_ARM, StateInfo{
        /*state_enum=*/ OptState::RESTORE_BEST_ARM,
        /*next_state=*/ OptState::UPDATING,
        /*name=*/ "Restore",
        /*get_wait_duration=*/ []() { return std::chrono::high_resolution_clock::duration::zero(); },
        /*action=*/ []() {
            if (!(snap_optimizer->current_iteration() > 0)) {
                return;
            }
            Eigen::VectorXd current_state = Eigen::VectorXd::Zero(snap_optimizer->get_dim_in());
            current_best_arm = snap_optimizer->best_arm_prediction(current_state);
            apply_scaled_parameters(current_best_arm);
        },
        /*get_log_message=*/ []() {
             std::stringstream ss;
             ss << ". Best arm: " << current_best_arm.transpose()
                << " (Using " << (snap_optimizer->current_iteration() >= INITIAL_SAMPLE_LIMIT ? "prediction" : "random/initial model") << ")";
             return ss.str();
         }
    });
    state_machine_config.emplace(OptState::UPDATING, StateInfo{
        /*state_enum=*/ OptState::UPDATING,
        /*next_state=*/ OptState::OPTIMIZE_HYPERPARAMS,
        /*name=*/ "Update",
        /*get_wait_duration=*/ []() { return std::chrono::high_resolution_clock::duration::zero(); },
        /*action=*/ []() {
            Eigen::VectorXd observation = Eigen::VectorXd::Constant(1, last_calculated_reward);

            if (last_calculated_reward == 0.0 && snap_optimizer->current_iteration() > 0) {
                std::cout << "(Skipping optimizer update and set_params due to zero reward)" << std::endl;
            } else {
                snap_optimizer->update(last_sample, observation);
            }
        },
        /*get_log_message=*/ []() {
            Eigen::VectorXd observation = Eigen::VectorXd::Constant(1, last_calculated_reward);
            std::stringstream ss;
            ss << ". Observation: " << observation.transpose() << " (Reward: " << last_calculated_reward << ")";
            return ss.str();
        }
    });


    state_machine_config.emplace(OptState::OPTIMIZE_HYPERPARAMS, StateInfo{
        /*state_enum=*/ OptState::OPTIMIZE_HYPERPARAMS,
        /*next_state=*/ OptState::ACTING,
        /*name=*/ "WaitHyp",
        /*get_wait_duration=*/ []() { return cycle_timings.hyperparam_wait; },
        /*action=*/ []() {
            snap_optimizer->trigger_hyperparams_optimization();
        },
        /*get_log_message=*/ []() { return "(Triggered hyperparams)"; }
    });

    current_op_state = OptState::ACTING;
    iteration_count = 0;
    performance_metric_before = snap_optimizer_get_performance_metric();
    return 0;
}

int cpp_optimizer_iteration(void) {
    if (!snap_optimizer || state_machine_config.empty()) {
        std::cerr << "Error: Optimizer or state machine not initialized." << std::endl;
        return -1;
    }

    auto& current_state_info = state_machine_config.at(current_op_state);

    auto required_wait = current_state_info.get_wait_duration();
    bool is_wait_over = (required_wait == std::chrono::high_resolution_clock::duration::zero()) || check_wait_time(required_wait);

    if (is_wait_over) {
        TIME_BLOCK({
            current_state_info.action();
        }, state_timings[current_op_state]);

        OptState next_state_enum = current_state_info.next_state;
        auto& next_state_info = state_machine_config.at(next_state_enum);
        std::cout << "State Transition: " << current_state_info.name << " -> " << next_state_info.name
                  << current_state_info.get_log_message() << std::endl;

        transition_to(next_state_enum);

        wait_start_time = std::chrono::high_resolution_clock::now();
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
    state_machine_config.clear();
    state_timings.clear();
    return 0;
}
}