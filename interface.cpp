#include "interface.hpp"
#include <snap_optimizer_interface.h>
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "limbo/limbo.hpp"
#include "limbo/tools/macros.hpp"
#include "limbo/tools/random_generator.hpp"

using namespace limbo;
using namespace limbo::tools;

// Class to interface with SNAP system
class SnapRPC {
public:
  SnapRPC() {}
  
  // Set parameters in the SNAP system
  bool set_params(const std::map<std::string, std::string>& params) {
    int poll_size = std::stoi(params.at("poll_size"));
    double poll_ratio = std::stod(params.at("poll_ratio"));
    int max_inflights = std::stoi(params.at("max_inflights"));
    int max_iog_batch = std::stoi(params.at("max_iog_batch"));
    int max_new_ios = std::stoi(params.at("max_new_ios"));
    
    return snap_optimizer_set_system_params(poll_size, poll_ratio, 
                                          max_inflights, max_iog_batch, 
                                          max_new_ios) == 0;
  }
  
  // Get performance metric from SNAP system
  double get_reward() {
    uint64_t metric = snap_optimizer_get_performance_metric();
    return static_cast<double>(metric);
  }
};

// Structure to hold SNAP parameters
struct SnapParams {
  int poll_size;
  double poll_ratio;
  int max_inflights;
  int max_iog_batch;
  int max_new_ios;

  // Helper function to convert from normalized 0-1 to exact power of 2
  static int from_power2_space(double normalized_value, int min_power,
                               int max_power) {
    // Convert normalized value to power of 2 exponent
    double power = min_power + normalized_value * (max_power - min_power);
    return 1 << static_cast<int>(std::round(power));
  }

  std::map<std::string, std::string> to_rpc_params() const {
    std::map<std::string, std::string> params;
    params["poll_size"] = std::to_string(poll_size);
    params["poll_ratio"] = std::to_string(poll_ratio);
    params["max_inflights"] = std::to_string(max_inflights);
    params["max_iog_batch"] = std::to_string(max_iog_batch);
    params["max_new_ios"] = std::to_string(max_new_ios);
    return params;
  }

  static SnapParams from_optimization_space(const Eigen::VectorXd &x) {
    SnapParams params;

    // poll_size: 1-256 (power of 2)
    params.poll_size = from_power2_space(x(0), 0, 8); // 2^0 to 2^8 = 1 to 256

    // poll_ratio: 0-1 (linear scale)
    params.poll_ratio = x(1);

    // max_inflights: 1-65535 (power of 2)
    params.max_inflights =
        from_power2_space(x(2), 0, 16); // 2^0 to 2^16 = 1 to 65536

    // max_iog_batch: 1-4096 (power of 2)
    params.max_iog_batch =
        from_power2_space(x(3), 0, 12); // 2^0 to 2^12 = 1 to 4096

    // max_new_ios: 1-4096 (power of 2)
    params.max_new_ios =
        from_power2_space(x(4), 0, 12); // 2^0 to 2^12 = 1 to 4096

    return params;
  }
};

// Parameters matching Python's SNAP environment
struct Params {
  struct bayes_opt_bobase : public limbo::defaults::bayes_opt_bobase {
    BO_PARAM(int, stats_enabled, true);
    BO_PARAM(bool, bounded, true);
  };

  struct bayes_opt_boptimizer : public limbo::defaults::bayes_opt_boptimizer {
    BO_PARAM(int, hp_period, -1); // -1 means no hyperparameter optimization
  };

  struct opt_nloptnograd : public limbo::defaults::opt_nloptnograd {
    BO_PARAM(int, iterations, 500);
    BO_PARAM(double, learning_rate, 0.01);
    BO_PARAM(double, tolerance, 1e-6);
  };

  struct kernel : public limbo::defaults::kernel {
    BO_PARAM(double, noise, 1e-10);
  };

  struct kernel_maternfivehalves : public limbo::defaults::kernel_maternfivehalves {
    BO_PARAM(double, sigma_sq, 1);
    BO_PARAM(double, l, 1);
  };

  struct init_randomsampling {
    BO_PARAM(int, samples, 10); // Matching Python's num_sobol_trials
  };

  struct stop_maxiterations {
    BO_PARAM(int, iterations, 50);
  };

  struct acqui_ucb : public limbo::defaults::acqui_ucb {
    BO_PARAM(double, alpha, 1.96);
  };
};

// Evaluation function that uses SNAP RPC
struct Eval {
  BO_PARAM(size_t, dim_in, 5);  // 5 SNAP parameters
  BO_PARAM(size_t, dim_out, 1); // Single objective (IOPS)

  SnapRPC &rpc;
  int num_samples = 1;

  Eval(SnapRPC &rpc_client) : rpc(rpc_client) {}

  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
    // Convert optimization parameters to SNAP parameters
    SnapParams params = SnapParams::from_optimization_space(x);

    // Take multiple samples
    std::vector<double> scores;
    for (int i = 0; i < num_samples; ++i) {
      // Set parameters
      rpc.set_params(params.to_rpc_params());

      // Get reward
      double score = rpc.get_reward();
      scores.push_back(score);
    }

    // Calculate median
    std::sort(scores.begin(), scores.end());
    double final_score = scores[scores.size() / 2];

    return Eigen::VectorXd::Constant(1, final_score);
  }
};

// Global variables
static SnapRPC* g_rpc = nullptr;
static bayes_opt::BOptimizer<Params>* g_optimizer = nullptr;
static Eigen::VectorXd g_best_params;
static bool g_initialized = false;

// Limbo optimizer initialization
extern "C" int cpp_optimizer_init(void) {
  try {
    if (g_initialized)
      return 0;
      
    g_rpc = new SnapRPC();
    g_optimizer = new bayes_opt::BOptimizer<Params>();
    g_initialized = true;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing limbo optimizer: " << e.what() << std::endl;
    return -1;
  }
}

// Limbo optimizer iteration
extern "C" int cpp_optimizer_iteration(void) {
  if (!g_rpc || !g_optimizer) {
    return -1;
  }
  
  try {
    // Get initial reward
    [[maybe_unused]] double initial_reward = g_rpc->get_reward();
    
    // Wait 50ms
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Get reward again
    [[maybe_unused]] double final_reward = g_rpc->get_reward();
    
    // Configure the evaluation function
    Eval eval(*g_rpc);
    
    // Perform optimization using Limbo
    g_optimizer->optimize(eval);
    
    // Get the best parameters found by the optimizer
    g_best_params = g_optimizer->best_sample();
    
    // Convert and set parameters
    SnapParams params = SnapParams::from_optimization_space(g_best_params);
    g_rpc->set_params(params.to_rpc_params());
    
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error in limbo optimizer iteration: " << e.what() << std::endl;
    return -1;
  }
}

// Limbo optimizer cleanup
extern "C" int cpp_optimizer_cleanup(void) {
  try {
    if (g_optimizer) {
      delete g_optimizer;
      g_optimizer = nullptr;
    }
    
    if (g_rpc) {
      delete g_rpc;
      g_rpc = nullptr;
    }
    
    g_initialized = false;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error cleaning up limbo optimizer: " << e.what() << std::endl;
    return -1;
  }
}
