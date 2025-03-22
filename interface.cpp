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
  SnapRPC() : last_completions(0), last_timestamp(0) {}
  
  // Set parameters in the SNAP system
  bool set_params(const std::map<std::string, std::string>& params) {
    auto start = std::chrono::high_resolution_clock::now();
    int poll_size = std::stoi(params.at("poll_size"));
    double poll_ratio = std::stod(params.at("poll_ratio"));
    int max_inflights = std::stoi(params.at("max_inflights"));
    int max_iog_batch = std::stoi(params.at("max_iog_batch"));
    int max_new_ios = std::stoi(params.at("max_new_ios"));
    
    bool result = snap_optimizer_set_system_params(poll_size, poll_ratio, 
                                          max_inflights, max_iog_batch, 
                                          max_new_ios) == 0;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "set_params took: " << duration.count() << " microseconds\n";
    return result;
  }
  
  // Get performance metric from SNAP system
  double get_reward() {
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t completions = snap_optimizer_get_performance_metric();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "get_performance_metric took: " << duration.count() << " microseconds\n";

    // Calculate IOPS based on difference from last measurement
    if (last_timestamp == 0) {
      last_completions = completions;
      last_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      return 0.0;  // First measurement, no IOPS yet
    }

    auto current_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    double time_diff = (current_timestamp - last_timestamp) / 1000000.0;  // Convert microseconds to seconds
    double completions_diff = completions - last_completions;
    
    last_completions = completions;
    last_timestamp = current_timestamp;

    double iops = completions_diff / time_diff;
    std::cout << "IOPS calculation: " << completions_diff << " completions over " 
              << time_diff << " seconds = " << iops << " IOPS\n";
    return iops;
  }

private:
  uint64_t last_completions;
  uint64_t last_timestamp;
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
    params.poll_size = from_power2_space(x[0], 0, 8); // 2^0 to 2^8 = 1 to 256

    // poll_ratio: 0-1 (linear scale)
    params.poll_ratio = x[1];

    // max_inflights: 1-65535 (power of 2)
    params.max_inflights =
        from_power2_space(x[2], 0, 16); // 2^0 to 2^16 = 1 to 65536

    // max_iog_batch: 1-4096 (power of 2)
    params.max_iog_batch =
        from_power2_space(x(3), 0, 12); // 2^0 to 2^12 = 1 to 4096

    // max_new_ios: 1-4096 (power of 2)
    params.max_new_ios =
        from_power2_space(x(4), 0, 12); // 2^0 to 2^12 = 1 to 4096

    return params;
  }
};

struct Params {
  struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
    BO_PARAM(int, stats_enabled, false);
    BO_PARAM(bool, bounded, true);
  };

  struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    BO_PARAM(int, hp_period,
             1); // Currently set to 1 (hp opt every iteration). In future a
                 // custom 'skip' logic should be implemented
  };

  struct opt_nloptnograd : public defaults::opt_nloptnograd {
    BO_PARAM(int, iterations, 50);
  };

  struct kernel : public defaults::kernel {
    BO_PARAM(double, noise, 0.01);
    BO_PARAM(bool, optimize_noise, true);
  };

  struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    BO_PARAM(double, sigma_sq, 1);
    BO_PARAM(int, k, 0);
  };

  struct init_randomsampling {
    BO_PARAM(int, samples, 20);
  };

  struct stop_maxiterations {
    BO_PARAM(int, iterations, 1);  // Keep at 1 iteration per optimize() call
  };

  struct acqui_ucb : public defaults::acqui_ucb {
    BO_PARAM(double, alpha, 0.5);
  };

  struct opt_rprop : public defaults::opt_rprop {};

  struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    BO_PARAM(int, repeats, 10);
    BO_PARAM(double, epsilon, 1);
  };
  struct mean_constant : public defaults::mean_constant {
    BO_PARAM(double, constant, 0);
  };
};

// Custom GP model that uses only the last N observations
template <typename Params, typename KernelFunction, typename MeanFunction, typename HyperParamsOptimizer>
class WindowedGP : public model::GP<Params, KernelFunction, MeanFunction, HyperParamsOptimizer> {
public:
    using base_t = model::GP<Params, KernelFunction, MeanFunction, HyperParamsOptimizer>;
    
    WindowedGP()
        : base_t() {}
    WindowedGP(int dim_in, int dim_out)
        : base_t(dim_in, dim_out) {}
        
    void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations) {
        static constexpr int N = 20;
        
        std::cout << "\n=== WindowedGP::compute Debug ===\n";
        std::cout << "Total observations: " << samples.size() << "\n";
        
        // If we have fewer than N observations, use all of them
        if (samples.size() <= N) {
            std::cout << "Using all observations (less than window size)\n";
            base_t::compute(samples, observations);
            return;
        }
        
        // Otherwise, use only the last N observations
        std::cout << "Using only last " << N << " observations\n";
        std::vector<Eigen::VectorXd> recent_samples(samples.end() - N, samples.end());
        std::vector<Eigen::VectorXd> recent_observations(observations.end() - N, observations.end());
        
        base_t::compute(recent_samples, recent_observations);
    }
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

    double score = rpc.get_reward();
    std::cout << "Final score (median): " << score << "\n";

    return Eigen::VectorXd::Constant(1, score);
  }
};
struct DummyEval {
  // number of input dimension (x.size())
  BO_PARAM(size_t, dim_in, 3);
  // number of dimensions of the result (res.size())
  BO_PARAM(size_t, dim_out, 1);

  // the function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
    Eigen::VectorXd vec(3);
    vec << 0.5, 0.5, 0.5;
    double y = -(x - vec).norm();
    // Add Gaussian noise to y
    double noise_mean = 0.0;
    double noise_stddev = 0.1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(noise_mean, noise_stddev);
    y += d(gen);
    // we return a 1-dimensional vector
    return tools::make_vector(y);
  }
};

// Global variables
static SnapRPC* g_rpc = nullptr;

static Eigen::VectorXd g_best_params;
static bool g_initialized = false;

using kernel_t = kernel::SquaredExpARD<Params>;
using mean_t = mean::Constant<Params>;
using gp_opt_t = model::gp::KernelMeanLFOpt<Params>;
using gp_t = model::GP<Params, kernel_t, mean_t, gp_opt_t>;
using acqui_t = acqui::UCB<Params, gp_t>;
using acqui_opt_t = opt::NLOptNoGrad<Params>;
using stat_t = boost::fusion::vector<
    stat::ConsoleSummary<Params>, stat::Samples<Params>,
    stat::Observations<Params>, stat::GPAcquisitions<Params>,
    stat::BestAggregatedObservations<Params>, stat::GPKernelHParams<Params>,
    stat::GPPredictionDifferences<Params>>;
using init_t = init::RandomSampling<Params>;

// Use the windowed GP model in the optimizer instantiation
using windowed_gp_t = WindowedGP<Params, kernel_t, mean_t, gp_opt_t>;
using windowed_acqui_t = acqui::UCB<Params, windowed_gp_t>;

// Update g_optimizer definition to use the windowed model
static bayes_opt::BOptimizer<Params, modelfun<windowed_gp_t>, acquifun<windowed_acqui_t>,
                             acquiopt<acqui_opt_t>, initfun<init_t>,
                             statsfun<stat_t>> *g_optimizer = nullptr;

// Limbo optimizer initialization
extern "C" int cpp_optimizer_init(void) {
  try {
    if (g_initialized)
      return 0;
      
    g_rpc = new SnapRPC();
    g_optimizer = new bayes_opt::BOptimizer<Params, modelfun<windowed_gp_t>, acquifun<windowed_acqui_t>, acquiopt<acqui_opt_t>, initfun<init_t>, statsfun<stat_t>>();
    g_initialized = true;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error initializing limbo optimizer: " << e.what() << "\n";
    return -1;
  }
}

// Limbo optimizer iteration
extern "C" int cpp_optimizer_iteration(void) {
  if (!g_rpc || !g_optimizer) {
    return -1;
  }
  
  try {
    auto iteration_start = std::chrono::high_resolution_clock::now();
    
    Eval eval(*g_rpc);
    #ifdef USE_DUMMY
        std::cout << "Optimizing Dummy function" << std::endl;
        optimizer.optimize(DummyEval());
    #else
        std::cout << "Optimizing SNAP" << std::endl;
        // The aggregator is already specified in the template parameters of g_optimizer
        g_optimizer->optimize(eval, FirstElem(), false);
    #endif
    
    auto optimization_end = std::chrono::high_resolution_clock::now();
    auto optimization_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        optimization_end - iteration_start);
    std::cout << "Optimization took: " << optimization_duration.count() / 1000.0 << " milliseconds\n";

    g_best_params = g_optimizer->best_sample();  
    SnapParams params = SnapParams::from_optimization_space(g_best_params);
    
    auto params_start = std::chrono::high_resolution_clock::now();
    g_rpc->set_params(params.to_rpc_params());
    auto params_end = std::chrono::high_resolution_clock::now();
    auto params_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        params_end - params_start);
    std::cout << "Setting parameters took: " << params_duration.count() << " microseconds\n";

    auto iteration_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        iteration_end - iteration_start);
    std::cout << "Total iteration time: " << total_duration.count() / 1000.0 << " milliseconds\n";
    
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error in limbo optimizer iteration: " << e.what() << "\n";
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
    std::cerr << "Error cleaning up limbo optimizer: " << e.what() << "\n";
    return -1;
  }
}
