#include "interface.hpp"

// Standard C++ Headers
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

// C Interface Header
#include "snap_optimizer/snap_optimizer_interface.h"

// System Headers (Potentially problematic)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

// Eigen Header
#include <Eigen/Core>

// Limbo Headers
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
  bool set_params(const std::map<std::string, std::string> &params) {
    auto start = std::chrono::high_resolution_clock::now();
    int poll_size = std::stoi(params.at("poll_size"));
    double poll_ratio = std::stod(params.at("poll_ratio"));
    int max_inflights = std::stoi(params.at("max_inflights"));
    int max_iog_batch = std::stoi(params.at("max_iog_batch"));
    int max_new_ios = std::stoi(params.at("max_new_ios"));

    bool result =
        snap_optimizer_set_system_params(poll_size, poll_ratio, max_inflights,
                                         max_iog_batch, max_new_ios) == 0;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "set_params took: " << duration.count() << " microseconds\n";
    return result;
  }

  // Get performance metric from SNAP system
  double get_reward() {
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t completions = snap_optimizer_get_performance_metric();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "get_performance_metric took: " << duration.count()
              << " microseconds\n";

    // Calculate IOPS based on difference from last measurement
    if (last_timestamp == 0) {
      last_completions = completions;
      last_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
      return 0.0; // First measurement, no IOPS yet
    }

    auto current_timestamp =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    double time_diff = (current_timestamp - last_timestamp) /
                       1000000.0; // Convert microseconds to seconds
    double completions_diff = completions - last_completions;

    last_completions = completions;
    last_timestamp = current_timestamp;

    double iops = completions_diff / time_diff;
    std::cout << "IOPS calculation: " << completions_diff
              << " completions over " << time_diff << " seconds = " << iops
              << " IOPS\n";
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
    BO_PARAM(int, iterations, 500);
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
    BO_PARAM(int, iterations, 50); // Keep at 1 iteration per optimize() call
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
// Evaluation function that uses SNAP RPC
struct Eval {
  BO_PARAM(size_t, dim_in, 5);  // 5 SNAP parameters
  BO_PARAM(size_t, dim_out, 1); // Single objective (IOPS)
};
struct RPCEval {
  BO_PARAM(size_t, dim_in, 5);  // 5 SNAP parameters
  BO_PARAM(size_t, dim_out, 1); // Single objective (IOPS)

  SnapRPC &rpc;
  int num_samples = 1;

  RPCEval(SnapRPC &rpc_client) : rpc(rpc_client) {}

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

// New evaluation function that decrements a counter
struct DecrementingEval {
  BO_PARAM(size_t, dim_in, 5);  // Same input dimension as Eval
  BO_PARAM(size_t, dim_out, 1); // Same output dimension as Eval

private:
  // Static counter to maintain state across calls if multiple instances are
  // created
  constexpr static double MAX_VALUE = 100000.0;
  constexpr static int REDUCTION_RATE = 100;
  static int iteration;

public:
  // The function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
    // DEBUG: Print the input x received from the optimizer
    std::cout << "  DEBUG (DecrementingEval): Received x = " << x.transpose()
              << std::endl;

    double calculated_value =
        MAX_VALUE -
        (++const_cast<DecrementingEval *>(this)->iteration) * REDUCTION_RATE;
    std::cout << "  DEBUG (DecrementingEval): Returning value = "
              << calculated_value << std::endl;
    return Eigen::VectorXd::Constant(1, calculated_value);
  }
};

// Initialize static member
int DecrementingEval::iteration = 0;

// New evaluation function that sums the input vector elements
struct SumEval {
  BO_PARAM(size_t, dim_in, 5);  // Expects 5 input dimensions
  BO_PARAM(size_t, dim_out, 1); // Returns 1 output dimension

public:
  // The function calculates the sum of the input vector x
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
    // Ensure the input vector has the expected dimension
    if (x.size() != dim_in()) {
      throw std::invalid_argument("SumEval: Input vector dimension mismatch.");
    }

    // Calculate the sum of the elements
    double sum = x.sum();

    std::cout << "  DEBUG (SumEval): Received x = " << x.transpose()
              << std::endl;
    std::cout << "  DEBUG (SumEval): Returning sum = " << sum << std::endl;

    // Return the sum as a 1-dimensional vector
    return tools::make_vector(sum);
  }
};

// Global variables
static SnapRPC *g_rpc = nullptr;

static Eigen::VectorXd g_best_params;
static bool g_initialized = false;
static std::future<void>
    g_opt_future; // Global future for the optimization task

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

// Define the new custom optimizer class inheriting from BoBase
template <
    class Params, class A1 = boost::parameter::void_,
    class A2 = boost::parameter::void_, class A3 = boost::parameter::void_,
    class A4 = boost::parameter::void_, class A5 = boost::parameter::void_,
    class A6 = boost::parameter::void_>
class StateBOptimizer
    : public limbo::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6> {
public:
  // Remove the constructor from the abstract base class
  /* StateBOptimizer()
      : limbo::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6>() {
    // This initialization was problematic here
    // this->_init(sfun, afun, true);
    // _model = model_t(StateFunction::dim_in(), StateFunction::dim_out());
  }*/
  virtual ~StateBOptimizer() = default; // Add virtual destructor

  // Pure virtual functions remain the same
  virtual Eigen::VectorXd act(Eigen::VectorXd state) = 0;
  virtual void update(Eigen::VectorXd sample, Eigen::VectorXd observation) = 0;
  virtual Eigen::VectorXd best_arm_prediction(Eigen::VectorXd state) = 0;
  virtual Eigen::VectorXd best_bo_prediction(Eigen::VectorXd state) = 0;

protected:
  virtual Eigen::VectorXd get_state() = 0;
  virtual Eigen::VectorXd get_state_samples() = 0;
};

class SnapStateBOptimizer
    : public StateBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>,
                             acquiopt<acqui_opt_t>, initfun<init_t>,
                             statsfun<stat_t>> {
public:
  // Constructor to initialize the base BoBase and the model
  SnapStateBOptimizer()
      : StateBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>,
                        acquiopt<acqui_opt_t>, initfun<init_t>,
                        statsfun<stat_t>>() {
    // Initialize model here, assuming Params defines model_t
    // And DecrementingEval defines dimensions
    // Note: _init is protected in BoBase, often called internally
    // We might not need to call _init explicitly if base constructor handles it
    this->_model =
        model_t(DecrementingEval::dim_in(), DecrementingEval::dim_out());
    // TODO: Verify if _init needs to be called or if base constructor is
    // sufficient Example: If _init(sfun, afun, stats_enabled) is needed:
    // this->_init(this->_eval, acquisition_function(),
    // Params::stats_enabled()); where acquisition_function() returns an
    // instance of acqui_t and this->_eval handles the evaluation logic based on
    // get_state/act
  }

  // Override virtual functions
  Eigen::VectorXd act(Eigen::VectorXd state) override {
    std::cout << "DEBUG: SnapStateBOptimizer::act called with state: "
              << state.transpose() << std::endl;
    // Placeholder implementation
    return Eigen::VectorXd::Zero(5); // Assuming action space dim is 5
  }

  void update(Eigen::VectorXd sample, Eigen::VectorXd observation) override {
    std::cout << "DEBUG: SnapStateBOptimizer::update called with sample: "
              << sample.transpose()
              << " and observation: " << observation.transpose() << std::endl;
  }

  Eigen::VectorXd best_arm_prediction(Eigen::VectorXd state) override {
    std::cout
        << "DEBUG: SnapStateBOptimizer::best_arm_prediction called with state: "
        << state.transpose() << std::endl;
    // Placeholder implementation
    return Eigen::VectorXd::Zero(5); // Assuming action space dim is 5
  }

  Eigen::VectorXd best_bo_prediction(Eigen::VectorXd state) override {
    std::cout
        << "DEBUG: SnapStateBOptimizer::best_bo_prediction called with state: "
        << state.transpose() << std::endl;
    // Placeholder - Implement actual BO prediction logic
    // This might involve optimizing the acquisition function based on the
    // current model For example: acqui_optimizer_t acqui_optimizer; auto
    // acqui_func = [&](const Eigen::VectorXd& x, bool g) {
    //     return acquisition_function()(x, g); // Assuming
    //     acquisition_function() provides acqui_t instance
    // };
    // Eigen::VectorXd starting_point =
    // tools::random_vector(StateFunction::dim_in(),
    // Params::bayes_opt_bobase::bounded()); Eigen::VectorXd next_sample =
    // acqui_optimizer(acqui_func, starting_point,
    // Params::bayes_opt_bobase::bounded()); return next_sample;

    return Eigen::VectorXd::Zero(5); // Placeholder return
  }

protected:
  Eigen::VectorXd get_state() override {
    struct snap_observations obs;
    snap_optimizer_get_observations(&obs, 0);

    Eigen::VectorXd state_vector(
        DecrementingEval::dim_in()); // Use StateFunction for dim
    state_vector(0) = static_cast<double>(obs.num_active_queues);
    state_vector(1) = static_cast<double>(obs.num_queues);
    state_vector(2) = static_cast<double>(obs.max_qdepth);
    state_vector(3) = static_cast<double>(obs.avg_qdepth);

    std::cout << "DEBUG: SnapStateBOptimizer::get_state returning: "
              << state_vector.transpose() << std::endl;

    return state_vector;
  }

  // Correct signature - no state argument
  Eigen::VectorXd get_state_samples() override {
    std::cout << "DEBUG: SnapStateBOptimizer::get_state_samples called"
              << std::endl;
    // Assumes get_samples() is available from a base class (like BoBase)
    // Need to ensure BoBase actually provides get_samples()
    // If BoBase stores samples in _samples, return that directly:
    // Combine samples into a single Eigen matrix or vector if needed
    if (this->_samples.empty()) {
      return Eigen::VectorXd(); // Return empty vector if no samples
    }
    // This concatenates all samples horizontally. Adjust if different format
    // needed.
    Eigen::MatrixXd samples_mat(this->_samples[0].size(),
                                this->_samples.size());
    for (size_t i = 0; i < this->_samples.size(); ++i) {
      samples_mat.col(i) = this->_samples[i];
    }
    // Returning samples might need clarification - returning the matrix for now
    // Or maybe just the last sample? Returning the whole matrix as a flattened
    // vector:
    return samples_mat.reshaped(); // Flatten the matrix
    // return this->_samples.back(); // Alternative: return only the last
    // sample?
  }
  model_t _model;
};
static SnapStateBOptimizer *g_optimizer = nullptr;
extern "C" int cpp_optimizer_init(void) { return 0; }
extern "C" int cpp_optimizer_iteration(void) { return 0; }
extern "C" int cpp_optimizer_cleanup(void) { return 0; }
// Limbo optimizer initialization

// Helper function to convert C array to Eigen Vector
Eigen::VectorXd to_eigen_vector(const double *data, int size) {
  return Eigen::Map<const Eigen::VectorXd>(data, size);
}

// Helper function to convert Eigen Vector to CVector (allocates memory)
CVector to_cvector(const Eigen::VectorXd &vec) {
  CVector c_vec;
  c_vec.size = vec.size();
  c_vec.data = new double[c_vec.size];
  Eigen::Map<Eigen::VectorXd>(c_vec.data, c_vec.size) = vec;
  return c_vec;
}

extern "C" {

// Assuming SnapStateBOptimizer is derived from StateBOptimizer<Params>
// If not, replace SnapStateBOptimizer with the actual class name.
// Also assuming Params is defined appropriately in the scope.
// If StateBOptimizer is the base class to be instantiated, adjust
// accordingly.

void *create_optimizer() {
  try {
    // IMPORTANT: Replace SnapStateBOptimizer if it's not the concrete class
    // you intend to instantiate. Maybe you want StateBOptimizer directly?
    // Or maybe the derived class needs specific constructor arguments?
    auto *optimizer = new SnapStateBOptimizer();
    // Perform any necessary initialization here if not done in constructor
    return static_cast<void *>(optimizer);
  } catch (const std::exception &e) {
    std::cerr << "Error creating optimizer instance: " << e.what() << std::endl;
    return nullptr;
  } catch (...) {
    std::cerr << "Unknown error creating optimizer instance." << std::endl;
    return nullptr;
  }
}

void destroy_optimizer(void *optimizer_handle) {
  if (!optimizer_handle)
    return;
  // IMPORTANT: Match the type used in create_optimizer
  auto *optimizer = static_cast<SnapStateBOptimizer *>(optimizer_handle);
  try {
    delete optimizer;
  } catch (const std::exception &e) {
    std::cerr << "Error destroying optimizer instance: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "Unknown error destroying optimizer instance." << std::endl;
  }
}

CVector optimizer_act(void *optimizer_handle, const double *state_data,
                      int state_size) {
  if (!optimizer_handle || !state_data || state_size <= 0) {
    return {nullptr, 0};
  }
  // IMPORTANT: Match the type used in create_optimizer
  auto *optimizer = static_cast<SnapStateBOptimizer *>(optimizer_handle);
  try {
    Eigen::VectorXd state = to_eigen_vector(state_data, state_size);
    Eigen::VectorXd result = optimizer->act(state);
    return to_cvector(result);
  } catch (const std::exception &e) {
    std::cerr << "Error in optimizer_act: " << e.what() << std::endl;
    return {nullptr, 0};
  } catch (...) {
    std::cerr << "Unknown error in optimizer_act." << std::endl;
    return {nullptr, 0};
  }
}

void optimizer_update(void *optimizer_handle, const double *sample_data,
                      int sample_size, const double *observation_data,
                      int observation_size) {
  if (!optimizer_handle || !sample_data || sample_size <= 0 ||
      !observation_data || observation_size <= 0) {
    return;
  }
  // IMPORTANT: Match the type used in create_optimizer
  auto *optimizer = static_cast<SnapStateBOptimizer *>(optimizer_handle);
  try {
    Eigen::VectorXd sample = to_eigen_vector(sample_data, sample_size);
    Eigen::VectorXd observation =
        to_eigen_vector(observation_data, observation_size);
    optimizer->update(sample, observation);
  } catch (const std::exception &e) {
    std::cerr << "Error in optimizer_update: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown error in optimizer_update." << std::endl;
  }
}

CVector optimizer_best_arm_prediction(void *optimizer_handle,
                                      const double *state_data,
                                      int state_size) {
  if (!optimizer_handle || !state_data || state_size <= 0) {
    return {nullptr, 0};
  }
  // IMPORTANT: Match the type used in create_optimizer
  auto *optimizer = static_cast<SnapStateBOptimizer *>(optimizer_handle);
  try {
    Eigen::VectorXd state = to_eigen_vector(state_data, state_size);
    Eigen::VectorXd result = optimizer->best_arm_prediction(state);
    return to_cvector(result);
  } catch (const std::exception &e) {
    std::cerr << "Error in optimizer_best_arm_prediction: " << e.what()
              << std::endl;
    return {nullptr, 0};
  } catch (...) {
    std::cerr << "Unknown error in optimizer_best_arm_prediction." << std::endl;
    return {nullptr, 0};
  }
}

CVector optimizer_best_bo_prediction(void *optimizer_handle,
                                     const double *state_data, int state_size) {
  if (!optimizer_handle || !state_data || state_size <= 0) {
    return {nullptr, 0};
  }
  // IMPORTANT: Match the type used in create_optimizer
  auto *optimizer = static_cast<SnapStateBOptimizer *>(optimizer_handle);
  try {
    Eigen::VectorXd state = to_eigen_vector(state_data, state_size);
    Eigen::VectorXd result = optimizer->best_bo_prediction(state);
    return to_cvector(result);
  } catch (const std::exception &e) {
    std::cerr << "Error in optimizer_best_bo_prediction: " << e.what()
              << std::endl;
    return {nullptr, 0};
  } catch (...) {
    std::cerr << "Unknown error in optimizer_best_bo_prediction." << std::endl;
    return {nullptr, 0};
  }
}

void free_cvector_data(CVector vec) {
  delete[] vec.data; // Matches the 'new double[]' in to_cvector
}

} // extern "C"
