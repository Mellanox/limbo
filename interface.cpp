#include "interface.hpp"

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

#include "snap_optimizer/snap_optimizer_interface.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <Eigen/Core>

#include "limbo/limbo.hpp"
#include "limbo/tools/macros.hpp"
#include "limbo/tools/random_generator.hpp"
#include <limbo/bayes_opt/bo_base.hpp>

using namespace limbo;
using namespace limbo::tools;

struct Params {
  struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
    BO_PARAM(int, stats_enabled, false);
    BO_PARAM(bool, bounded, true);
  };

  struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    BO_PARAM(int, hp_period, 1);
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
    BO_PARAM(int, samples, 50);
  };

  struct stop_maxiterations {
    BO_PARAM(int, iterations, 50);
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

struct Eval {
  BO_PARAM(size_t, dim_in, 5);
  BO_PARAM(size_t, dim_out, 1);
  virtual Eigen::VectorXd operator()(const Eigen::VectorXd &x) const = 0;
};

struct StateConfig {
  BO_PARAM(size_t, dim_state, 4);
};

struct SumEval : Eval {
  BO_PARAM(size_t, dim_in, 5);
  BO_PARAM(size_t, dim_out, 1);

public:
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {

    double sum = x.sum();

    std::cout << "  DEBUG (SumEval): Received x = " << x.transpose()
              << std::endl;
    std::cout << "  DEBUG (SumEval): Returning sum = " << sum << std::endl;

    // Return the sum as a 1-dimensional vector
    return tools::make_vector(sum);
  }
};

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
using boptimizer_signature =
    boost::parameter::parameters<boost::parameter::optional<tag::acquiopt>,
                                 boost::parameter::optional<tag::statsfun>,
                                 boost::parameter::optional<tag::initfun>,
                                 boost::parameter::optional<tag::acquifun>,
                                 boost::parameter::optional<tag::stopcrit>,
                                 boost::parameter::optional<tag::modelfun>>;
template <
    class Params, class A1 = boost::parameter::void_,
    class A2 = boost::parameter::void_, class A3 = boost::parameter::void_,
    class A4 = boost::parameter::void_, class A5 = boost::parameter::void_,
    class A6 = boost::parameter::void_>
class StateBOptimizer
    : public limbo::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6> {
public:
  virtual ~StateBOptimizer() = default;
  using base_t = limbo::bayes_opt::BoBase<Params, A1, A2, A3, A4, A5, A6>;

  using model_t = typename base_t::model_t;
  using acquisition_function_t = typename base_t::acquisition_function_t;

  struct defaults {
#ifdef USE_NLOPT
    using acquiopt_t = opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>;
#elif defined(USE_LIBCMAES)
    using acquiopt_t = opt::Cmaes<Params>;
#else
    using acquiopt_t = opt::GridSearch<Params>;
#endif
  };
  using args =
      typename boptimizer_signature::bind<A1, A2, A3, A4, A5, A6>::type;
  using acqui_optimizer_t =
      typename boost::parameter::binding<args, tag::acquiopt,
                                         typename defaults::acquiopt_t>::type;

  virtual Eigen::VectorXd act(Eigen::VectorXd state) = 0;
  virtual void update(Eigen::VectorXd sample, Eigen::VectorXd observation) = 0;
  virtual Eigen::VectorXd best_arm_prediction(Eigen::VectorXd state) = 0;

protected:
  virtual Eigen::VectorXd get_state() = 0;
  virtual Eigen::VectorXd get_state_samples() = 0;
  const model_t &model() const { return _model; }

protected:
  model_t _model;
};

template <typename EvalHandler>
class SnapStateBOptimizer
    : public StateBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>,
                             acquiopt<acqui_opt_t>, initfun<init_t>,
                             statsfun<stat_t>> {
public:
  using base_t =
      StateBOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>,
                      acquiopt<acqui_opt_t>, initfun<init_t>, statsfun<stat_t>>;
  using model_t = typename base_t::model_t;
  using acquisition_function_t = typename base_t::acquisition_function_t;
  using acqui_optimizer_t = typename base_t::acqui_optimizer_t;

  SnapStateBOptimizer(EvalHandler &e) : base_t(), _eval_handler(e) {
    this->_init(e, FirstElem(), true);
    this->_model = model_t(EvalHandler::dim_in(), EvalHandler::dim_out());
    this->_model.compute(this->_samples, this->_observations);
    std::cout << "SnapStateBOptimizer constructor called" << std::endl;
  }

  Eigen::VectorXd act(Eigen::VectorXd state) override {
    std::cout << "SnapStateBOptimizer::act called with state dimension "
              << state.size() << std::endl;

    acqui_optimizer_t acqui_optimizer;
    acquisition_function_t acqui(this->_model, this->_current_iteration);

    auto acqui_optimization = [&](const Eigen::VectorXd &x, bool g) {
      return acqui(x, FirstElem(), g);
    };
    Eigen::VectorXd starting_point = tools::random_vector(
        EvalHandler::dim_in(), Params::bayes_opt_bobase::bounded());
    Eigen::VectorXd new_sample =
        acqui_optimizer(acqui_optimization, starting_point,
                        Params::bayes_opt_bobase::bounded());
    std::cout << "SnapStateBOptimizer::act returning sample: "
              << new_sample.transpose() << std::endl;
    return new_sample;
  }

  void update(Eigen::VectorXd sample, Eigen::VectorXd observation) override {
    std::cout << "SnapStateBOptimizer::update called with sample dimension "
              << sample.size() << " and observation dimension "
              << observation.size() << " sample: " << sample.transpose()
              << "observation: " << observation.transpose() << std::endl;

    this->add_new_sample(sample, observation);

    this->_model.add_sample(this->_samples.back(), this->_observations.back());

    if (Params::bayes_opt_boptimizer::hp_period() > 0 &&
        (this->_current_iteration + 1) %
                Params::bayes_opt_boptimizer::hp_period() ==
            0) {

      this->_model.optimize_hyperparams();
    }
    this->_current_iteration++;
    this->_total_iterations++;
  }

  Eigen::VectorXd best_arm_prediction(Eigen::VectorXd state) override {
    std::cout << "SnapStateBOptimizer::best_arm_prediction called with state "
                 "dimension "
              << state.size() << std::endl;

    auto rewards = std::vector<double>(this->_observations.size());
    std::transform(this->_observations.begin(), this->_observations.end(),
                   rewards.begin(), FirstElem());
    auto max_e = std::max_element(rewards.begin(), rewards.end());

    Eigen::VectorXd result =
        this->_samples[std::distance(rewards.begin(), max_e)];

    return result;
  }

protected:
  Eigen::VectorXd get_state() override {
    struct snap_observations obs;
    snap_optimizer_get_observations(&obs, 0);

    Eigen::VectorXd state_vector(StateConfig::dim_state());

    state_vector(0) = static_cast<double>(obs.num_active_queues);
    state_vector(1) = static_cast<double>(obs.num_queues);
    state_vector(2) = static_cast<double>(obs.max_qdepth);
    state_vector(3) = static_cast<double>(obs.avg_qdepth);

    return state_vector;
  }

  Eigen::VectorXd get_state_samples() override {
    if (this->_samples.empty()) {
      Eigen::VectorXd empty_vec = Eigen::VectorXd::Zero(EvalHandler::dim_in());
      return empty_vec;
    }
    return this->_samples.back();
  }

  EvalHandler &get_eval_handler() { return _eval_handler; }
  EvalHandler &_eval_handler;
};

static SnapStateBOptimizer<Eval> *g_optimizer = nullptr;

extern "C" int cpp_optimizer_init(void) { return 0; }
extern "C" int cpp_optimizer_iteration(void) { return 0; }
extern "C" int cpp_optimizer_cleanup(void) { return 0; }

Eigen::VectorXd to_eigen_vector(const double *data, int size) {
  return Eigen::Map<const Eigen::VectorXd>(data, size);
}

CVector to_cvector(const Eigen::VectorXd &vec) {
  CVector c_vec;
  c_vec.size = vec.size();
  c_vec.data = new double[c_vec.size];
  Eigen::Map<Eigen::VectorXd>(c_vec.data, c_vec.size) = vec;
  return c_vec;
}
static SumEval global_eval_handler;
extern "C" {

void *create_optimizer() {
  try {
    auto *optimizer = new SnapStateBOptimizer<Eval>(global_eval_handler);
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
  auto *optimizer = static_cast<SnapStateBOptimizer<Eval> *>(optimizer_handle);
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
    std::cerr << "Invalid parameters: optimizer_handle=" << optimizer_handle
              << ", state_data=" << state_data << ", state_size=" << state_size
              << std::endl;
    return {nullptr, 0};
  }

  auto *optimizer = static_cast<SnapStateBOptimizer<Eval> *>(optimizer_handle);
  try {
    Eigen::VectorXd state = to_eigen_vector(state_data, state_size);
    std::cout << "DEBUG: optimizer_act received state of dimension "
              << state.size() << ": " << state.transpose() << std::endl;

    if (state.size() != StateConfig::dim_state()) {
      std::cerr << "Warning: Expected state dimension "
                << StateConfig::dim_state() << " but received " << state.size()
                << std::endl;
    }

    Eigen::VectorXd result = optimizer->act(state);

    std::cout << "DEBUG: optimizer_act returning result of dimension "
              << result.size() << ": " << result.transpose() << std::endl;

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
  auto *optimizer = static_cast<SnapStateBOptimizer<Eval> *>(optimizer_handle);
  try {
    Eigen::VectorXd sample = to_eigen_vector(sample_data, sample_size);
    Eigen::VectorXd observation =
        to_eigen_vector(observation_data, observation_size);

    std::cout << "DEBUG: optimizer_update received sample of dimension "
              << sample.size() << ": " << sample.transpose() << std::endl;
    std::cout << "DEBUG: optimizer_update received observation of dimension "
              << observation.size() << ": " << observation.transpose()
              << std::endl;

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
    std::cerr << "Invalid parameters: optimizer_handle=" << optimizer_handle
              << ", state_data=" << state_data << ", state_size=" << state_size
              << std::endl;
    return {nullptr, 0};
  }

  auto *optimizer = static_cast<SnapStateBOptimizer<Eval> *>(optimizer_handle);
  try {
    Eigen::VectorXd state = to_eigen_vector(state_data, state_size);

    std::cout
        << "DEBUG: optimizer_best_arm_prediction received state of dimension "
        << state.size() << ": " << state.transpose() << std::endl;

    if (state.size() != StateConfig::dim_state()) {
      std::cerr << "Warning: Expected state dimension "
                << StateConfig::dim_state() << " but received " << state.size()
                << std::endl;
    }

    Eigen::VectorXd result = optimizer->best_arm_prediction(state);
    std::cout
        << "DEBUG: optimizer_best_arm_prediction returning result of dimension "
        << result.size() << ": " << result.transpose() << std::endl;

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

void free_cvector_data(CVector vec) { delete[] vec.data; }
}