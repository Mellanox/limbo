#pragma once
#include <Eigen/Core>
#include <limbo/limbo.hpp>
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

