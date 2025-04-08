#include <public/stateboptimizer.hpp>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iterator>

using namespace limbo;
using namespace limbo::defaults;

template <typename EvalHandler>
class DummyStateBOptimizer
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

  DummyStateBOptimizer(EvalHandler &e) : base_t(), _eval_handler(e) {
    this->_init(e, FirstElem(), true);
    this->_model = model_t(EvalHandler::dim_in(), EvalHandler::dim_out());
    this->_model.compute(this->_samples, this->_observations);
    std::cout << "DummyStateBOptimizer constructor called" << std::endl;
  }

  Eigen::VectorXd act(Eigen::VectorXd state) override {
    std::cout << "DummyStateBOptimizer::act called with state dimension "
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
    std::cout << "DummyStateBOptimizer::act returning sample: "
              << new_sample.transpose() << std::endl;
    return new_sample;
  }

  void update(Eigen::VectorXd sample, Eigen::VectorXd observation) override {
    std::cout << "DummyStateBOptimizer::update called with sample dimension "
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
    std::cout << "DummyStateBOptimizer::best_arm_prediction called with state "
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
    return Eigen::VectorXd::Zero(StateConfig::dim_state());
  }

  Eigen::VectorXd get_state_samples() override {
    return Eigen::VectorXd::Zero(StateConfig::dim_state());
  }

  EvalHandler &get_eval_handler() { return _eval_handler; }
  EvalHandler &_eval_handler;
};