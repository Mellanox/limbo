#include <public/stateboptimizer.hpp>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

using namespace limbo;
using namespace limbo::defaults;

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

  DummyStateBOptimizer(int dim_in, int dim_out)
    : base_t(), _dim_in(dim_in), _dim_out(dim_out) { 
    this->_model = model_t(_dim_in, _dim_out);
  }

  Eigen::VectorXd act(Eigen::VectorXd state) override {
    std::cout << "SnapStateBOptimizer::act called with state dimension "
              << state.size() << std::endl;

    if (this->_current_iteration < Params::init_randomsampling::samples()) {
      Eigen::VectorXd starting_point = tools::random_vector(
          _dim_in, Params::bayes_opt_bobase::bounded());
      return starting_point;
    }

    acqui_optimizer_t acqui_optimizer;
    acquisition_function_t acqui(this->_model, this->_current_iteration);

    auto acqui_optimization = [&](const Eigen::VectorXd &x, bool g) {
      return acqui(x, FirstElem(), g);
    };
    Eigen::VectorXd starting_point = tools::random_vector(
        _dim_in, Params::bayes_opt_bobase::bounded());
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

    // TODO: use update_stats
    this->add_new_sample(sample, observation);

    if (this->_current_iteration == Params::init_randomsampling::samples()) {
      // Compute model with initial samples and observations
      this->_model.compute(this->_samples, this->_observations);
    } else if (this->_current_iteration >
               Params::init_randomsampling::samples()) {
      // Update model with new sample and observation
      this->_model.add_sample(sample, observation);
      if (Params::bayes_opt_boptimizer::hp_period() > 0 &&
          (this->_current_iteration + 1) %
                  Params::bayes_opt_boptimizer::hp_period() ==
              0)
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

    int best_arm_index = std::distance(rewards.begin(), max_e);
    Eigen::VectorXd best_arm = this->_samples[best_arm_index];

    // TODO: best_arm_prediction should also return reward prediction and
    // uncertainty (should be returned as vector of size dim + 2?)
    Eigen::VectorXd reward_prediction =
        this->_model_prediction(best_arm, state);
    Eigen::VectorXd result(_dim_in + 2);
    result << best_arm, reward_prediction;
    return result;
  }

protected:
  Eigen::VectorXd _model_prediction(Eigen::VectorXd params,
                                    Eigen::VectorXd state) {
    Eigen::VectorXd reward_prediction = this->_model.mu(params);
    double uncertainty =
        this->_model.sigma(params) - this->_model.kernel_function().noise();
    Eigen::VectorXd result(2);
    result << reward_prediction, uncertainty;
    return result;
  };
  Eigen::VectorXd get_state() override {
    return Eigen::VectorXd::Zero(StateConfig::dim_state());
  }

  Eigen::VectorXd get_state_samples() override {
    return Eigen::VectorXd::Zero(StateConfig::dim_state());
  }

private:
  int _dim_in;
  int _dim_out;
};