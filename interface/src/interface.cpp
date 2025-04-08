#include "private/priv_interface.hpp"
#include "public/interface.hpp"
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

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static StateBOptimizer<Eval> *g_optimizer = nullptr;

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
extern "C" {

void *create_optimizer(OptimizerFactoryFunc factory_func) {
  try {
    if (!factory_func) {
      std::cerr << "Error: Optimizer factory function is null." << std::endl;
      return nullptr;
    }
    // Call the provided factory function to get the optimizer instance
    void *optimizer_handle = factory_func(); 
    if (!optimizer_handle) {
      std::cerr << "Error: Optimizer factory function returned null." << std::endl;
      return nullptr;
    }
    // We assume the factory function returns a valid pointer to a concrete optimizer
    // The type safety needs to be handled by the caller providing the factory
    return optimizer_handle;
  } catch (const std::exception &e) {
    std::cerr << "Error creating optimizer via factory: " << e.what() << std::endl;
    return nullptr;
  } catch (...) {
    std::cerr << "Unknown error creating optimizer via factory." << std::endl;
    return nullptr;
  }
}

void destroy_optimizer(void *optimizer_handle) {
  if (!optimizer_handle)
    return;
  auto *optimizer = static_cast<StateBOptimizer<Eval> *>(optimizer_handle);
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

  auto *optimizer = static_cast<StateBOptimizer<Eval> *>(optimizer_handle);
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
  auto *optimizer = static_cast<StateBOptimizer<Eval> *>(optimizer_handle);
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

  auto *optimizer = static_cast<StateBOptimizer<Eval> *>(optimizer_handle);
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