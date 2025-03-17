#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/limbo.hpp>
#include <limbo/model/gp.hpp>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>
//#include <limbo/acquisition/ucb.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/opt/nlopt_grad.hpp>
#include <limbo/limbo.hpp>
#include <limbo/serialize/text_archive.hpp>
using json = nlohmann::json;
using namespace limbo;

// RPC communication class
class SnapRPC {
private:
  int sock_fd;
  std::string socket_path;

  // RPC command strings
  const std::string PERF_POLL_CMD;
  const std::string SET_PARAMS_CMD;
  const std::string GET_OBS_CMD;

  void connect() {
    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd == -1) {
      throw std::runtime_error("Failed to create socket");
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      close(sock_fd);
      throw std::runtime_error("Failed to connect to SNAP socket");
    }
  }

  json send_command(const std::string &method, const json &params = json()) {
    // Create JSON-RPC 2.0 request
    json request;
    request["jsonrpc"] = "2.0";
    request["method"] = method;
    request["id"] = 1;

    if (!params.empty()) {
      request["params"] = params;
    }

    // Send request
    std::string request_str = request.dump();
    if (write(sock_fd, request_str.c_str(), request_str.length()) == -1) {
      throw std::runtime_error("Failed to write to socket");
    }

    // Read response with timeout
    std::string response_str;
    char buffer[4096];
    time_t start_time = time(nullptr);
    const time_t timeout = 60;

    while (time(nullptr) - start_time < timeout) {
      struct timeval tv;
      tv.tv_sec = 0;
      tv.tv_usec = 100000; // 100ms timeout

      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(sock_fd, &fds);

      int ready = select(sock_fd + 1, &fds, nullptr, nullptr, &tv);
      if (ready == -1) {
        throw std::runtime_error("Select error");
      }
      if (ready == 0) {
        continue; // Timeout, try again
      }

      int n = read(sock_fd, buffer, sizeof(buffer) - 1);
      if (n == -1) {
        throw std::runtime_error("Failed to read from socket");
      }
      if (n == 0) {
        break;
      }
      buffer[n] = '\0';
      response_str += std::string(buffer);

      try {
        json response = json::parse(response_str);
        if (response.find("error") != response.end()) {
          throw std::runtime_error("RPC error: " + response["error"].dump());
        }
        return response; // Return the full response instead of just the result
      } catch (const json::parse_error &e) {
        // Incomplete response, continue reading
        continue;
      }
    }

    throw std::runtime_error("Timeout waiting for response");
  }

public:
  SnapRPC(const std::string &path = "/var/tmp/spdk.sock")
      : socket_path(path), PERF_POLL_CMD("snap_reward_get"),
        SET_PARAMS_CMD("snap_actions_set"),
        GET_OBS_CMD("snap_observations_get") {
    connect();
  }

  ~SnapRPC() {
    if (sock_fd != -1) {
      close(sock_fd);
    }
  }

  void set_params(const std::map<std::string, std::string> &params) {
    // Map parameters according to Python implementation
    json mapped_params;
    for (std::map<std::string, std::string>::const_iterator it = params.begin();
         it != params.end(); ++it) {
      std::string mapped_key;
      if (it->first == "poll_size")
        mapped_key = "poll_cycle_size";
      else if (it->first == "poll_ratio")
        mapped_key = "poll_ratio";
      else if (it->first == "max_inflights")
        mapped_key = "max_inflights";
      else if (it->first == "max_iog_batch")
        mapped_key = "max_iog_batch";
      else if (it->first == "max_new_ios")
        mapped_key = "max_new_ios";
      else
        mapped_key = it->first;

      // Convert values to appropriate types
      if (it->first == "poll_ratio") {
        mapped_params[mapped_key] = it->second; // Keep as string
      } else {
        mapped_params[mapped_key] = std::stoi(it->second); // Convert to integer
      }
    }
    send_command(SET_PARAMS_CMD, mapped_params);
  }

  double get_reward() {
    // First measurement
    json response1 = send_command(PERF_POLL_CMD);
    std::pair<int, int> meas1 = parse_perf_response(response1);

    // Wait for sample duration
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Second measurement
    json response2 = send_command(PERF_POLL_CMD);
    std::pair<int, int> meas2 = parse_perf_response(response2);

    // Calculate IOPS
    return static_cast<double>(meas2.first - meas1.first) /
           static_cast<double>(meas2.second - meas1.second);
  }

  std::map<std::string, std::string> get_observations() {
    json params;
    params["time"] = 10; // Add time parameter as in Python
    json response = send_command(GET_OBS_CMD, params);
    return parse_obs_response(response);
  }

private:
  std::pair<int, int> parse_perf_response(const json &response) {
    // Parse the response to extract completions and timestamp
    // The response format is:
    // {"jsonrpc":"2.0","id":1,"result":{"Total
    // completions":474945387,"Timestamp (ms)":1741601210200}}
    int completions = response["result"]["Total completions"].get<int>();
    int timestamp = response["result"]["Timestamp (ms)"].get<int>();
    return std::make_pair(completions, timestamp);
  }

  std::map<std::string, std::string> parse_obs_response(const json &response) {
    std::map<std::string, std::string> obs;
    // Convert JSON response to map
    for (json::const_iterator it = response.begin(); it != response.end();
         ++it) {
      obs[it.key()] = it.value().dump();
    }
    return obs;
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
  struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
    BO_PARAM(int, stats_enabled, true);
    BO_PARAM(bool, bounded, true);
  };

  struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    BO_PARAM(int, hp_period, 1); // -1 means no hyperparameter optimization
  };

  struct opt_nloptnograd : public defaults::opt_nloptnograd { // TODO: BO seems to get stuck at local point very quickly, need to swap acqusition optimizer?
    BO_PARAM(int, iterations, 500);
    BO_PARAM(double, learning_rate, 0.01);
    BO_PARAM(double, tolerance, 1e-6);
  };

  struct kernel : public defaults::kernel {
    BO_PARAM(double, noise, 0.01);
    BO_PARAM(bool, optimize_noise,true);
  };

  struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard{
    BO_PARAM(double, sigma_sq, 1);
    BO_PARAM(int,k,0);
  };

  struct init_randomsampling {
    BO_PARAM(int, samples, 10); // Matching Python's num_sobol_trials
  };

  struct stop_maxiterations {
    BO_PARAM(int, iterations, 150);
  };

  struct acqui_ucb : public defaults::acqui_ucb {
    BO_PARAM(double, alpha, 0.5);
  };

  struct opt_rprop : public defaults::opt_rprop { //TODO: hyperparameters can converge on extremely high lengthscales - need to investigate
  };

  struct mean_constant : public defaults::mean_constant {
    BO_PARAM(double, constant, 0);
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

    return tools::make_vector(final_score);
  }
};

struct DummyEval {
    // number of input dimension (x.size())
    BO_PARAM(size_t, dim_in, 3);
    // number of dimensions of the result (res.size())
    BO_PARAM(size_t, dim_out, 1);

    // the function to be optimized
    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        //Optimal point is at (0.5, 0 , 0.3) (since optimization is bounded to [0,1]^3)
        Eigen::VectorXd vec(3);
        vec << 0.5, -0.5 , 0.3;//, 0,0;
        double y = - (x-vec).norm() * 0.1;
        // Add Gaussian noise to y
        double noise_mean = 0.0;
        double noise_stddev = 0.0;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(noise_mean, noise_stddev);
        y += d(gen);
        // we return a 1-dimensional vector
        return tools::make_vector(y);
    }
};

int main() {
  try {
    // Initialize RPC client


    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    using kernel_t = kernel::SquaredExpARD<Params>;
    //using mean_t = mean::Data<Params>;
    //using mean_t = mean:FunctionARD<Params, mean::Constant>;
    using mean_t = mean::Constant<Params>;
    using gp_opt_t = model::gp::KernelMeanLFOpt<Params>;
    using gp_t = model::GP<Params, kernel_t, mean_t, gp_opt_t>;
    using Acqui_t = acqui::UCB<Params, gp_t>;
    // Initialize and run optimizer
    bayes_opt::BOptimizer<Params,modelfun<gp_t>,acquifun<Acqui_t>> optimizer;

    #ifdef USE_DUMMY
      std::cout << "Optimizing Dummy function" << std::endl;
      optimizer.optimize(DummyEval());
    #else
      std::cout << "Optimizing SNAP" << std::endl;
      SnapRPC rpc;
      optimizer.optimize(Eval(rpc));
    #endif

      // End timing
      auto end_time = std::chrono::high_resolution_clock::now();
      double total_time =
      std::chrono::duration<double>(end_time - start_time).count();

      // Print results
      std::cout << "Optimization completed successfully!" << std::endl;
      std::cout << "Total time: " << total_time << " seconds" << std::endl;

      // Convert best sample to SNAP parameters
      Eigen::VectorXd best_x = optimizer.best_sample();
      std::cout << best_x << std::endl;

    #ifndef USE_DUMMY
      SnapParams best_params = SnapParams::from_optimization_space(best_x);

      std::cout << "\nBest SNAP parameters:" << std::endl;
      std::cout << "poll_size: " << best_params.poll_size << std::endl;
      std::cout << "poll_ratio: " << best_params.poll_ratio << std::endl;
      std::cout << "max_inflights: " << best_params.max_inflights << std::endl;
      std::cout << "max_iog_batch: " << best_params.max_iog_batch << std::endl;
      std::cout << "max_new_ios: " << best_params.max_new_ios << std::endl;
      std::cout << "\nBest IOPS: " << optimizer.best_observation()(0)
        << std::endl;
    #endif

    std::string save_directory = "gp_model";
    gp_t gp_model = optimizer.model();
    gp_model.save<serialize::TextArchive>(serialize::TextArchive(save_directory));

  } catch (const std::exception &e) {
    std::cerr << "Error during optimization: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}