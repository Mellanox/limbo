#include <dummy_interface/include/public/dummy_state_optimizer.hpp> // Include the header with the class definition
#include <iostream>


extern "C" void* create_optimizer_instance() {
    std::cout << "Executing DUMMY factory (in cpp): create_optimizer_instance()" << std::endl;

    // Define the minimal EvalHandler needed by the dummy optimizer class template
    struct DummyEvalHandler : public Eval {
        static constexpr int dim_in() { return 5; } // Placeholder dimension
        static constexpr int dim_out() { return 1; } // Placeholder dimension
        Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
            return Eigen::VectorXd::Zero(dim_out());
        }
    };

    // Need an instance of the handler
    static DummyEvalHandler dummy_handler;

    // Instantiate and return the dummy optimizer
    // DummyStateBOptimizer class template is defined in the included header
    return new DummyStateBOptimizer<DummyEvalHandler>(dummy_handler);
} 