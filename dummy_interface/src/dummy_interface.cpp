#include <dummy_interface/include/public/dummy_state_optimizer.hpp> // Include the header with the class definition
#include <iostream>


extern "C" void* create_optimizer_instance(int dim_in, int dim_out) {
    std::cout << "Executing DUMMY factory (in cpp): create_optimizer_instance()" << std::endl;

    return new DummyStateBOptimizer(dim_in, dim_out);
} 