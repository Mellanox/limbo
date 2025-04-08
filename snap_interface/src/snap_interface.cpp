#include "snap_interface.hpp"
#include "interface/interface.hpp"
extern "C" {

static SnapStateBOptimizer* snap_optimizer;

static void* snap_optimizer_factory(void) {
    return new SnapStateBOptimizer();
}

int cpp_optimizer_init(void) {;
    SnapStateBOptimizer* optimizer = (SnapStateBOptimizer*)create_optimizer(&snap_optimizer_factory);
    if (!optimizer) {
        std::cerr << "Error: Failed to create optimizer." << std::endl;
        return -1;
    }
    snap_optimizer = optimizer;
    return 0;
}

int cpp_optimizer_iteration(void) { return 0; }

int cpp_optimizer_cleanup(void) {
    destroy_optimizer(snap_optimizer);
    return 0;
}

}