#ifndef LIMBO_SNAP_OPTIMIZER_INTERFACE_H
#define LIMBO_SNAP_OPTIMIZER_INTERFACE_H

// Add necessary includes if any
#include <cstdint>
#define IO_STATS_CUSTOM_BUCKETS 5
struct snap_observations {
  uint16_t num_active_queues;
  uint16_t num_queues;
  uint32_t max_qdepth;
  uint32_t avg_qdepth;
  uint64_t read_io_sizes[IO_STATS_CUSTOM_BUCKETS];
  uint64_t write_io_sizes[IO_STATS_CUSTOM_BUCKETS];
  uint64_t zero_size_ios;
};
// Declare functions or structures for the snap optimizer interface
void snap_optimizer_get_observations(struct snap_observations *obs,
                                     uint64_t time) {
  return;
}
int snap_optimizer_set_system_params(int poll_size, double poll_ratio,
                                     int max_inflights, int max_iog_batch,
                                     int max_new_ios) {
  return 0;
}

/**
 * Get performance metrics to calculate reward.
 *
 * @return Current performance metric (completion count)
 */
uint64_t snap_optimizer_get_performance_metric(void) { return 0; }

#endif // LIMBO_SNAP_OPTIMIZER_INTERFACE_H