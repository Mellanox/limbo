project('limbo', 'cpp',
  version: '0.0.1',
  default_options: [
    'cpp_std=c++11',
    'warning_level=2',
    'buildtype=release'
  ]
)

# Find required dependencies
boost_dep = dependency('boost', required: true)
tbb_dep = dependency('tbb', required: true)
eigen_dep = dependency('eigen3', required: true, include_type: 'system')
spdlog_dep = dependency('spdlog', required: true)

# Find optional dependencies
use_nlopt = get_option('use-nlopt')
use_cmaes = get_option('use-cmaes')
use_tbb = get_option('use-tbb')

nlopt_dep = dependency('nlopt', required: use_nlopt)
libcmaes_dep = dependency('libcmaes', required: use_cmaes)

# Create interface library target
limbo_lib = library('limbo',
    include_directories: include_directories('src'),
    cpp_args: [
        '-DNOMINMAX',
        '-D_CRT_SECURE_NO_WARNINGS',
        '-Wno-error=cpp',  # Disable treating #warning as error
        '-Wno-unused-variable'  # Disable unused variable warnings
    ] + (use_nlopt ? ['-DUSE_NLOPT'] : []) + 
      (use_cmaes ? ['-DUSE_LIBCMAES'] : []) +
      (use_tbb ? ['-DUSE_TBB'] : []),
    dependencies: [
        boost_dep,
        tbb_dep,
        eigen_dep,
        spdlog_dep
    ] + (use_nlopt ? [nlopt_dep] : []) +
      (use_cmaes ? [libcmaes_dep] : [])
)

# Create dependency object to expose the library to parent project
limbo_dep = declare_dependency(
    link_with: limbo_lib,
    include_directories: include_directories('src'),
    compile_args: [
        '-DNOMINMAX',
        '-D_CRT_SECURE_NO_WARNINGS',
        '-Wno-error=cpp',  # Disable treating #warning as error
        '-Wno-unused-variable'  # Disable unused variable warnings
    ] + (use_nlopt ? ['-DUSE_NLOPT'] : []) + 
      (use_cmaes ? ['-DUSE_LIBCMAES'] : []) +
      (use_tbb ? ['-DUSE_TBB'] : []),
    dependencies: [
        boost_dep,
        tbb_dep,
        eigen_dep,
        spdlog_dep
    ] + (use_nlopt ? [nlopt_dep] : []) +
      (use_cmaes ? [libcmaes_dep] : [])
) 