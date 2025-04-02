project('limbo', 'cpp',
  version: '0.0.1',
  default_options: [
    'cpp_std=c++11',
    'warning_level=2',
    'buildtype=release'
  ]
)
# Define required preprocessor macros
add_project_arguments('-DUSE_NLOPT', '-DUSE_LIBCMAES', language: 'cpp')

# Find required dependencies
boost_dep = dependency('boost', required: true, modules: ['filesystem'])
#tbb_dep = dependency('tbb', required: true)
eigen_dep = dependency('eigen3', required: true, include_type: 'system')
spdlog_dep = dependency('spdlog', required: true)
nlopt_dep = dependency('nlopt', required: true)
libcmaes_dep = dependency('libcmaes', required: true)
openmp_dep = dependency('openmp', required: true)

dependencies = [
    boost_dep,
    #tbb_dep,
    eigen_dep,
    spdlog_dep,
    nlopt_dep,
    libcmaes_dep,
    openmp_dep
]
args = [
    '-DNOMINMAX',
    '-D_CRT_SECURE_NO_WARNINGS',
    '-Wno-error=cpp',  # Disable treating #warning as error
    '-Wno-unused-variable',
    '-Wno-unused-parameter',  # Ignore unused parameter warnings
    '-Wno-unused-but-set-variable',  # Ignore unused but set variable warnings
    '-Wno-attributes',  # Ignore attribute warnings
    '-Wno-non-virtual-dtor',  # Ignore non-virtual destructor warnings
    '-Wno-error=non-virtual-dtor',  # Disable treating non-virtual destructor as error
    '-Wno-error=unused-variable',  # Disable treating unused variables as errors
    '-Wno-error=unused-but-set-variable',  # Disable treating unused but set variables as errors
]

# Define include directories
limbo_inc = include_directories('src')
snap_inc = include_directories('.') # Include current dir for snap_optimizer/ header

all_includes = [limbo_inc]
if get_option('dummy')
  all_includes += [snap_inc]
endif

# Create interface library target
limbo_lib = library('limbo',
    sources: ['interface.cpp'],
    include_directories: all_includes,
    cpp_args: args + [
        '-DUSE_NLOPT',  # Always define USE_NLOPT since we're using it
        '-DUSE_LIBCMAES',  # Always define USE_LIBCMAES since we're using it
    ],
    dependencies: dependencies
)

# Install the limbo headers
install_subdir('src/limbo', install_dir: 'include')
if get_option('dummy')
    install_subdir('snap_optimizer', install_dir: 'include/snap_optimizer')
endif
# Create dependency object that includes both library and interface files
limbo_dep = declare_dependency(
    link_with: limbo_lib,
    include_directories: all_includes,
    sources: [
        'src/limbo/limbo.hpp'  # Add the main limbo header that defines the namespace
    ],
    dependencies: dependencies
)