project('limbo', 'cpp',
  version: '0.0.1',
  default_options: [
    'cpp_std=c++11',
    'warning_level=2',
    'buildtype=release'
  ]
)
add_project_arguments('-DUSE_NLOPT', '-DUSE_LIBCMAES', language: 'cpp')

boost_dep = dependency('boost', required: true, modules: ['filesystem'])
eigen_dep = dependency('eigen3', required: true, include_type: 'system')
nlopt_dep = dependency('nlopt', required: true)
libcmaes_dep = dependency('libcmaes', required: true)
openmp_dep = dependency('openmp', required: true)

dependencies = [
    boost_dep,
    eigen_dep,
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


limbo_inc = include_directories('src')
limbo_lib = library('limbo',
    cpp_args: args + [
        '-DUSE_NLOPT',
        '-DUSE_LIBCMAES',
    ],
    include_directories: limbo_inc,
    dependencies: dependencies,
    install: true
)

install_subdir('src/limbo', install_dir: 'include')
install_subdir('interface', install_dir: 'include')
all_includes = [limbo_inc]
interface_include = []
interface_sources = []
subdir('interface')
all_includes += interface_include

# Define variables for conditional sources/includes
specific_sources = []
specific_inc = []

if get_option('dummy')
  message('Building with DUMMY interface')
  subdir('dummy_interface')
  specific_sources += dummy_sources
  specific_inc += dummy_inc
  message('Specific sources: ', specific_inc.length())
else
  message('Building with SNAP interface')
  subdir('snap_interface')
  specific_sources += snap_sources
  specific_inc += snap_inc
endif
all_includes += specific_inc
message('All includes: ', all_includes.length())
message('Total interface-specific sources: ', specific_sources.length())
limbo_dep = declare_dependency(
    link_with: limbo_lib,
    include_directories: all_includes,
    sources: interface_sources + specific_sources + [
        'src/limbo/limbo.hpp'
    ], # Combine core interface + specific sources
    dependencies: dependencies
)
if get_option('dummy')
    limbo_library_interfaced = library('limbo_interface',
        include_directories: [all_includes],
        dependencies: limbo_dep,
        install: true
    )
endif
