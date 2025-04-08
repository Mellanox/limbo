


src_files = files(
    'src/interface.cpp'
)


interface_lib = library('interface_lib', src_files,
  include_directories : [include_directories('include')]+[limbo_inc],
  dependencies: dependencies,
  link_with: limbo_lib,
  install : true
)

install_headers(
  'include/public/interface.hpp',
  subdir : 'interface' 
)
