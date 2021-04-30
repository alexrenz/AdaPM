from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

##############################################
#
#   PyTorch bindings to Lapse
#
##############################################

# helpful info on this: https://docs.python.org/3/distutils/setupscript.html

lapse_dir = '../'
# we need absolute paths at least for the so-links to protobuf-lite and zmq
lapse_dir = os.path.abspath(lapse_dir)+'/'
deps_dir = lapse_dir + 'deps_bindings/'

# include_dirs = cpp_extension.include_paths() # get default include dirs
lapse_include_dirs = [lapse_dir,
                      lapse_dir + 'src',
                      lapse_dir + 'include',
                      deps_dir + 'include']

setup(name='lapse',
      version='0.1',
      description='PyTorch bindings to the Lapse parameter server',
      ext_modules=[cpp_extension.CppExtension(
          name='lapse',
          include_dirs = lapse_include_dirs,
          extra_objects = [lapse_dir + 'build/libps.a'],
          depends       = [lapse_dir + 'build/libps.a',
                           lapse_dir + 'include/ps/addressbook.h',
                           lapse_dir + 'include/ps/base.h',
                           lapse_dir + 'include/ps/coloc_kv_server.h',
                           lapse_dir + 'include/ps/coloc_kv_server_handle.h',
                           lapse_dir + 'include/ps/coloc_kv_transfers.h',
                           lapse_dir + 'include/ps/coloc_kv_worker.h',
                           lapse_dir + 'include/ps/kv_app.h',
                           lapse_dir + 'include/ps/ps.h',
                           lapse_dir + 'include/ps/replica_manager.h',
                           lapse_dir + 'include/ps/sampling.h',
                          ],
          # The linking we do below in `extra_link_args` would probably be cleaner with
          # `runtime_library_dirs` and `libraries`, but I did not get that to work.
          extra_link_args = ['-Wl,-rpath,'+deps_dir+'lib',
                             '-L'+deps_dir+'lib',
                             '-lprotobuf-lite',
                             '-lzmq'],
          sources=['bindings.cc'],
          extra_compile_args=['-DKEY_TYPE=int64_t'],
          # define_macros=[('NDEBUG', '1')],
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

