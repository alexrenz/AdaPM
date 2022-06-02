from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

##############################################
#
#   PyTorch bindings to AdaPS
#
##############################################

# helpful info on this: https://docs.python.org/3/distutils/setupscript.html

ps_dir = '../'
# we need absolute paths at least for the so-links to protobuf-lite and zmq
ps_dir = os.path.abspath(ps_dir)+'/'
deps_dir = ps_dir + 'deps_bindings/'

# include_dirs = cpp_extension.include_paths() # get default include dirs
ps_include_dirs = [ps_dir,
                      ps_dir + 'src',
                      ps_dir + 'include',
                      deps_dir + 'include']

setup(name='adaps',
      version='0.1',
      description='PyTorch bindings to the AdaPS parameter server',
      ext_modules=[cpp_extension.CppExtension(
          name='adaps',
          include_dirs = ps_include_dirs,
          extra_objects = [ps_dir + 'build/libps.a'],
          depends       = [ps_dir + 'build/libps.a',
                           ps_dir + 'include/ps/addressbook.h',
                           ps_dir + 'include/ps/base.h',
                           ps_dir + 'include/ps/coloc_kv_server.h',
                           ps_dir + 'include/ps/coloc_kv_server_handle.h',
                           ps_dir + 'include/ps/coloc_kv_worker.h',
                           ps_dir + 'include/ps/kv_app.h',
                           ps_dir + 'include/ps/ps.h',
                           ps_dir + 'include/ps/sync_manager.h',
                           ps_dir + 'include/ps/sampling.h',
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

