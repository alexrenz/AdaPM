# PyTorch Bindings

We provide (so far experimental) bindings to PyTorch. The bindings are specified in [bindings.cc](bindings.cc), usage examples can be found in [example.py](example.py). 

We provide bindings for the main primitives of AdaPM: `pull`, `push`, `intent`, `advance_clock`, `prepare_sample`, and `pull_sample`. They further provide a `set` variant of the `push` primitive that sets parameter to specific values (instead of adding to the values, as `push` does).

### PyTorch tensors

```python
keys = torch.LongTensor([1,2,3,4])
values = torch.ones((len(keys)*10), dtype=torch.float32)

# pull
kv.pull(keys, values)

# push
kv.push(keys, values)

# set (variant of push)
kv.set(keys, values)

# signal intent and advance clock
kv.intent(keys, 2, 3)

kv.advance_clock()
```

### NumPy arrays
The bindings also accept numpy arrays:

```python
keys = np.array([1,2,3,4])
values = np.ones(len(keys)*10, dtype=np.float32)

# pull
kv.pull(keys, values)

# push
kv.push(keys, values)

# set (variant of push)
kv.set(keys, values)

# signal intent and advance clock
kv.intent(keys, 2, 3)

kv.advance_clock()
```

### Synchronous and asynchronous operations
By default, operations run synchronously. To run asynchronously, pass `async=True` to the operation:
```python
kv.pull(keys, values, True)
kv.push(keys, values, True)
```
In particular, `push` is often executed asynchronously. 

`kv.wait()` explicitly waits for the execution of a specific operation: 
```python
timestamp = kv.pull(keys, values, True)

# do something else

kv.wait(timestamp) # wait for pull to finish
```
Parameter access operations (`pull`, `push`, `set`, and `pull_sample`) return a timestamp that can be used this way.


## Installation

To compile working bindings, it is important that AdaPM is built with a key
data type that matches PyTorch and NumPy default integer data types (`int64`)
and that AdaPM is built with the the same C++ ABI as the installed PyTorch. If
you installed PyTorch in a normal way (e.g., pip or conda), it is likely that
this PyTorch installation uses the old pre-C++11 ABI (apparently, this is done
to ensure compatibility to old Python versions). If you are on a recent OS, this
will be incompatible with system-provided Protocol Buffers. Thus, you will
probably have to install Protocol Buffers manually. To check the ABI of your
installed PyTorch, run `python bindings/lookup_torch_abi.py` (`0` means
pre-C++11 ABI, `1` means C++ ABI). If that script returns `0` and you are on a
recent OS with C++ ABI (e.g., compare `echo '#include <string>' | g++ -x c++ -E
-dM - | fgrep _GLIBCXX_USE_CXX11_ABI`), you have to install Protocol Buffers
manually. (Sorry :/)


To manually install protocol buffers with the pre-c++11 ABI, use:

```bash
# (in the root folder of this repository)
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protobuf-cpp-3.6.1.tar.gz
tar zxf protobuf-cpp-3.6.1.tar.gz
cd protobuf-3.6.1
mkdir release && cd release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/../../deps/ -D CMAKE_CXX_FLAGS="-fPIC -D_GLIBCXX_USE_CXX11_ABI=0" -D protobuf_BUILD_TESTS=OFF ../cmake/
make -j  # this can take a while
make install
```

Then to compile AdaPM with the matching key type and the ABI of the installed PyTorch:
```bash
# (in the root folder of this repository)
cmake -S . -B build_bindings -D PS_KEY_TYPE=int64_t -DPROTOBUF_PATH=$(pwd)/deps/lib/cmake/protobuf/ -D CMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$(python bindings/lookup_torch_abi.py)"
cmake --build build_bindings --target adapm
```

> **Note**
> Make sure that PyTorch is installed before you build AdaPM. Otherwise, the ABI read-out (`$(python bindings/lookup_torch_abi.py)` above) will fail. Also make sure that you run the read-out with the python that you are using (i.e., potentially `python3` or `python3.9` rather than `python`).

And finally, to compile the bindings:
```bash
cd bindings
python setup.py install --user
```

> **Note**
> If you built AdaPM to another path than `build_bindings/` or you installed Protocol Buffers to a folder other than `deps/`, you need to pass these paths to `setup.py` explicitly via environment variables `BUILD_PATH` and `DEPS_PATH` (using absolute paths). E.g.: `BUILD_PATH=[ABSOLUTE_PATH_TO_BUILD] DEPS_PATH=[ABSOLUTE_PATH_TO_PROTOBUF_INSTALL] python setup.py install --user`

If successful, you can now use AdaPM from Python

```python
#!/usr/bin/python
import torch
import adapm
```

You should also be able to run `python bindings/example.py` without error messages.



## Experimental status

These bindings have an experimental status. If you run into any problems or have questions, please don't hesitate to contact us.
