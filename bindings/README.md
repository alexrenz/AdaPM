# PyTorch Bindings

We provide (so far experimental) bindings to PyTorch. The bindings are specified in [bindings.cc](bindings.cc), usage examples can be found in [example.py](example.py). 

The bindings provide the three primitives of Lapse: `pull`, `push`, and `localize`. They further provide a `set` variant of the `push` primitive that sets parameter to specific values (instead of adding to the values, as `push` does).

### PyTorch tensors

```python
keys = torch.LongTensor([1,2,3,4])
values = torch.ones((len(keys)*10), dtype=torch.float32)

# pull
kv.push(keys, values)

# push
kv.pull(keys, values)

# set (variant of push)
kv.set(keys, values)

# localize
kv.localize(keys)
```

### NumPy arrays
The bindings also accept numpy arrays:

```python
keys = np.array([1,2,3,4])
values = np.ones(len(keys)*10, dtype=np.float32)

# pull
kv.push(keys, values)

# push
kv.pull(keys, values)

# set (variant of push)
kv.set(keys, values)

# localize
kv.localize(keys)
```

### Synchronous and asynchronous operations
By default, all operations run synchronously. To run asynchronously, pass `async=True` to any operation:
```python
kv.pull(keys, values, True)
kv.push(keys, values, True)
kv.localize(keys, True)
```
In particular, `localize` and `push` are often executed asynchronously. 

`kv.wait()` explicitly waits for the execution of a specific operation: 
```python
timestamp = kv.pull(keys, values, True)

# do something else

kv.wait(timestamp) # wait for pull to finish
```
All operations (`pull`, `push`, `set`, and `localize`) return a timestamp that can be used this way.


## Installation


Compile Lapse with `int64` keys (to match PyTorch and NumPy default integer data
types) and with the appropriate ABI version, then use [setup.py](setup.py) to
compile the bindings (see below). Pip PyTorch installations often use the old
(pre C++11) ABI. The script [lookup_torch_abi.py](lookup_torch_abi.py) reads out
the ABI version of the installed PyTorch. You can also specify the ABI version
manually. To avoid ABI version conflicts with system provided libraries (e.g.,
boost) for the apps in [apps/](apps/) (which use system-provided boost), compile
the pre-C++ ABI dependencies to a separate path `deps_bindings` (using
`DEPS_PATH=$(pwd)/deps_bindings`). If you compile the dependencies to another
path than `deps_bindings` you have to modify [setup.py](setup.py).

```bash
make clean
make ps KEY_TYPE=int64_t CXX11_ABI=$(python bindings/lookup_torch_abi.py) DEPS_PATH=$(pwd)/deps_bindings
cd bindings
python setup.py install --user
```


If successful, you can now use lapse in Python

```python
#!/usr/bin/python
import torch
import lapse
```

## Experimental status

These bindings have an experimental status. If you run into any problems or have questions, please don't hesitate to contact us.
