from ax import (
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.modelbridge.factory import get_sobol

N = 5

search_space = SearchSpace(
    parameters=[
        RangeParameter(name="num_replicas",
                       parameter_type=ParameterType.FLOAT,
                       lower=0.01,
                       upper=100,
                       log_scale=True),
        RangeParameter(name="ahead",
                       parameter_type=ParameterType.INT,
                       lower=1,
                       upper=1000,
                       log_scale=True),
    ]
)

m = get_sobol(search_space)
gr = m.gen(n=N)

for arm in gr.arms:
    print(str(arm.parameters) + ",")
