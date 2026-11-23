"""Test-session settings."""

import torch._dynamo

# The optimizer's fused kernels are compiled with dynamic=False, i.e. once per
# tensor shape. A real run uses ~4-5 shapes (within the default limit of 8), but
# the test session exercises many more, which would hit the recompile limit.
for name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, name):
        setattr(torch._dynamo.config, name, 256)
