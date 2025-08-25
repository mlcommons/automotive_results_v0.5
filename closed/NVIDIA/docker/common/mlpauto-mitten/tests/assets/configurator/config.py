from tests.assets.configurator.my_fields import batch_size, use_some_kernel

EXPORTS = {
    "default": {
        batch_size: 12,
        use_some_kernel: True
    },
    "MaxP": {
        batch_size: 16,
    },
    "MaxQ": {
        batch_size: 8,
        use_some_kernel: False,
    }
}