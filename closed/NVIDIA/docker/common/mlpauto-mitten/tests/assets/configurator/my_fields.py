from nvmitten.configurator import Field


batch_size = Field("batch_size",
                   description="A batch size for the model to use",
                   from_string=int,
                   disallow_default=True)


use_some_kernel = Field("use_some_kernel",
                        description="If set, will use some custom kernel.",
                        from_string=bool)
