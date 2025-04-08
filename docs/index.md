# A Nomad plugin for `ropt`

The `ropt-nomad` package extends the
[`ropt`](https://tno-ropt.github.io/ropt/dev/) module by providing a plugin that
integrates the `MADS` optimization algorithm from the
[Nomad](https://www.gerad.ca/en/software/nomad/) toolkit.  `ropt` itself is a
robust optimization framework designed for both continuous and discrete
optimization workflows and is extensible through its plugin architecture.
Installing `ropt-nomad` makes `MADS` directly available within `ropt`.

## Usage
An optimization by `ropt` using the plugin works mostly as any other
optimization run. However, there are a few things to consider:

1. Gradients are not used, any specifications relating to gradient calculations
   in `ropt` are ignored.
2. The `tolerance` optimization parameter, which can be specified in the
   optimization section is ignored.
3. Only inequality constraints are supported by `NOMAD`.
4. Linear and non-linear constraints are both supported. Linear constraints are
   not supported directly, but are internally converted to non-linear
   constraints.
5. Some additional `Nomad` options can be passed as a list of strings via the
   `options` field in the optimizer configuration. Refer to the documentation of
   the [`NomadOptimizer`][ropt_nomad.nomad.NomadOptimizer] class for supported
   options.

## Reference

::: ropt_nomad.nomad.NomadOptimizer
    options:
        members: False
