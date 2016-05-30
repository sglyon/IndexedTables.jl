# Benchmarking

Saving and loading benchmarks will require *JLD*.

## Simple Benchmark execution
`julia benchmarks/run.jl execute`
The resulting `BenchmarkGroup` will be stored in a variable named `result`

## Save Benchmark to file
`julia benchmarks/run.jl save <filename>`
The default filename is `benchmarks/latest.jld`.

## Jude Benchmark with previous result
`julia benchmarks/run.jl judge <reference_filename>`
The default reference filename is `benchmarks/latest.jld`


# Typical Workflow
- Run `julia benchmarks/run.jl`. 
- Make your changes.
- Run `julia benchmarks/run.jl judge` to if things have improved.

# Retuning
If you make changes to the BenchmarkGroups, make sure to `delete params.jld`.