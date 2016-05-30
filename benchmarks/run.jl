using BenchmarkTools
using NDSparseData

suite = BenchmarkGroup()

###
# Helpers
###

function copy_unflushed(S, cols, data, default)
   NDSparse(Indexes(map(c->similar(c, 0), cols)...), similar(data, 0), default, (S,S,S), Indexes(deepcopy(cols)...), copy(data))
end

function copy_flushed(S, cols, data)
   NDSparse((S,S,S), deepcopy(cols)..., copy(data))
end

###
# Seeding
###

seed = 103
srand(seed)

###
# Input parameters:
###

const S = 10^4          # Size of array along any dimension.
const N = 10^3          # Number of entries in array
const D = 3             # Number of dimensions.
const vect_size = div(N, 10)  # Size of vector indices

cols = [rand(1 : S, N) for _ in 1 : D]                   # Columns for NDSparse object
cols_ = [rand(1 : S, N) for _ in 1 : D]
data = trues(N)                                          # Data for NDSparse object
flushed_arr = copy_flushed(S, cols, data)                   # NDSparse object

###
# Construction
###

numeric_data = rand(Int, N)
mixed_data = let sz = div(N, 4)
   vcat(rand(Int, sz), rand(sz), map(x->randstring(5), 1 : sz), bitrand(N - 3*sz))::Vector{Any}
end

suite["construction"] = BenchmarkGroup()
suite["construction"]["numeric"] = @benchmarkable NDSparse($(S,S,S), c..., d) setup=(c = deepcopy(cols); d = copy(numeric_data))
suite["construction"]["mixed"] = @benchmarkable NDSparse($(S,S,S), c..., d) setup=(c = deepcopy(cols); d = WithDefault(mixed_data, nothing))

###
# Flush
###
suite["flush!"] = BenchmarkGroup()

suite["flush!"]["unflushed"] = @benchmarkable flush!(arr) setup=(arr = copy_unflushed(S, cols, data, false))
suite["flush!"]["flushed"] = @benchmarkable flush!($flushed_arr)

###
# Getindex
###

unit_get_index = Base.ith_all(rand(1 : N), (cols...))
range_get_indices = let start = rand(1 : N - vect_size)
   (map(x -> UnitRange(sort!([x[start], x[start + vect_size]])...), cols)...)
end
vector_get_indices = let v = rand(1 : N, vect_size)
   (map(x -> x[v], cols)...)
end

suite["getindex"] = BenchmarkGroup()

suite["getindex"]["unit"] = @benchmarkable getindex($flushed_arr, $unit_get_index...)
suite["getindex"]["range"] = @benchmarkable getindex($flushed_arr, $range_get_indices...)
suite["getindex"]["vector"] = @benchmarkable getindex($flushed_arr, $vector_get_indices...)


###
# Setindex!
###

unit_set_index = map(x -> N + 1 + x, unit_get_index)
range_set_indices = (map(x -> x : (x + vect_size), rand(1 : (N - vect_size), D))...)
vector_set_indices = map(x -> x .+ N, vector_get_indices)

unit_val = true
range_vals = trues(mapreduce(length, +, 0, range_set_indices))
vector_vals = trues(vect_size)

suite["setindex!"] = BenchmarkGroup(["mutator"])

suite["setindex!"]["overwrite", "unit"] = @benchmarkable setindex!(arr, unit_val, unit_get_index...) setup=(arr=copy_flushed(S, cols, data))
suite["setindex!"]["overwrite", "range"] = @benchmarkable setindex!(arr, range_vals, range_set_indices...) setup=(arr=copy_flushed(S, cols, data))
suite["setindex!"]["overwrite", "vector"] = @benchmarkable setindex!(arr, vector_vals, vector_get_indices...) setup=(arr=copy_flushed(S, cols, data))

suite["setindex!"]["freshwrite", "unit"] = @benchmarkable setindex!(arr, unit_val,unit_set_index...) setup=(arr=copy_flushed(S, cols, data))
suite["setindex!"]["freshwrite", "vector"] = @benchmarkable setindex!(arr, vector_vals, vector_set_indices...) setup=(arr=copy_flushed(S, cols, data))


###
# Operations
###
data_ = rand(1:S, N)
filter_function(x) = x < S/2

suite["operations"] = BenchmarkGroup()

suite["operations"]["merge"] = @benchmarkable merge(target, to_merge) setup=(target = copy_flushed(S, cols, data); to_merge = copy_flushed(S, cols_, data))
suite["operations"]["naturaljoin"] = @benchmarkable naturaljoin(left, right, |) setup=(left = copy_flushed(S, cols, data); right = copy_flushed(S, cols_, data))
suite["operations"]["select"] = @benchmarkable select($flushed_arr, map(d -> d => filter_function, 1 : D)...)
suite["operations"]["filter"] = @benchmarkable filter(filter_function, $flushed_arr)


###
# Tuning 
###

try
   using JLD
   try
      loadparams!(suite, JLD.load("params.jld", "suite"), :evals, :samples);
   catch err
      println(err, " Tuning (this may take a while)")
      tune!(suite);
      JLD.save("params.jld", "suite", params(suite));
   end
catch err
   println(err, "Tuning (this may take a while)")
   tune!(suite)
end

###
# Execution
###

const duration = 10

result = median(run(suite, verbose = true, seconds = duration))

if(ARGS[1] == "execute")
   showall(result)
elseif ARGS[1] == "save"
   result = median(run(suite, verbose = true, seconds = duration))
   filename = (length(ARGS) > 1) ? ARGS[2] : "benchmarks/latest.jld";
   JLD.save(filename, "result", result)
   println("Result saved successfully")
elseif ARGS[1] == "judge"
   filename = (length(ARGS) > 1) ? ARGS[2] : "benchmarks/latest.jld";
   previous_result = JLD.load(filename, "result")
   showall(judge(result, previous_result))
end