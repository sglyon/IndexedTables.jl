using BenchmarkTools
using NDSparseData

suite = BenchmarkGroup()

###
# Helpers
###

function copy_unflushed(cols, data)
   NDSparse(Indexes(map(c->similar(c, 0), cols)...), similar(data, 0), Indexes(deepcopy(cols)...), copy(data))
end

function copy_flushed(cols, data)
   NDSparse(deepcopy(cols)..., copy(data))
end

###
# Seeding
###

seed = 103
srand(seed)


###
# Construction
###

function construction_suite(N::Int, D::Int, cols)
   numeric_data = rand(Int, N)
   mixed_data = let sz = div(N, 4)
      vcat(rand(Int, sz), rand(sz), map(x->randstring(5), 1 : sz), bitrand(N - 3*sz))::Vector{Any}
   end

   suite = BenchmarkGroup(["Construction"])
   suite["numeric"] = @benchmarkable NDSparse(c..., d) setup=(c = deepcopy($cols); d = copy($numeric_data))
   suite["mixed"] = @benchmarkable NDSparse(c..., d) setup=(c = deepcopy($cols); d = $mixed_data)
   return suite
end

###
# Flush
###

function flush_suite(cols, data, flushed_arr)
   suite = BenchmarkGroup(["Flush"])
   suite["unflushed"] = @benchmarkable flush!(arr) setup=(arr = copy_unflushed($cols, $data))
   suite["flushed"] = @benchmarkable flush!($flushed_arr)
   return suite
end

###
# Indexing
###

function indexing_suite(N::Int, D::Int, cols, data, vect_size, flushed_arr)
   unit_get_index = Base.ith_all(rand(1 : N), (cols...))
   range_get_indices = let start = rand(1 : N - vect_size)
      (map(x -> UnitRange(sort!([x[start], x[start + vect_size]])...), cols)...)
   end
   vector_get_indices = let v = rand(1 : N, vect_size)
      (map(x -> sort(x[v]), cols)...)
   end
   unit_set_index = map(x -> N + 1 + x, unit_get_index)
   range_set_indices = (map(x -> x : (x + vect_size), sort(rand(1 : (N - vect_size), D)))...)
   vector_set_indices = map(x -> x .+ N, vector_get_indices)

   unit_val = true
   range_vals = trues(mapreduce(length, +, 0, range_set_indices))
   vector_vals = trues(vect_size)

   suite = BenchmarkGroup(["Indexing"])

   suite["getindex"] = BenchmarkGroup()
   suite["getindex"]["unit"] = @benchmarkable getindex($flushed_arr, $unit_get_index...)
   suite["getindex"]["range"] = @benchmarkable getindex($flushed_arr, $range_get_indices...)
   suite["getindex"]["vector"] = @benchmarkable getindex($flushed_arr, $vector_get_indices...)

   suite["setindex!"] = BenchmarkGroup()
   suite["setindex!"]["overwrite", "unit"] = @benchmarkable setindex!(arr, $unit_val, $unit_get_index...) setup=(arr=copy_flushed($cols, $data))
   suite["setindex!"]["overwrite", "range"] = @benchmarkable setindex!(arr, $range_vals, $range_set_indices...) setup=(arr=copy_flushed($cols, $data))
   suite["setindex!"]["overwrite", "vector"] = @benchmarkable setindex!(arr, $vector_vals, $vector_get_indices...) setup=(arr=copy_flushed($cols, $data))

   suite["setindex!"]["freshwrite", "unit"] = @benchmarkable setindex!(arr, $unit_val, $unit_set_index...) setup=(arr=copy_flushed($cols, $data))
   suite["setindex!"]["freshwrite", "vector"] = @benchmarkable setindex!(arr, $vector_vals, $vector_set_indices...) setup=(arr=copy_flushed($cols, $data))

   return suite
end

function operations_suite(S::Int, N::Int, D::Int, cols, data, flushed_arr, filter_function=x->x<S/2)
   data_ = rand(1:S, N)
   cols_ = [rand(1 : S, N) for _ in 1 : D]

   suite = BenchmarkGroup(["Operations"])

   suite["merge"] = @benchmarkable merge(target, to_merge) setup=(target = copy_flushed(cols, data); to_merge = copy_flushed($cols_, $data))
   suite["naturaljoin"] = @benchmarkable naturaljoin(left, right, |) setup=(left = copy_flushed(cols, data); right = copy_flushed($cols_, $data))
   suite["select"] = @benchmarkable select($flushed_arr, map(d -> d => filter_function, 1 : D)...)
   suite["filter"] = @benchmarkable filter($filter_function, $flushed_arr)
end

function build_suite(S, N, D, vect_size, cols, data, flushed_arr)
   suite = BenchmarkGroup(["ROOT"])
   suite["Construction"] = construction_suite(N, D, cols)
   suite["Flush"] = flush_suite(cols, data, flushed_arr)
   suite["Indexing"] = indexing_suite(N, D, cols, data, vect_size, flushed_arr)
   suite["Operations"] = operations_suite(S, N, D, cols, data, flushed_arr)

   return suite
end

###
# CLI
###
using JLD

function main()
   println("Usage: julia run.jl [tune-file] [output-file] [compare-file] [Array scale] [Number of elements] [Number of dimensions] [vect_size]")

   S = try parse(Int, ARGS[4]) catch 10^4 end                 # Size of array along any dimension.
   N = try parse(Int, ARGS[5]) catch 10^3 end                 # Number of entries in array
   D = try parse(Int, ARGS[6]) catch 3 end                    # Number of dimensions.
   vect_size = try parse(Int, ARGS[7]) catch div(N, 10) end   # Size of vector indices

   cols = [rand(1 : S, N) for _ in 1 : D]                     # Columns for NDSparse object
   data = trues(N)                                            # Data for NDSparse object
   flushed_arr = copy_flushed(cols, data)                     # NDSparse object)

   suite = build_suite(S, N, D, vect_size, cols, data, flushed_arr)

   if length(ARGS) > 0
      tune_file = ARGS[1]
      if isfile(tune_file)
         println("Using benchmark tuning data in $tune_file")
         loadparams!(suite, JLD.load(tune_file, "suite"), :evals, :samples)
      else
         println("Creating benchmark tuning file $tune_file")
         tune!(suite)
         JLD.save(tune_file, "suite", params(suite))
      end
   else
      println("Tuning benchmarks. Tip: provide a tune_file argument to save this step.")
      tune!(suite)
   end

   println("running benchmarks")
   res = median(run(suite))
   @show(res)
   output = get(ARGS, 2, "Output-$(Dates.now()).jld")
   println("saving results in $output")
   JLD.save(output, "result", res)
   println("Done.")

   if length(ARGS) > 2
      compare_file = ARGS[3]
      println("Comparing with results in $compare_file")
      reference_results = JLD.load(compare_file, "result")
      @show judge(reference_results, res)
   end
end

main()
