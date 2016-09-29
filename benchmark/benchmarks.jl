using PkgBenchmark
using IndexedTables

###
# Helpers
###

function copy_unflushed(cols, data)
   NDSparse(Columns(map(c->similar(c, 0), cols)...), similar(data, 0), Columns(deepcopy(cols)...), copy(data))
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
   vector_vals = trues(mapreduce(length, +, 0, vector_set_indices))

   @benchgroup "getindex" begin
       @bench "unit" getindex($flushed_arr, $unit_get_index...)
       @bench "range" getindex($flushed_arr, $range_get_indices...)
       @bench "vector" getindex($flushed_arr, $vector_get_indices...)
   end

   @benchgroup "setindex!" begin
       @bench "overwrite", "unit" setindex!(arr, $unit_val, $unit_get_index...) setup=(arr=copy_flushed($cols, $data))
       @bench "overwrite", "range" setindex!(arr, $range_vals, $range_set_indices...) setup=(arr=copy_flushed($cols, $data))
       #@bench "overwrite", "vector" setindex!(arr, $vector_vals, $vector_get_indices...) setup=(arr=copy_flushed($cols, $data))

       @bench "freshwrite", "unit" setindex!(arr, $unit_val, $unit_set_index...) setup=(arr=copy_flushed($cols, $data))
       #@bench "freshwrite", "vector" setindex!(arr, $vector_vals, $vector_set_indices...) setup=(arr=copy_flushed($cols, $data))
   end
end

function build_suite(S, N, D, vect_size, cols, data, flushed_arr)
   @benchgroup "Construction" ["Construction"] begin
       numeric_data = rand(Int, N)
       mixed_data = let sz = div(N, 4)
          vcat(rand(Int, sz), rand(sz), map(x->randstring(5), 1 : sz), bitrand(N - 3*sz))::Vector{Any}
       end

       @bench "numeric" NDSparse(c..., d) setup=(c = deepcopy($cols); d = copy($numeric_data))
       @bench "mixed" NDSparse(c..., d) setup=(c = deepcopy($cols); d = $mixed_data)
   end

   @benchgroup "Flush" ["Flush"] begin
       @bench "flushed" flush!($flushed_arr)
   end

   @benchgroup "Indexing" ["Indexing"] indexing_suite(N, D, cols, data, vect_size, flushed_arr)

   @benchgroup "Operations" ["Operations"] begin
       data_ = rand(1:S, N)
       cols_ = [rand(1 : S, N) for _ in 1 : D]
       filter_function=x->x<S/2

       @bench "merge" merge(target, to_merge) setup=(target = copy_flushed($cols, $data); to_merge = copy_flushed($cols_, $data))
       @bench "naturaljoin" naturaljoin(left, right, |) setup=(left = copy_flushed($cols, $data); right = copy_flushed($cols_, $data))
       @bench "select" select($flushed_arr, $(map(d -> d => filter_function, 1 : D))...)
       @bench "filter" filter($filter_function, $flushed_arr)
   end
end

function main()
   println("Usage: julia run.jl [Array scale] [Number of elements] [Number of dimensions] [vect_size]")

   S = try parse(Int, ARGS[1]) catch 10^4 end                 # Size of array along any dimension.
   N = try parse(Int, ARGS[2]) catch 10^3 end                 # Number of entries in array
   D = try parse(Int, ARGS[3]) catch 3 end                    # Number of dimensions.
   vect_size = try parse(Int, ARGS[4]) catch div(N, 10) end   # Size of vector indices

   cols = [rand(1 : S, N) for _ in 1 : D]                     # Columns for NDSparse object
   data = trues(N)                                            # Data for NDSparse object
   flushed_arr = copy_flushed(cols, data)                     # NDSparse object)

   build_suite(S, N, D, vect_size, cols, data, flushed_arr)
end

main()
