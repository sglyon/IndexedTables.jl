module IndexedTables

using Compat
using NamedTuples, PooledArrays

import Base:
    show, eltype, length, getindex, setindex!, ndims, map, convert, keys, values,
    ==, broadcast, empty!, copy, similar, sum, merge, merge!, mapslices,
    permutedims, reducedim, serialize, deserialize

export NDSparse, flush!, aggregate!, aggregate_vec, where, pairs, convertdim, columns, column, rows, as,
    itable, update!, aggregate, reducedim_vec, dimlabels

const Tup = Union{Tuple,NamedTuple}
const DimName = Union{Int,Symbol}

include("common/utils.jl")
include("common/columns.jl")

include("table/table.jl")
include("table/join.jl")
include("table/query.jl")


include("ndsparse/ndsparse.jl")
include("ndsparse/indexing.jl")
include("ndsparse/join.jl")
include("ndsparse/query.jl")

include("common/tabletraits.jl")

end # module
