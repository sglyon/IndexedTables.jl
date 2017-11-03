module IndexedTables

using Compat
using NamedTuples, PooledArrays

import Base:
    show, eltype, length, getindex, setindex!, ndims, map, convert, keys, values,
    ==, broadcast, empty!, copy, similar, sum, merge, merge!, mapslices,
    permutedims, reducedim, serialize, deserialize

export NDSparse, flush!, aggregate!, aggregate_vec, where, pairs, convertdim, columns, column, rows,
    itable, update!, aggregate, reducedim_vec, dimlabels

const Tup = Union{Tuple,NamedTuple}
const DimName = Union{Int,Symbol}

include("utils.jl")
include("columns.jl")
include("table/table.jl")
include("ndsparse/ndsparse.jl")

#=
# Poor man's traits

# These support `colnames` and `columns`
const TableTrait = Union{AbstractVector, NextTable, NDSparse}

# These support `colnames`, `columns`,
# `pkeynames`, `permcache`, `cacheperm!`
=#
const IndexedTrait = Union{NextTable, NDSparse}

include("sortperm.jl")

# no-copy convert
_convert(::Type{NextTable}, x::NextTable) = x
function _convert(::Type{NDSparse}, t::NextTable)
    NDSparse(rows(t, pkeynames(t)), rows(t, excludecols(t, pkeynames(t))),
             copy=false, presorted=true)
end

function _convert(::Type{NextTable}, x::NDSparse)
    convert(NextTable, x.index, x.data;
            perms=x._table.perms,
            presorted=true, copy=false)
end

include("table/query.jl")
include("table/join.jl")

# getindex and setindex!
include("indexing.jl")

# query and aggregate
include("query.jl")

# joins
include("join.jl")

# TableTraits.jl integration
include("tabletraits.jl")

## New table type

end # module
