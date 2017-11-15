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
include("table.jl")
include("ndsparse.jl")

#=
# Poor man's traits

# These support `colnames` and `columns`
const TableTrait = Union{AbstractVector, NextTable, NDSparse}

# These support `colnames`, `columns`,
# `pkeynames`, `permcache`, `cacheperm!`
=#

const Dataset = Union{NextTable, NDSparse}

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

include("sortperm.jl")
include("indexing.jl") # x[y]
include("selection.jl")
include("reduce.jl")
include("flatten.jl")
include("join.jl")

# TableTraits.jl integration
include("tabletraits.jl")

# Plots.jl integration
include("plotrecipes.jl")

end # module
