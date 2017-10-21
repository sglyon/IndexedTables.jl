export NextTable, colnames, columns, reindex

"""
A permutation

Fields:

- `columns`: The columns being indexed as a vector of integers (column numbers)
- `perm`: the permutation - an array or iterator which has the sorted permutation
"""
struct Perm{X}
    columns::Vector{Int}
    perm::X
end

abstract type AbstractIndexedTable end

"""
An indexed table

Fields:

- `columns`: `Columns` object which iterates to give an array of rows
- `perms`: A vector of `Perm` objects
- `cardinality`: Used internally to store what percent of the data in each column is unique
"""
struct NextTable{C<:Columns} <: AbstractIndexedTable
    columns::C
    primarykey::Vector{Int}
    perms::Vector{Perm}
    cardinality::Vector{Nullable{Float64}}

    columns_buffer::C
end

function NextTable(cs::Columns;
                   primarykey=Int[],
                   perms=Perm[],
                   presorted=false,
                   copy=true,
                   cardinality=fill(Nullable{Float64}(), length(columns(cs))))

    if isa(primarykey, Union{Int, Symbol})
        primarykey = [primarykey]
    elseif isa(primarykey, Tuple)
        primarykey = [primarykey...]
    end

    if !presorted && !isempty(primarykey)
        pkey = rows(cs, (primarykey...))
        if !issorted(pkey)
            perm = sortperm(pkey)
            if copy
                cs = cs[perm]
            else
                cs = permute!(cs, perm)
            end
        elseif copy
            cs = Base.copy(cs)
        end
    elseif copy
        cs = Base.copy(cs)
    end

    intprimarykey = map(k->colindex(cs, k), primarykey)

    NextTable{typeof(cs)}(cs,
           intprimarykey,
           perms,
           cardinality,
           similar(cs, 0))
end

using Base.Test

function NextTable(cols::AbstractArray...;
                   names=nothing,
                   kwargs...)

    NextTable(Columns(cols...; names=names); kwargs...)
end

# Easy constructor to create a derivative table
function NextTable(t::NextTable;
                   columns=t.columns,
                   primarykey=t.primarykey,
                   perms=t.perms,
                   cardinality=t.cardinality,
                   presorted=false,
                   copy=true)

    NextTable(columns,
              primarykey=primarykey,
              perms=perms,
              cardinality=cardinality,
              presorted=presorted,
              copy=copy)

end

Base.@pure colnames(t::AbstractIndexedTable) = fieldnames(eltype(t))
columns(t::NextTable) = columns(t.columns)

Base.eltype(t::NextTable) = eltype(t.columns)
Base.copy(t::NextTable) = NextTable(t)
Base.:(==)(a::NextTable, b::NextTable) = rows(a) == rows(b)

Base.getindex(t::NextTable, i::Integer) = getindex(t.columns, i)

Base.length(t::NextTable) = length(t.columns)
Base.start(t::NextTable) = start(t.columns)
Base.next(t::NextTable, i) = next(t.columns, i)
Base.done(t::NextTable, i) = done(t.columns, i)
function getindex(t::NextTable, idxs::AbstractVector{<:Integer})
    if issorted(idxs)
        perms = map(t.perms) do p
            Perm(p.columns, p.perm[idxs])
        end
        NextTable(t, columns=t.columns[idxs], perms=perms, copy=false)
    else
        # this is for gracefully allowing this later
        throw(ArgumentError("`getindex` called with unsorted index. This is not allowed at this time."))
    end
end

function primaryperm(t::NextTable)
    Perm(t.primarykey, Base.OneTo(length(t)))
end

permcache(t::NextTable) = [primaryperm(t), t.perms;]
cacheperm!(t::NextTable, p) = push!(t.perms, p)

function pkeynames(t::NextTable)
    if eltype(t) <: NamedTuple
        (colnames(t)[t.primarykey]...)
    else
        (t.primarykey...)
    end
end

primarykeys(t::NextTable) = rows(t, pkeynames(t))

function excludecols(t::NextTable, cols)
    ns = colnames(t)
    cols = Iterators.filter(c->c in ns || isa(c, Integer),
                  map(x->isa(x, As) ? x.src : x, cols))
    (setdiff(ns, cols)...)
end

function convert(::Type{NextTable},
                 key::AbstractVector,
                 val::AbstractVector; kwargs...)

    cs = Columns(concat_tup(columns(key), columns(val)))
    NextTable(cs, primarykey=[1:ncols(key);]; kwargs...)
end

function reindex(t::NextTable, by=pkeynames(t), select=excludecols(t, by); kwargs...)
    convert(NextTable, rows(t, by), rows(t, select); kwargs...)
end
