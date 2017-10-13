export NextTable, colnames

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

    intprimarykey = [colindex(cs, primarykey)...]

    NextTable{typeof(cs)}(cs,
           intprimarykey,
           perms,
           cardinality,
           similar(cs, 0))
end

using Base.Test

function NextTable(cols...;
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

Base.eltype(t::NextTable) = eltype(t.columns)
colnames(t::AbstractIndexedTable) = fieldnames(eltype(t))
colindex(t::NextTable, col) = colindex(t.columns, col)
Base.copy(t::NextTable) = NextTable(t)
Base.:(==)(a::NextTable, b::NextTable) = rows(a) == rows(b)

columns(t::NextTable, args...) = columns(t.columns, args...)
column(t::NextTable, args...) = column(t.columns, args...)
rows(t::NextTable, args...) = rows(t.columns, args...)
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

function pkeynames(t::NextTable)
    if eltype(t) <: NamedTuple
        (colnames(t)[t.primarykey]...)
    else
        (t.primarykey...)
    end
end

primarykeys(t::NextTable) = rows(t, pkeynames(t))

function sortpermby(t::NextTable, by; cache=true)
    canonorder = colindex(t, by)
    canonorder_vec = Int[canonorder...]
    perms = [primaryperm(t), t.perms;]
    matched_cols, partial_perm = best_perm_estimate(perms, canonorder_vec)

    if matched_cols == length(by)
        # first n index columns
        return partial_perm
    end

    bycols = columns(t, canonorder)
    perm = if matched_cols > 0
        nxtcol = bycols[matched_cols+1]
        p = convert(Array{UInt32}, partial_perm)
        refine_perm!(p, bycols, matched_cols,
                     rows(t, canonorder[1:matched_cols]),
                     sortproxy(nxtcol), 1, length(t))
        p
    else
        sortperm(rows(bycols))
    end
    if cache
        push!(t.perms, Perm(canonorder_vec, perm))
    end

    return perm
end

function sortpermby(t::NextTable, by::AbstractArray; cache=true)
    sortperm(by)
end

"""
Returns: (n, perm) where n is the number of columns in
the beginning of `cols`, `perm` is one possible permutation of those
first `n` columns.
"""
function best_perm_estimate(perms, cols)
    bestperm = nothing
    bestmatch = 0
    for p in perms
        matched_cols = 0
        l = min(length(cols), length(p.columns))
        for i in 1:l
            cols[i] != p.columns[i] && break
            matched_cols += 1
        end
        if matched_cols == length(cols)
            return (matched_cols, p.perm)
        elseif bestmatch < matched_cols
            bestmatch = matched_cols
            bestperm = p.perm
        end
    end
    return (bestmatch, bestperm)
end

function convert(::Type{NextTable},
                 key::AbstractVector,
                 val::AbstractVector)

    cs = Columns(concat_tup(columns(key), columns(val)))
    NextTable(cs, primarykey=[1:nfields(eltype(key));])
end
