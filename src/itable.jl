using IndexedTables
import IndexedTables: columns, rows, refine_perm!, sortproxy, flush!

struct Perm{X}
    columns::Vector{Int}
    perm::X
end

struct NextTable{C<:Columns}
    columns::C
    perms::Vector{Perm}
    cardinality::Vector{Nullable{Float64}}

    columns_buffer::C
end

function NextTable(cs::Columns; perms=Perm[])
    NextTable{typeof(cs)}(cs,
           perms,
           fill(Nullable{Float64}(), length(columns(cs))),
           similar(cs, 0))
end

function NextTable(cols...; names=nothing)
    NextTable(Columns(cols...; names=names))
end

columns(t::NextTable, args...) = columns(t.columns, args...)
rows(t::NextTable, args...) = rows(t.columns, args...)
Base.length(t::NextTable) = length(t.columns)
Base.eltype(t::NextTable) = eltype(t.columns)
Base.start(t::NextTable) = start(t.columns)
Base.next(t::NextTable, i) = next(t.columns, i)
Base.done(t::NextTable, i) = done(t.columns, i)
Base.getindex(t::NextTable, i::Integer) = getindex(t.columns, i)
Base.getindex(t::NextTable, i::AbstractArray) = NextTable(getindex(t.columns, i))

function Base.sortperm(t::NextTable, by, cache=true)
    fns = fieldnames(eltype(t))
    canonorder = map(i->colindex(fns, i), by)
    canonorder_vec = Int[canonorder...]
    matched_cols, partial_perm = best_perm_estimate(t.perms,
                                                    canonorder_vec)

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

function colindex(fnames, col)
    if isa(col, Int) && 1 <= col <= length(fnames)
        return col
    elseif isa(col, Symbol)
        idx = findfirst(fnames, col)
        idx > 0 && return idx
    end
    error("column $col not found.")
end
