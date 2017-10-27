export NextTable, colnames, columns, reindex, primarykeys

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

subtable(t::NextTable, r) = t[r]

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

function primarykeys(t::NextTable)
    if isempty(t.primarykey)
        Columns(Base.OneTo(length(t)))
    else
        rows(t, pkeynames(t))
    end
end

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

# showing

import Base.Markdown.with_output_format

function showtable(io::IO, t; header=nothing, cnames=colnames(t), divider=nothing, cstyle=[], full=false, ellipsis=:middle)
    height, width = displaysize(io) 
    showrows = height-5 - (header !== nothing)
    n = length(t)
    header !== nothing && println(io, header)
    if full
        rows = [1:n;]
        showrows = n
    else
        if ellipsis == :middle
            lastfew = div(showrows, 2) - 1
            firstfew = showrows - lastfew - 1
            rows = n > showrows ? [1:firstfew; (n-lastfew+1):n] : [1:n;]
        elseif ellipsis == :end
            rows = n == showrows ?
                [1:showrows;] : [1:showrows-1;]
        else
            error("ellipsis must be either :middle or :end")
        end
    end
    nc = length(columns(t))
    reprs  = [ sprint(io->showcompact(io,columns(t)[j][i])) for i in rows, j in 1:nc ]
    strcnames = map(string, cnames)
    widths  = [ max(strwidth(get(strcnames, c, "")), isempty(reprs) ? 0 : maximum(map(strwidth, reprs[:,c]))) for c in 1:nc ]
    if sum(widths) + 2*nc > width
        return showmeta(io, t, cnames)
    end
    for c in 1:nc
        nm = get(strcnames, c, "")
        style = get(cstyle, c, nothing)
        txt = c==nc ? nm : rpad(nm, widths[c]+(c==divider ? 1 : 2), " ")
        if style == nothing
            print(io, txt)
        else
            with_output_format(style, print, io, txt)
        end
        if c == divider
            print(io, "│")
            length(cnames) > divider && print(io, " ")
        end
    end
    println(io)
    if divider !== nothing
        print(io, "─"^(sum(widths[1:divider])+2*divider-1), "┼", "─"^(sum(widths[divider+1:end])+2*(nc-divider)-1))
    else
        print(io, "─"^(sum(widths)+2*nc-2))
    end
    for r in 1:size(reprs,1)
        println(io)
        for c in 1:nc
            print(io, c==nc ? reprs[r,c] : rpad(reprs[r,c], widths[c]+(c==divider ? 1 : 2), " "))
            if c == divider
                print(io, "│ ")
            end
        end
        if n > showrows && ((ellipsis == :middle && r == firstfew) || (ellipsis == :end && r == size(reprs, 1)))
            if divider === nothing
                println(io)
                print(io, "⋮")
            else
                println(io)
                print(io, " "^(sum(widths[1:divider]) + 2*divider-1), "⋮")
            end
        end
    end
end

function showmeta(io, t, cnames)
    nc = length(columns(t))
    println(io, "Columns:")
    metat = Columns(([1:nc;], [Text(string(get(cnames, i, "<noname>"))) for i in 1:nc],
                       eltype.([columns(t)...])))
    showtable(io, metat, cnames=["#", "colname", "type"], cstyle=fill(:bold, nc), full=true)
end

function show(io::IO, t::NextTable{T}) where {T}
    header = "NextTable with $(length(t)) rows, $(length(columns(t))) columns:"
    cstyle = Dict([i=>:bold for i in t.primarykey])
    showtable(io, t, header=header, cstyle=cstyle)
end
