import Base: setindex!, reduce
import DataValues: dropna
export NextTable, table, colnames, pkeynames, columns, pkeys, reindex, dropna

"""
A permutation

# Fields:

- `columns`: The columns being indexed as a vector of integers (column numbers)
- `perm`: the permutation - an array or iterator which has the sorted permutation
"""
struct Perm{X}
    columns::Vector{Int}
    perm::X
end

abstract type AbstractIndexedTable end

"""
A table.
"""
struct NextTable{C<:Columns} <: AbstractIndexedTable
    # `Columns` object which iterates to give an array of rows
    columns::C
    # columns that are primary keys (Vector{Int})
    pkey::Vector{Int}
    # Cache permutations by various subsets of columns
    perms::Vector{Perm}
    # store what percent of the data in each column is unique
    cardinality::Vector{Nullable{Float64}}

    columns_buffer::C
end

"""
`table(cols::AbstractVector...; names, pkey, presorted, copy, chunks)`

Create a table with columns given by `cols`.
```jldoctest
julia> a = table([1,2,3], [4,5,6])
Table with 3 rows, 2 columns:
1  2
────
1  4
2  5
3  6
```

`names` specify names for columns. If specified, the table will be an iterator of named tuples.

```jldoctest
julia> b = table([1,2,3], [4,5,6], names=[:x, :y])
Table with 3 rows, 2 columns:
x  y
────
1  4
2  5
3  6

```

`table(cols::Union{Tuple, NamedTuple}; pkey, presorted, copy, chunks)`

```jldoctest
julia> table(([1,2,3], [4,5,6])) == a
true

julia> table(@NT(x=[1,2,3], y=[4,5,6])) == b
true
```

Convert a struct of columns to a table of structs.

`table(cols::Columns; pkey, presorted, copy, chunks)`

Construct a table from a vector of tuples. See [`rows`](@ref) for constructing `Columns` object.

```jldoctest
julia> table(Columns([1,2,3], [4,5,6])) == a
true

julia> table(Columns(x=[1,2,3], y=[4,5,6])) == b
true
```

`table(t::Union{Table, NDSparse}; pkey=pkeynames(t), presorted=true, copy, chunks)`

Copy a Table or NDSparse to create a new table. The same primary keys as the input are used.

```jldoctest
julia> b == table(b)
true
```


# Arguments:

- `pkey`: select columns to act as the primary key. By default, no columns are used as primary key.
- `presorted`: is the data pre-sorted by primary key columns? If so, skip sorting. `false` by default. Irrelevant if `chunks` is specified.
- `copy`: creates a copy of the input vectors if `true`. `true` by default. Irrelavant if `chunks` is specified.
- `chunks`: distribute the table into `chunks` (Integer) chunks (a safe bet is nworkers()). Table is not distributed by default. See [Distributed](@distributed) docs.

# Examples:

Specifying `pkey` will cause the table to be sorted by the columns named in pkey:

```jldoctest
julia> b = table([2,3,1], [4,5,6], names=[:x, :y], pkey=:x)
Table with 3 rows, 2 columns:
x  y
────
1  6
2  4
3  5

julia> b = table([2,1,2,1],[2,3,1,3],[4,5,6,7], names=[:x, :y, :z], pkey=(:x,:y))
Table with 4 rows, 3 columns:
x  y  z
───────
1  3  5
1  3  7
2  1  6
2  2  4
```
Note that the keys do not have to be unique.

`chunks` attribute allows you to create a distributed table. Note this argument needs the JuliaDB package to be loaded.

```jldoctest
julia> t = table([2,3,1,4], [4,5,6,7], names=[:x, :y], pkey=:x, chunks=2)
Distributed Table with 4 rows in 2 chunks:
x  y
────
1  6
2  4
3  5
4  7
```

A distributed table will be constructed if one of the arrays passed into `table` constructor is a distributed
array. A distributed Array can be constructed using `distribute`:

```jldoctest

julia> x = distribute([1,2,3,4], 2);

julia> t = table(x, [5,6,7,8], names=[:x,:y])
Distributed Table with 4 rows in 2 chunks:
x  y
────
1  5
2  6
3  7
4  8

julia> table(columns(t)..., [9,10,11,12], names=[:x,:y,:z])
Distributed Table with 4 rows in 2 chunks:
x  y  z
────────
1  5  9
2  6  10
3  7  11
4  8  12

```

Distribution is done to match the first distributed column from left to right. Specify `chunks` to override this.
"""
function table end

function table(::Val{:serial}, cols::Tup;
               pkey=Int[],
               chunks=nothing, # unused here
               perms=Perm[],
               presorted=false,
               copy=true,
               cardinality=fill(Nullable{Float64}(), length(cols)))

    cs = rows(cols)

    if isa(pkey, Union{Int, Symbol})
        pkey = [pkey]
    elseif isa(pkey, Tuple)
        pkey = [pkey...]
    end

    if !presorted && !isempty(pkey)
        pkeys = rows(cs, (pkey...))
        if !issorted(pkeys)
            perm = sortperm(pkeys)
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

    intpkey = map(k->colindex(cs, k), pkey)

    NextTable{typeof(cs)}(cs,
           intpkey,
           perms,
           cardinality,
           similar(cs, 0))
end

function table{impl}(::Val{impl}, cols; kwargs...)
    if impl == :distributed && isa(cols, Tup)
        error("""You requested to create a distributed table.
                 Distributed table is implemented by JuliaDB.
                 run `using JuliaDB` and try again.""")
    else
        error("unknown table implementation invoked")
    end
end

# detect if a distributed table has to be constructed.
_impl(impl::Val) = impl
_impl(impl::Val, x::AbstractArray, z...) = _impl(impl, z...)
_impl(x::AbstractArray...) = _impl(Val{:serial}(), x...)

function table(cs::Tup; chunks=nothing, kwargs...)
    if chunks !== nothing
        impl = Val{:distributed}()
    else
        impl = _impl(astuple(cs)...)
    end
    table(impl, cs; chunks=chunks, kwargs...)
end

table(cs::Columns; kwargs...) = table(columns(cs); kwargs...)

function table(cols::AbstractArray...; names=nothing, kwargs...)
    if isa(names, AbstractArray) && all(x->isa(x, Symbol), names)
        cs = namedtuple(names...)(cols...)
    else
        cs = cols
    end
    table(cs; kwargs...)
end

# Easy constructor to create a derivative table
function table(t::NextTable;
               columns=t.columns,
               pkey=t.pkey,
               perms=t.perms,
               cardinality=t.cardinality,
               presorted=false,
               copy=true)

    table(columns,
          pkey=pkey,
          perms=perms,
          cardinality=cardinality,
          presorted=presorted,
          copy=copy)
end

Base.@pure colnames(t::AbstractIndexedTable) = fieldnames(eltype(t))
columns(t::NextTable) = columns(t.columns)

Base.eltype(::Type{NextTable{C}}) where {C} = eltype(C)
Base.eltype(t::NextTable) = eltype(typeof(t))
Base.copy(t::NextTable) = NextTable(t)
Base.:(==)(a::NextTable, b::NextTable) = rows(a) == rows(b)

Base.getindex(t::NextTable, i::Integer) = getindex(t.columns, i)

Base.length(t::NextTable) = length(t.columns)
Base.start(t::NextTable) = start(t.columns)
Base.next(t::NextTable, i) = next(t.columns, i)
Base.done(t::NextTable, i) = done(t.columns, i)
function getindex(t::NextTable, idxs::AbstractVector{<:Integer})
    if issorted(idxs)
       #perms = map(t.perms) do p
       #    # TODO: make the ranks continuous
       #    Perm(p.columns, p.perm[idxs])
       #end
        perms = Perm[]
        table(t, columns=t.columns[idxs], perms=perms, copy=false)
    else
        # this is for gracefully allowing this later
        throw(ArgumentError("`getindex` called with unsorted index. This is not allowed at this time."))
    end
end

function Base.getindex(d::ColDict{<:AbstractIndexedTable}, key::Tuple)
    t = d[]
    idx = [colindex(t, k) for k in key]
    pkey = Int[]
    for (i, pk) in enumerate(t.pkey)
        j = findfirst(idx, pk)
        if j > 0
            push!(pkey, j)
        end
    end
    table(d.src, columns=columns(t, key), pkey=pkey)
end

ColDict(t::AbstractIndexedTable) = ColDict(copy(t.pkey), t,
                                copy(colnames(t)), Any[columns(t)...])

function Base.getindex(d::ColDict{<:AbstractIndexedTable})
    table(d.columns...; names=d.names, pkey=d.pkey)
end

subtable(t::NextTable, r) = t[r]

function primaryperm(t::NextTable)
    Perm(t.pkey, Base.OneTo(length(t)))
end

permcache(t::NextTable) = [primaryperm(t), t.perms;]
cacheperm!(t::NextTable, p) = push!(t.perms, p)

"""
`pkeynames(t::Table)`

Names of the primary key columns in `t`.

# Example

```jldoctest

julia> t = table([1,2], [3,4]);

julia> pkeynames(t)
()

julia> t = table([1,2], [3,4], pkey=1);

julia> pkeynames(t)
(1,)

julia> t = table([2,1],[1,3],[4,5], names=[:x,:y,:z], pkey=(1,2));

julia> pkeys(t)
2-element IndexedTables.Columns{NamedTuples._NT_x_y{Int64,Int64},NamedTuples._NT_x_y{Array{Int64,1},Array{Int64,1}}}:
 (x = 1, y = 3)
 (x = 2, y = 1)

```
"""
function pkeynames(t::AbstractIndexedTable)
    if eltype(t) <: NamedTuple
        (colnames(t)[t.pkey]...)
    else
        (t.pkey...)
    end
end

"""
    pkeys(itr::Table)

Primary keys of the table. If Table doesn't have any designated
primary key columns (constructed without `pkey` argument) then
a default key of tuples `(1,):(n,)` is generated.

# Example

```jldoctest

julia> a = table(["a","b"], [3,4]) # no pkey
Table with 2 rows, 2 columns:
1    2
──────
"a"  3
"b"  4

julia> pkeys(a)
2-element Columns{Tuple{Int64}}:
 (1,)
 (2,)

julia> a = table(["a","b"], [3,4], pkey=1)
Table with 2 rows, 2 columns:
1    2
──────
"a"  3
"b"  4

julia> pkeys(a)
2-element Columns{Tuple{String}}:
 ("a",)
 ("b",)
```

"""
function pkeys(t::NextTable)
    if isempty(t.pkey)
        Columns(Base.OneTo(length(t)))
    else
        rows(t, pkeynames(t))
    end
end

Base.values(t::NextTable) = rows(t)

"""
    excludecols(itr, cols)

Names of all columns in `itr` except `cols`. `itr` can be any of
`Table`, `NDSparse`, `Columns`, or `AbstractVector`

```jldoctest
julia> t = table([2,1],[1,3],[4,5], names=[:x,:y,:z], pkey=(1,2))
Table with 2 rows, 3 columns:
x  y  z
───────
1  3  5
2  1  4

julia> excludecols(t, (:x,))
(:y, :z)

julia> excludecols(t, (2,))
(:x, :z)

julia> excludecols(t, pkeynames(t))
(:z,)

julia> excludecols([1,2,3], (1,))
()

```
"""
function excludecols(t, cols)
    ns = colnames(t)
    mask = ones(Bool, length(ns))
    for c in cols
        i = colindex(t, c)
        if i !== 0
            mask[i] = false
        end
    end
    (ns[mask]...)
end

"""
    convert(NextTable, pkeys, vals; kwargs...)

Construct a table with `pkeys` as primary keys and `vals` as corresponding non-indexed items.
keyword arguments will be forwarded to [`table`](@ref) constructor.

# Example

```jldoctest
julia> convert(NextTable, Columns(x=[1,2],y=[3,4]), Columns(z=[1,2]), presorted=true)
Table with 2 rows, 3 columns:
x  y  z
───────
1  3  1
2  4  2
```
"""
function convert(::Type{NextTable}, key, val; kwargs...)
    cs = Columns(concat_tup(columns(key), columns(val)))
    table(cs, pkey=[1:ncols(key);]; kwargs...)
end

"""
`select(t::Table, which::Selection)`

Select a single column or a subset of columns.

`Selection` is a type union of many types that can select from a table. It can be:

1. `Integer` -- returns the column at this position.
2. `Symbol` -- returns the column with this name.
3. `Pair{Selection => Function}` -- selects and maps a function over the selection, returns the result.
4. `AbstractArray` -- returns the array itself. This must be the same length as the table.
5. `Tuple` of `Selection` -- returns a table containing a column for every selector in the tuple. The tuple may also contain the type `Pair{Symbol, Selection}`, which the selection a name. The most useful form of this when introducing a new column.

# Examples:

Selection with `Integer` -- returns the column at this position.

```jldoctest
julia> tbl = table([0.01, 0.05], [2,1], [3,4], names=[:t, :x, :y], pkey=:t)
Table with 2 rows, 3 columns:
t     x  y
──────────
0.01  2  3
0.05  1  4

julia> select(tbl, 2)
2-element Array{Int64,1}:
 2
 1

```

Selection with `Symbol` -- returns the column with this name.

```jldoctest
julia> select(tbl, :t)
2-element Array{Float64,1}:
 0.01
 0.05

```

Selection with `Pair{Selection => Function}` -- selects some columns and maps a function over it, then returns the mapped column.

```jldoctest
julia> select(tbl, :t=>t->1/t)
2-element Array{Float64,1}:
 100.0
  20.0

julia> vx = select(tbl, (:x, :t)=>p->p.x/p.t) # see selection with Tuple
2-element Array{Float64,1}:
 200.0
  20.0

```

Selection with `AbstractArray` -- returns the array itself.

```jldoctest
julia> select(tbl, [3,4])
2-element Array{Int64,1}:
 3
 4

```
Selection with `Tuple`-- returns a table containing a column for every selector in the tuple.

```jldoctest
julia> select(tbl, (2,1))
Table with 2 rows, 2 columns:
x  t
───────
2  0.01
1  0.05

julia> select(tbl, (:x,:t=>-))
Table with 2 rows, 2 columns:
x  t
────────
1  -0.05
2  -0.01
```

Note that since `tbl` was initialized with `t` as the primary key column, selections that retain the
key column will retain its status as a key. The same applies when multiple key columns are selected.

Selection with a custom array in the tuple will cause the name of the columns to be removed and replaced with integers.

```jldoctest
julia> select(tbl, (:x, :t, [3,4]))
Table with 2 rows, 3 columns:
1  2     3
──────────
2  0.01  3
1  0.05  4
```

This is because the third column's name is unknown. In general if a column's name cannot be determined, then selection
returns an iterable of tuples rather than named tuples. In other words, it strips column names.

To specify a new name to a custom column, you can use `Symbol => Selection` selector.

```jldoctest
julia> select(tbl, (:x,:t,:z=>[3,4]))
Table with 2 rows, 3 columns:
x  t     z
──────────
2  0.01  3
1  0.05  4

julia> select(tbl, (:x, :t, :minust=>:t=>-))
Table with 2 rows, 3 columns:
x  t     minust
───────────────
2  0.01  -0.01
1  0.05  -0.05

julia> select(tbl, (:x, :t, :vx=>(:x,:t)=>p->p.x/p.t))
Table with 2 rows, 3 columns:
x  t     vx
──────────────
2  0.01  200.0
1  0.05  20.0

```
"""
function Base.select(t::AbstractIndexedTable, which)
    ColDict(t)[which]
end

"""
`reindex(itr, by[, select])`

Reindex `itr` (a Table or NDSparse) by columns selected in `by`.
Keeps columns selected by `select` columns as non-indexed columns.
By default all columns not mentioned in `by` are kept.

```jldoctest

julia> t = table([2,1],[1,3],[4,5], names=[:x,:y,:z], pkey=(1,2))

julia> reindex(t, (:y, :z))
Table with 2 rows, 3 columns:
y  z  x
───────
1  4  2
3  5  1

julia> pkeynames(t)
(:y, :z)

julia> reindex(t, (:w=>[4,5], :z))
Table with 2 rows, 4 columns:
w  z  x  y
──────────
4  5  1  3
5  4  2  1

julia> pkeynames(t)
(:w, :z)

```
"""
function reindex end

function reindex(T::Type, t, by, select; kwargs...)
    perm = sortpermby(t, by)
    if isa(perm, Base.OneTo)
        convert(T, rows(t, by), rows(t, select); presorted=true, kwargs...)
    else
        convert(T, rows(t, by)[perm], rows(t, select)[perm]; presorted=true, copy=false, kwargs...)
    end
end

function reindex(t::NextTable, by=pkeynames(t), select=excludecols(t, by); kwargs...)
    reindex(NextTable, t, by, select; kwargs...)
end

canonname(t, x::Symbol) = x
canonname(t, x::Int) = colnames(t)[colindex(t, x)]

"""
`map(f, t::Table; select)`

Apply `f` to every row in `t`. `select` selects fields
passed to `f`.

Returns a new table if `f` returns a tuple or named tuple.
If not, returns a vector.

# Examples

```jldoctest
julia> t = table([0.01, 0.05], [1,2], [3,4], names=[:t, :x, :y])
Table with 2 rows, 3 columns:
t     x  y
──────────
0.01  1  3
0.05  2  4

julia> manh = map(row->row.x + row.y, t)
2-element Array{Int64,1}:
 4
 6

julia> polar = map(p->@NT(r=hypot(p.x + p.y), θ=atan2(p.y, p.x)), t)
Table with 2 rows, 2 columns:
r    θ
────────────
4.0  1.24905
6.0  1.10715

```

`select` argument selects a subset of columns while iterating.

```jldoctest

julia> vx = map(row->row.x/row.t, t, select=(:t,:x)) # row only cotains t and x
2-element Array{Float64,1}:
 100.0
  40.0

julia> map(sin, polar, select=:θ)
2-element Array{Float64,1}:
 0.948683
 0.894427

```
"""
function map(f, t::AbstractIndexedTable; select=rows(t)) end

function map(f, t::NextTable; select=rows(t))
    d = rows(t, select)
    T = _promote_op(f, eltype(d))
    x = similar(arrayof(T), length(t))
    map!(f, x, d)
    isa(x, Columns) ? table(x) : x
end

using OnlineStatsBase

"""
`reduce(f, t::Table; select)`

Reduce `t` row-wise using `f`. Equivalent to `reduce(f, rows(t, select))`

```jldoctest
julia> t = table([0.1, 0.5], [2,1], names=[:t, :x])
Table with 2 rows, 2 columns:
t    x
──────
0.1  2
0.5  1

julia> reduce((y, x) ->map(+, y, x), t)
(t = 0.6, x = 3)

julia> reduce(+, t, select=:t)
0.6
```

If you pass an OnlineStat object from the [OnlineStats](https://github.com/joshday/OnlineStats.jl) package,
the statistic is computed.

```jldoctest
julia> using OnlineStats

julia> reduce(Mean(), t, select=:t)
▦ Series{0,Tuple{Mean},EqualWeight}
┣━━ EqualWeight(nobs = 2)
┗━━━┓
    ┗━━ Mean(0.3)
```
"""
function reduce(f, t::NextTable; select=rows(t))
    reduce(f, rows(t, select))
end

function reduce(f, t::NextTable, v0; select=rows(t))
    reduce(f, rows(t, select), v0)
end

function reduce(f::OnlineStat, t::NextTable; select=rows(t))
    Series(columns(t, select), f)
end

function reduce(f::OnlineStat, t::NextTable, v0; select=rows(t))
    merge(v0, Series(columns(t, select), f))
end

function _nonna(t::Union{Columns, NextTable}, by=(colnames(t)...))
    indxs = [1:length(t);]
    if !isa(by, Tuple)
        by = (by,)
    end
    bycols = columns(t, by)
    d = ColDict(t)
    for (key, c) in zip(by, bycols)
        x = rows(t, c)
        filt_by_col!(!isnull, x, indxs)
        if isa(x, Array{<:DataValue})
            y = Array{eltype(eltype(x))}(length(x))
            y[indxs] = map(get, x[indxs])
            x = y
        elseif isa(x, DataValueArray)
            x = x.values # unsafe unwrap
        end
        d[key] = x
    end
    (d[], indxs)
end

"""
`dropna(t[, select])`

Drop rows which contain NA values.

```jldoctest
julia> t = table([0.1, 0.5, NA,0.7], [2,NA,4,5], [NA,6,NA,7],
                  names=[:t,:x,:y])
Table with 4 rows, 3 columns:
t    x    y
─────────────
0.1  2    #NA
0.5  #NA  6
#NA  4    #NA
0.7  5    7

julia> dropna(t)
Table with 1 rows, 3 columns:
t    x  y
─────────
0.7  5  7
```
Optionally `select` can be speicified to limit columns to look for NAs in.

```jldoctest

julia> dropna(t, :y)
Table with 2 rows, 3 columns:
t    x    y
───────────
0.5  #NA  6
0.7  5    7

julia> t1 = dropna(t, (:t, :x))
Table with 2 rows, 3 columns:
t    x  y
───────────
0.1  2  #NA
0.7  5  7
```

Any columns whose NA rows have been dropped will be converted
to non-na array type. In our last example, columns `t` and `x`
got converted from `Array{DataValue{Int}}` to `Array{Int}`.
Similarly if the vectors are of type `DataValueArray{T}`
(default for `loadtable`) they will be converted to `Array{T}`.
```julia
julia> typeof(column(dropna(t,:x), :x))
Array{Int64,1}
```
"""
function dropna(t::Union{Columns, NextTable}, by=(colnames(t)...))
    subtable(_nonna(t, by)...)
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
            lst = n == showrows ?
                showrows : showrows-1 # make space for ellipse
            rows = [1:min(length(t), showrows);]
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
    header = "Table with $(length(t)) rows, $(length(columns(t))) columns:"
    cstyle = Dict([i=>:bold for i in t.pkey])
    showtable(io, t, header=header, cstyle=cstyle)
end

function Base.merge(a::NextTable, b::NextTable)
    @assert colnames(a) == colnames(b)
    @assert a.pkey == b.pkey
    table(map(vcat, columns(a), columns(b)), pkey=a.pkey, copy=false)
end
