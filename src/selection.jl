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
    if !isa(by, Tuple)
        return reindex(T, t, (by,), select; kwargs...)
    end
    if !isa(select, Tuple)
        return reindex(T, t, by, (select,); kwargs...)
    end
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
    fs, input, T = init_inputs(f, rows(t, select), mapped_type, false)
    x = similar(arrayof(T), length(t))
    map!(a->_apply(fs, a), x, input)
    isa(x, Columns) ? table(x) : x
end

"""
`reduce(f, t::Table; select)`

Reduce `t` row-wise using `f`.

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
    fs, input, T = init_reduce(f, rows(t, select), false)
    _reduce(fs, input)
end

function reduce(f, t::NextTable, v0; select=rows(t))
    fs, input, T = init_reduce(f, rows(t, select), false)
    reduce((x,y)->_apply(fs,x,y), input, v0)
end

function _reduce(fs, input)
    acc = init_first(fs, input[1])
    @inbounds @simd for i=2:length(input)
        acc = _apply(fs, acc, input[i])
    end
    acc
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

filt_by_col!(f, col, indxs) = filter!(i->f(col[i]), indxs)

Base.@deprecate select(arr::NDSparse, conditions::Pair...) filter(conditions, arr)
Base.@deprecate select(arr::NDSparse, which::DimName...; agg=nothing) reindex(arr, which; agg=agg)

"""
`filter(pred, t; select)`

Filter rows in `t` according to `pred`. `select` choses the fields that act as input to `pred`.

`pred` can be:

- a function - selected structs or values are passed to this function
- a tuple of column => function pairs: applies to each named column the corresponding function, keeps only rows where all such conditions are satisfied.

```jldoctest
julia> t = table(["a","b","c"], [0.01, 0.05, 0.07], [2,1,0],
                 names=[:n, :t, :x])
Table with 3 rows, 3 columns:
n    t     x
────────────
"a"  0.01  2
"b"  0.05  1
"c"  0.07  0

julia> filter(p->p.x/p.t < 100, t)
Table with 2 rows, 3 columns:
n    t     x
────────────
"b"  0.05  1
"c"  0.07  0

julia> filter(p->p.x/p.t < 100, t, select=(:x,:t))
Table with 2 rows, 3 columns:
n    t     x
────────────
"b"  0.05  1
"c"  0.07  0
```
Although the two examples do the same thing, the second one will allocate structs of only `x` and `y` fields to be passed to the predicate function. This results in better performance because we aren't allocating a struct with a string object.

```jldoctest
julia> filter(iseven, t, select=:x)
Table with 2 rows, 3 columns:
n    t     x
────────────
"a"  0.01  2
"c"  0.07  0

julia> filter((:x=>iseven,), t, select=:x)
Table with 2 rows, 3 columns:
n    t     x
────────────
"a"  0.01  2
"c"  0.07  0
```

Filtering by a single column is convenient.

```jldoctest
julia> filter((:x=>iseven, :t => a->a>0.01), t)
Table with 1 rows, 3 columns:
n    t     x
────────────
"c"  0.07  0

```

"""
function Base.filter(fn, t::Dataset; select=rows(t))
    x = rows(t, select)
    indxs = filter(i->fn(x[i]), eachindex(x))
    t[indxs]
end

function Base.filter(pred::Tuple, t::Dataset; select=values(t))
    indxs = [1:length(t);]
    x = rows(t, select)
    for (c,f) in pred
        filt_by_col!(f, rows(x, c), indxs)
    end
    subtable(t, indxs)
end

function Base.filter(pred::Pair, t::Dataset; select=values(t))
    filter([pred], t, select=select)
end

# Filter on data field
function Base.filter(fn::Function, arr::NDSparse)
    flush!(arr)
    data = arr.data
    indxs = filter(i->fn(data[i]), eachindex(data))
    NDSparse(Columns(map(x->x[indxs], arr.index.columns)), data[indxs], presorted=true)
end

