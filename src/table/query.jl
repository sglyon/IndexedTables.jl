export groupreduce, groupby

"""
    filter(pred, t; select)

Filter rows in `t` according to `pred`. `select` choses the fields that act as input to `pred`.

`pred` can be:

- a function - selected structs or values are passed to this function
- a tuple of column => function pairs: applies to each named column the corresponding function, keeps only rows where all such conditions are satisfied.

```jldoctest
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

"""
`select(t::NextTable, which::DimName...)`

Select a subset of columns.
"""
function Base.select(t::AbstractIndexedTable, which::DimName...)
    ColDict(t)[which]
end

# Filter on data field
function groupreduce_to!(f, key, data, dest_key, dest_data, perm)
    n = length(key)
    i1 = 1
    while i1 <= n
        val = data[perm[i1]]
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            val = f(val, data[perm[i]])
            i += 1
        end
        push!(dest_key, key[perm[i1]])
        push!(dest_data, val)
        i1 = i
    end
    dest_key, dest_data
end

function bestname(t, col, fallback)
    if isa(col, Pair{Symbol, <:Any})
        col[1]
    elseif isa(col, Union{Symbol, Integer}) && eltype(t) <: NamedTuple
        # keep the name of the original column
        name = fieldnames(eltype(t))[colindex(t, col)]
    else
        fallback
    end
end

function groupreduce(f, t::NextTable, by=pkeynames(t);
                     select=rows(t), name=nothing)

    key  = rows(t, by)
    data = rows(t, select)
    perm = sortpermby(t, by)

    dest_key = similar(key, 0)
    dest_data = similar(data, 0)

    groupreduce_to!(f, key, data, dest_key, dest_data, perm)

    if !(typeof(dest_data) <: Columns)
        if name == nothing
            name = bestname(t, select, Symbol(f))
        end
        dest_data = Columns(dest_data, names=[name])
    end

    if !(typeof(dest_key) <: Columns)
        name = bestname(t, by, :key)
        dest_key = Columns(dest_key, names=[name])
    end

    convert(NextTable, dest_key, dest_data)
end

function groupby(f, t::NextTable, by=pkeynames(t); select=rows(t), name=nothing)
    key  = rows(t, by)
    data = rows(t, select)

    perm = sortpermby(t, by)

    if isa(f, AbstractVector)
        T = isa(name, AbstractVector) ?
            namedtuple(name...) : namedtuple(map(Symbol, f)...)

        f = T(f...)
    end

    dest_key, dest_data = _groupby(f, key, data, perm)

    if !(typeof(dest_data) <: Columns)
        if name == nothing
            name = bestname(t, select, Symbol(f))
        end
        dest_data = Columns(dest_data, names=[name])
    end

    if !(typeof(dest_key) <: Columns)
        name = bestname(t, by, :key)
        dest_key = Columns(dest_key, names=[name])
    end

    convert(NextTable, dest_key, dest_data)
end

@inline _apply_many(f::Tup, xs) = map(g->g(xs), f)
@inline _apply_many(f, xs) = f(xs)

function _groupby(f, key, data, perm, dest_key=similar(key,0),
                  dest_data=nothing, i1=1)
    n = length(key)
    while i1 <= n
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            i += 1
        end
        val = _apply_many(f, data[perm[i1:(i-1)]])
        push!(dest_key, key[perm[i1]])
        if dest_data === nothing
            newdata = [val]
            if isa(val, Tup)
                newdata = convert(Columns, newdata)
            end
            return _groupby(f, key, data, perm, dest_key, newdata, i)
        else
            push!(dest_data, val)
        end
        i1 = i
    end
    (dest_key, dest_data===nothing ? Union{}[] : dest_data)
end
