using OnlineStatsBase
export groupreduce, groupby

"""
`reduce(f, t::Table; select::Selection)`

Reduce `t` by applying `f` pair-wise on values or structs
selected by `select`.

`f` can be:

1. A function
2. An OnlineStat
3. A tuple of functions and/or OnlineStats
4. A named tuple of functions and/or OnlineStats
5. A named tuple of (selector => function or OnlineStat) pairs

```jldoctest
julia> t = table([0.1, 0.5, 0.75], [0,1,2], names=[:t, :x])
Table with 3 rows, 2 columns:
t     x
───────
0.1   0
0.5   1
0.75  2
```

When `f` is a function, it reduces the selection as usual:

```
julia> reduce(+, t, select=:t)
1.35
```

If `select` is omitted, the rows themselves are passed to reduce as tuples.

```jldoctest
julia> reduce((a, b) -> @NT(t=a.t+b.t, x=a.x+b.x), t)
(t = 1.35, x = 3)
```

If `f` is an OnlineStat object from the [OnlineStats](https://github.com/joshday/OnlineStats.jl) package, the statistic is computed on the selection.

```jldoctest
julia> using OnlineStats

julia> reduce(Mean(), t, select=:t)
▦ Series{0,Tuple{Mean},EqualWeight}
┣━━ EqualWeight(nobs = 3)
┗━━━┓
    ┗━━ Mean(0.45)
```

# Reducing with multiple functions

Often one needs many aggregate values from a table. This is when `f` can be passed as a tuple of functions:

```jldoctest
julia> y = reduce((min, max), t, select=:x)
(min = 0, max = 2)

julia> y.max
2

julia> y.min
0
```

Note that the return value of invoking reduce with a tuple of functions
will be a named tuple which has the function names as the keys. In the example, we reduced using `min` and `max` functions to obtain the minimum and maximum values in column `x`.

If you want to give a different name to the fields in the output, use a named tuple as `f` instead:

```jldoctest
julia> y = reduce(@NT(sum=+, prod=*), t, select=:x)
(sum = 3, prod = 0)
```

You can also compute many OnlineStats by passing tuple or named tuple of OnlineStat objects as the reducer.

```jldoctest
julia> y = reduce((Mean(), Variance()), t, select=:t)
(Mean = ▦ Series{0,Tuple{Mean},EqualWeight}
┣━━ EqualWeight(nobs = 3)
┗━━━┓
    ┗━━ Mean(0.45), Variance = ▦ Series{0,Tuple{Variance},EqualWeight}
┣━━ EqualWeight(nobs = 3)
┗━━━┓
    ┗━━ Variance(0.1075))

julia> y.Mean
▦ Series{0,Tuple{Mean},EqualWeight}
┣━━ EqualWeight(nobs = 3)
┗━━━┓
    ┗━━ Mean(0.45)

julia> y.Variance
▦ Series{0,Tuple{Variance},EqualWeight}
┣━━ EqualWeight(nobs = 3)
┗━━━┓
    ┗━━ Variance(0.1075)
```

# Combining reduction and selection

In the above section where we computed many reduced values at once, we have been using the same selection for all reducers, that specified by `select`. It's possible to select different inputs for different reducers by using a named tuple of `slector => function` pairs:

```jldoctest
julia> reduce(@NT(xsum=:x=>+, negtsum=(:t=>-)=>+), t)
(xsum = 3, negtsum = -1.35)

```

See [`Selection`](@ref) for more on what selectors can be specified. Here since each output can select its own input, `select` keyword is unsually unnecessary. If specified, the slections in the reducer tuple will be done over the result of selecting with the `select` argument.

"""
function reduce(f, t::Dataset; select=valuenames(t))
    fs, input, T = init_inputs(f, rows(t, select), reduced_type, false)
    _reduce(fs, input)
end

function reduce(f, t::Dataset, v0; select=valuenames(t))
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

## groupreduce

function groupreduce_to!(f, key, data, dest_key, dest_data, perm)
    n = length(key)
    i1 = 1
    while i1 <= n
        val = init_first(f, data[perm[i1]])
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            _apply(f, val, data[perm[i]])
            val = _apply(f, val, data[perm[i]])
            i += 1
        end
        push!(dest_key, key[perm[i1]])
        push!(dest_data, val)
        i1 = i
    end
    dest_key, dest_data
end

"""
`groupreduce(f, t[, by::Selection]; select::Selection)`

Group rows by `by`, and apply `f` to reduce each group. `f` can be a function, OnlineStat or a struct of these as described in [`reduce`](@ref). Recommended: see documentation for [`reduce`](@ref) first. The result of reducing each group is put in a table keyed by unique `by` values, the names of the output columns are the same as the names of the fields of the reduced tuples.

# Examples

```jldoctest
julia> groupreduce(+, t, :x, select=:z)
Table with 2 rows, 2 columns:
x  +
─────
1  6
2  15

julia> groupreduce(+, t, (:x, :y), select=:z)
Table with 4 rows, 3 columns:
x  y  +
────────
1  1  3
1  2  3
2  1  11
2  2  4

julia> groupreduce((+, min, max), t, (:x, :y), select=:z)
Table with 4 rows, 5 columns:
x  y  +   min  max
──────────────────
1  1  3   1    2
1  2  3   3    3
2  1  11  5    6
2  2  4   4    4
```

If `f` is a single function or a tuple of functions, the output columns will be named the same as the functions themselves. To change the name, pass a named tuple:

```jldoctest
julia> groupreduce(@NT(zsum=+, zmin=min, zmax=max), t, (:x, :y), select=:z)
Table with 4 rows, 5 columns:
x  y  zsum  zmin  zmax
──────────────────────
1  1  3     1     2
1  2  3     3     3
2  1  11    5     6
2  2  4     4     4
```

Finally, it's possible to select different inputs for different reducers by using a named tuple of `slector => function` pairs:

```jldoctest
julia> groupreduce(@NT(xsum=:x=>+, negysum=(:y=>-)=>+), t, :x)
Table with 2 rows, 3 columns:
x  xsum  negysum
────────────────
1  3     -4
2  6     -4

```

"""
function groupreduce(f, t::Dataset, by=pkeynames(t); select=valuenames(t))
    if typeof(t)<:NextTable && !isa(f, Tup)
        return groupreduce((f,), t, by, select=select)
    end
    if typeof(t)<:NextTable && !isa(by, Tuple)
        by=(by,)
    end
    key  = rows(t, by)
    data = rows(t, select)
    perm = sortpermby(t, by)

    dest_key = similar(key, 0)

    fs, input, T = init_inputs(f, data, reduced_type, false)
    dest_data = similar(arrayof(T), 0)

    groupreduce_to!(fs, key, input, dest_key, dest_data, perm)

    convert(collectiontype(t), dest_key, dest_data,
            presorted=true, copy=false)
end

## GroupBy

struct SubArrClosure{R}
    r::R
end

(f::SubArrClosure)(x) = SubArray(x, f.r)

function _groupby(f, key, data, perm, dest_key=similar(key,0),
                  dest_data=nothing, i1=1)
    n = length(key)
    cs = f isa Tup ? columns(data) : data
    while i1 <= n
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            i += 1
        end
        # needed this hack to avoid allocations. i loses type info
        #val = _apply(f, map(x->SubArray(x, (perm[i1:(i-1)],)), cs))
        if isa(cs, Tup)
            val = _apply(f, map(SubArrClosure((perm[i1:(i-1)],)), cs))
        else
            val = _apply(f, SubArray(cs, (perm[i1:(i-1)],)))
        end

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

collectiontype(::Type{<:NDSparse}) = NDSparse
collectiontype(::Type{<:NextTable}) = NextTable
collectiontype(t::Dataset) = collectiontype(typeof(t))

function groupby(f, t::Dataset, by=pkeynames(t); select=valuenames(t))
    # we want to try and keep the column names
    if typeof(t)<:NextTable && !isa(by, Tuple)
        by=(by,)
    end
    if typeof(t)<:NextTable && !isa(f, Tup)
        f=(f,)
    end

    key  = rows(t, by)
    data = rows(t, select)

    perm = sortpermby(t, by)
    fs, input, S = init_inputs(f, data, reduced_type, true)
    # Note: we're not using S here, we'll let _groupby figure it out
    dest_key, dest_data = _groupby(fs, key, input, perm)

    convert(collectiontype(t), dest_key, dest_data, presorted=true, copy=false)
end

Base.@deprecate aggregate(f, t;
                          by=pkeynames(t),
                          with=valuenames(t)) groupreduce(f, t, by; select=with)


Base.@deprecate aggregate_vec(
    fs::Function, x::NDSparse;
    names=nothing,
    by=pkeynames(x),
    with=valuenames(x)) groupby(names === nothing ? fs : (names => fs,), x; select=with)

Base.@deprecate aggregate_vec(
    fs::AbstractVector, x::NDSparse;
    names=nothing,
    by=pkeynames(x),
    with=valuenames(x)) groupby(names === nothing ? (fs...) : (map(=>, names, fs)...,), x; select=with)

Base.@deprecate aggregate_vec(t::NDSparse; funs...) groupby(namedtuple(first.(funs)...)(last.(funs)...), t)


"""
`convertdim(x::NDSparse, d::DimName, xlate; agg::Function, vecagg::Function, name)`

Apply function or dictionary `xlate` to each index in the specified dimension.
If the mapping is many-to-one, `agg` or `vecagg` is used to aggregate the results.
If `agg` is passed, it is used as a 2-argument reduction function over the data.
If `vecagg` is passed, it is used as a vector-to-scalar function to aggregate
the data.
`name` optionally specifies a new name for the translated dimension.
"""
function convertdim(x::NDSparse, d::DimName, xlat; agg=nothing, vecagg=nothing, name=nothing, select=valuenames(x))
    ks = setcol(pkeys(x), d, d=>xlat)
    if name !== nothing
        ks = renamecol(ks, d, name)
    end

    if vecagg !== nothing
        y = convert(NDSparse, ks, rows(x, select))
        return groupby(vecagg, y)
    end

    if agg !== nothing
        return convert(NDSparse, ks, rows(x, select), agg=agg)
    end
    convert(NDSparse, ks, rows(x, select))
end

convertdim(x::NDSparse, d::Int, xlat::Dict; agg=nothing, vecagg=nothing, name=nothing, select=valuenames(x)) = convertdim(x, d, i->xlat[i], agg=agg, vecagg=vecagg, name=name, select=select)

convertdim(x::NDSparse, d::Int, xlat, agg) = convertdim(x, d, xlat, agg=agg)

sum(x::NDSparse) = sum(x.data)

function reducedim(f, x::NDSparse, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    selectkeys(x, (keep...), agg=f)
end

reducedim(f, x::NDSparse, dims::Symbol) = reducedim(f, x, [dims])

"""
`reducedim_vec(f::Function, arr::NDSparse, dims)`

Like `reducedim`, except uses a function mapping a vector of values to a scalar instead
of a 2-argument scalar function.
"""
function reducedim_vec(f, x::NDSparse, dims; with=valuenames(x))
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    idxs, d = _groupby(f, keys(x, (keep...)), columns(x, with), sortpermby(x, (keep...)))
    NDSparse(idxs, d, presorted=true, copy=false)
end

reducedim_vec(f, x::NDSparse, dims::Symbol) = reducedim_vec(f, x, [dims])
