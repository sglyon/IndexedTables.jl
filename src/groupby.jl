using OnlineStatsBase
export groupreduce, groupby

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
`groupreduce(f, t[, by::Selection]; select::Selection, name)`

Group rows by a given key (a [Selection](@ref)).
Apply a function `f` on the rows pair-wise to reduce each group to a single value.
"""
function groupreduce(f, t::NextTable, by=pkeynames(t);
                     select=excludecols(t, by))

    if isa(f, Pair)
        return groupreduce((f,), t, by, select=select)
    end
    if !isa(by, Tuple)
        by=(by,)
    end
    key  = rows(t, by)
    data = rows(t, select)
    perm = sortpermby(t, by)

    dest_key = similar(key, 0)

    fs, input, T = init_arrays(f, data, reduced_type, false)
    dest_data = similar(arrayof(T), 0)

    groupreduce_to!(fs, key, input, dest_key, dest_data, perm)

    convert(NextTable, dest_key, dest_data)
end

## GroupBy

struct SubArrClosure{R}
    r::R
end

(f::SubArrClosure)(x) = SubArray(x, f.r)

function _groupby(f, key, data, perm, dest_key=similar(key,0),
                  dest_data=nothing, i1=1)
    n = length(key)
    cs = columns(data)
    while i1 <= n
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            i += 1
        end
        # needed this hack to avoid allocations. i loses type info
        #val = _apply(f, map(x->SubArray(x, (perm[i1:(i-1)],)), cs))
        val = _apply(f, map(SubArrClosure((perm[i1:(i-1)],)), cs))
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

function groupby(f, t::NextTable, by=pkeynames(t); select=rows(t))

    if !isa(by, Tuple)
        by=(by,) # this will help keep the column name
    end
    if !isa(f, Tup)
        f=(f,)
    end

    key  = rows(t, by)
    data = rows(t, select)

    perm = sortpermby(t, by)
    fs, input, T = init_arrays(f, data, reduced_type, true)
    # Note: we're not using T here, we'll let _groupby figure it out
    dest_key, dest_data = _groupby(fs, key, input, perm)

    convert(NextTable, dest_key, dest_data)
end


## NDSparse

function groupreduce(f, t::NDSparse, by=pkeynames(t);
                     select=valueselector(t), name=nothing)

    key  = rows(t, by)
    data = rows(t, select)
    perm = sortpermby(t, by)

    dest_key = similar(key, 0)
    dest_data = similar(data, 0)

    groupreduce_to!(f, key, data, dest_key, dest_data, perm)

    NDSparse(dest_key, dest_data; presorted=true, copy=false)
end

Base.@deprecate aggregate(f, t;
                          by=pkeynames(t),
                          with=valueselector(t)) groupreduce(f, t, by; select=with)


"""
`groupby(f::Function, x::NDSparse)`

Combine adjacent rows with equal indices using a function from vector to scalar,
e.g. `mean`.
"""
function groupby(f, x::NDSparse, by=pkeynames(x);
                     select=valueselector(x), name=nothing)

    if isa(f, AbstractVector)
        T = isa(name, AbstractVector) ?
            namedtuple(name...) : namedtuple(map(Symbol, f)...)

        f = T(f...)
    end

    perm = sortpermby(x, by)
    idxs, data = _groupby(f, rows(x, by), rows(x, select), perm)
    NDSparse(idxs, data, presorted=true, copy=false)
end

Base.@deprecate aggregate_vec(
    fs, x::NDSparse;
    names=nothing,
    by=pkeynames(x),
    with=valueselector(x)) groupby(fs, x; name=names, select=with)

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
function convertdim(x::NDSparse, d::DimName, xlat; agg=nothing, vecagg=nothing, name=nothing)
    cols = x.index.columns
    d2 = map(xlat, cols[d])
    n = fieldindex(cols, d)
    names = nothing
    if isa(x.index.columns, NamedTuple)
        names = fieldnames(x.index.columns)
        if name !== nothing
            names[n] = name
        end
    end
    if vecagg !== nothing
        y = NDSparse(cols[1:n-1]..., d2, cols[n+1:end]..., x.data, copy=false, names=names)
        idxs, data = _groupby(vecagg, y.index, y.data, Base.OneTo(length(x)))
        return NDSparse(idxs, data, copy=false)
    end
    NDSparse(cols[1:n-1]..., d2, cols[n+1:end]..., x.data, agg=agg, copy=true, names=names)
end

convertdim(x::NDSparse, d::Int, xlat::Dict; agg=nothing, vecagg=nothing, name=nothing) = convertdim(x, d, i->xlat[i], agg=agg, vecagg=vecagg, name=name)

convertdim(x::NDSparse, d::Int, xlat, agg) = convertdim(x, d, xlat, agg=agg)

sum(x::NDSparse) = sum(x.data)

function reducedim(f, x::NDSparse, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    select(x, keep..., agg=f)
end

reducedim(f, x::NDSparse, dims::Symbol) = reducedim(f, x, [dims])

"""
`reducedim_vec(f::Function, arr::NDSparse, dims)`

Like `reducedim`, except uses a function mapping a vector of values to a scalar instead
of a 2-argument scalar function.
"""
function reducedim_vec(f, x::NDSparse, dims; with=valueselector(x))
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    idxs, d = _groupby(f, keys(x, (keep...)), columns(x, with), sortpermby(x, (keep...)))
    NDSparse(idxs, d, presorted=true, copy=false)
end

reducedim_vec(f, x::NDSparse, dims::Symbol) = reducedim_vec(f, x, [dims])
