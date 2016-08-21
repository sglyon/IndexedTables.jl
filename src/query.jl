filt_by_col!(f, col, indxs) = filter!(i->f(col[i]), indxs)

"""
`select(arr::NDSparse, conditions::Pair...)`

Filter based on index columns. Conditions are accepted as column-function pairs.

Example: `select(arr, 1 => x->x>10, 3 => x->x!=10 ...)`
"""
function Base.select(arr::NDSparse, conditions::Pair...)
    flush!(arr)
    indxs = [1:length(arr);]
    cols = arr.index.columns
    for (c,f) in conditions
        filt_by_col!(f, cols[c], indxs)
    end
    NDSparse(Columns(map(x->x[indxs], cols)), arr.data[indxs], presorted=true)
end

"""
`select(arr:NDSparse, which::DimName...; agg::Function)`

Select a subset of index columns. If the resulting array has duplicate index entries,
`agg` is used to combine the values.
"""
function Base.select(arr::NDSparse, which::DimName...; agg=nothing)
    flush!(arr)
    NDSparse(Columns(arr.index.columns[[which...]]), arr.data, agg=agg, copy=true)
end

# Filter on data field
function Base.filter(fn::Function, arr::NDSparse)
    flush!(arr)
    data = arr.data
    indxs = filter(i->fn(data[i]), eachindex(data))
    NDSparse(Columns(map(x->x[indxs], arr.index.columns)), data[indxs], presorted=true)
end

# aggregation

"""
`aggregate!(f::Function, arr::NDSparse)`

Combine adjacent rows with equal indices using the given 2-argument reduction function.
"""
function aggregate!(f, x::NDSparse)
    idxs, data = x.index, x.data
    n = length(idxs)
    newlen = 0
    i1 = 1
    while i1 <= n
        val = data[i1]
        i = i1+1
        while i <= n && roweq(idxs, i, i1)
            val = f(val, data[i])
            i += 1
        end
        newlen += 1
        if newlen != i1
            copyrow!(idxs, newlen, i1)
        end
        data[newlen] = val
        i1 = i
    end
    resize!(idxs, newlen)
    resize!(data, newlen)
    x
end

# the same, except returns a new vector where each element is computed
# by applying `f` to a vector of all values associated with equal indexes.
function vec_aggregate!(f, idxs::Columns, data)
    n = length(idxs)
    local newdata
    newlen = 0
    i1 = 1
    while i1 <= n
        i = i1+1
        while i <= n && roweq(idxs, i, i1)
            i += 1
        end
        val = f(data[i1:(i-1)])
        if newlen == 0
            newdata = [val]
        else
            push!(newdata, val)
        end
        newlen += 1
        if newlen != i1
            copyrow!(idxs, newlen, i1)
        end
        i1 = i
    end
    resize!(idxs, newlen)
    newlen==0 ? Union{}[] : newdata
end

"""
`convertdim(x::NDSparse, d::DimName, xlate; agg::Function)`

Apply function or dictionary `xlate` to each index in the specified dimension.
If the mapping is many-to-one, `agg` is used to aggregate the results.
"""
function convertdim(x::NDSparse, d::DimName, xlat; agg=nothing)
    cols = x.index.columns
    d2 = map(xlat, cols[d])
    n = fieldindex(cols, d)
    NDSparse(cols[1:n-1]..., d2, cols[n+1:end]..., x.data, agg=agg, copy=true)
end

convertdim(x::NDSparse, d::Int, xlat::Dict; agg=nothing) = convertdim(x, d, i->xlat[i], agg=agg)

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
function reducedim_vec(f, x::NDSparse, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    cols = Columns(x.index.columns[[keep...]])
    if issorted(cols)
        idxs = copy(cols)
        xd = x.data
    else
        p = sortperm(cols)
        idxs = cols[p]
        xd = x.data[p]
    end
    d = vec_aggregate!(f, idxs, xd)
    NDSparse(idxs, d, presorted=true)
end

reducedim_vec(f, x::NDSparse, dims::Symbol) = reducedim_vec(f, x, [dims])
