# Column-wise filtering (Accepts conditions as column-function pairs)
# Example: select(arr, 1 => x->x>10, 3 => x->x!=10 ...)

filt_by_col!(f, col, indxs) = filter!(i->f(col[i]), indxs)

function Base.select(arr::NDSparse, conditions::Pair...)
    flush!(arr)
    indxs = [1:length(arr);]
    cols = arr.index.columns
    for (c,f) in conditions
        filt_by_col!(f, cols[c], indxs)
    end
    NDSparse(Columns(map(x->x[indxs], cols)), arr.data[indxs], presorted=true)
end

# select a subset of columns
function Base.select(arr::NDSparse, which::DimName...; agg=nothing)
    flush!(arr)
    NDSparse(Columns(arr.index.columns[[which...]]), copy(arr.data), agg=agg)
end

# Filter on data field
function Base.filter(fn::Function, arr::NDSparse)
    flush!(arr)
    data = arr.data
    indxs = filter(i->fn(data[i]), eachindex(data))
    NDSparse(Columns(map(x->x[indxs], arr.index.columns)), data[indxs], presorted=true)
end

# aggregation

# combine adjacent rows with equal index using the given function
function aggregate!(f, x::NDSparse)
    idxs, data = x.index, x.data
    n = length(idxs)
    newlen = 1
    for i = 2:n
        if roweq(idxs, i, newlen)
            data[newlen] = f(data[newlen], data[i])
        else
            newlen += 1
            if newlen != i
                data[newlen] = data[i]
                copyrow!(idxs, newlen, i)
            end
        end
    end
    resize!(idxs, newlen)
    resize!(data, newlen)
    x
end

# the same, except returns a new vector where each element is computed
# by applying `f` to a vector of all values associated with equal indexes.
function vec_aggregate!(f, x::NDSparse)
    idxs, data = x.index, x.data
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

# convert dimension `d` of `x` using the given translation function.
# if the relation is many-to-one, aggregate with function `agg`
function convertdim(x::NDSparse, d::DimName, xlat; agg=nothing)
    cols = x.index.columns
    d2 = map(xlat, cols[d])
    n = fieldindex(cols, d)
    NDSparse(map(copy,cols[1:n-1])..., d2, map(copy,cols[n+1:end])..., copy(x.data), agg=agg)
end

convertdim(x::NDSparse, d::Int, xlat::Dict; agg=nothing) = convertdim(x, d, i->xlat[i], agg=agg)

convertdim(x::NDSparse, d::Int, xlat, agg) = convertdim(x, d, xlat, agg=agg)

sum(x::NDSparse) = sum(x.data)

reducedim(f, x::NDSparse, dims::Symbol) = reducedim(f, x, [dims])
function reducedim(f, x::NDSparse, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    select(x, keep..., agg=f)
end

reducedim_vec(f, x::NDSparse, dims::Symbol) = reducedim_vec(f, x, [dims])
function reducedim_vec(f, x::NDSparse, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    cols = Columns(x.index.columns[[keep...]])
    if issorted(cols)
        y = NDSparse(copy(cols), x.data, presorted=true)
    else
        y = NDSparse(cols, x.data)
    end
    d = vec_aggregate!(f, y)
    NDSparse(y.index, d, presorted=true)
end
