filt_by_col!(f, col, indxs) = filter!(i->f(col[i]), indxs)

"""
`select(arr::IndexedTable, conditions::Pair...)`

Filter based on index columns. Conditions are accepted as column-function pairs.

Example: `select(arr, 1 => x->x>10, 3 => x->x!=10 ...)`
"""
function Base.select(arr::IndexedTable, conditions::Pair...)
    flush!(arr)
    indxs = [1:length(arr);]
    cols = arr.index.columns
    for (c,f) in conditions
        filt_by_col!(f, cols[c], indxs)
    end
    IndexedTable(Columns(map(x->x[indxs], cols)), arr.data[indxs], presorted=true)
end

"""
`select(arr:IndexedTable, which::DimName...; agg::Function)`

Select a subset of index columns. If the resulting array has duplicate index entries,
`agg` is used to combine the values.
"""
function Base.select(arr::IndexedTable, which::DimName...; agg=nothing)
    flush!(arr)
    IndexedTable(Columns(arr.index.columns[[which...]]), arr.data, agg=agg, copy=true)
end

# Filter on data field
function Base.filter(fn::Function, arr::IndexedTable)
    flush!(arr)
    data = arr.data
    indxs = filter(i->fn(data[i]), eachindex(data))
    IndexedTable(Columns(map(x->x[indxs], arr.index.columns)), data[indxs], presorted=true)
end

# aggregation

"""
`aggregate!(f::Function, arr::IndexedTable)`

Combine adjacent rows with equal indices using the given 2-argument reduction function,
in place.
"""
function aggregate!(f, x::IndexedTable)
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

"""
`aggregate(f::Function, arr::IndexedTable)`

Combine adjacent rows with equal indices using the given 2-argument reduction function,
returning the result in a new array.
"""
function aggregate(f, x::IndexedTable)
    idxs, data = aggregate_to(f, x.index, x.data)
    IndexedTable(idxs, data, presorted=true, copy=false)
end

# aggregate out of place, building up new indexes and data
function aggregate_to(f, src_idxs, src_data)
    dest_idxs, dest_data = similar(src_idxs,0), similar(src_data,0)
    n = length(src_idxs)
    i1 = 1
    while i1 <= n
        val = src_data[i1]
        i = i1+1
        while i <= n && roweq(src_idxs, i, i1)
            val = f(val, src_data[i])
            i += 1
        end
        push!(dest_idxs, src_idxs[i1])
        push!(dest_data, val)
        i1 = i
    end
    dest_idxs, dest_data
end

# returns a new vector where each element is computed by applying `f` to a vector of
# all values associated with equal indexes. idxs is modified in place.
function _aggregate_vec!(f, idxs::Columns, data)
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

# out of place vector aggregation
function aggregate_vec_to(f, src_idxs, src_data)
    n = length(src_idxs)
    dest_idxs = similar(src_idxs,0)
    local newdata
    newlen = 0
    i1 = 1
    while i1 <= n
        i = i1+1
        while i <= n && roweq(src_idxs, i, i1)
            i += 1
        end
        val = f(src_data[i1:(i-1)])
        if newlen == 0
            newdata = [val]
        else
            push!(newdata, val)
        end
        newlen += 1
        push!(dest_idxs, src_idxs[i1])
        i1 = i
    end
    (dest_idxs, (newlen==0 ? Union{}[] : newdata))
end

# vector aggregation, not modifying or computing new indexes. only returns new data.
function _aggregate_vec(f, idxs::Columns, data)
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
        i1 = i
    end
    newlen==0 ? Union{}[] : newdata
end

"""
`aggregate_vec(f::Function, x::IndexedTable)`

Combine adjacent rows with equal indices using a function from vector to scalar,
e.g. `mean`.
"""
function aggregate_vec(f, x::IndexedTable)
    idxs, data = aggregate_vec_to(f, x.index, x.data)
    IndexedTable(idxs, data, presorted=true, copy=false)
end

function _aggregate_vec(x::IndexedTable, names::Vector, funs::Vector)
    n = length(funs)
    n == 0 && return x
    n != length(names) && return x
    datacols = Any[ _aggregate_vec(funs[i], x.index, x.data) for i = 1:n-1 ]
    idx, lastcol = aggregate_vec_to(funs[n], x.index, x.data)
    IndexedTable(idx, Columns(datacols..., lastcol, names = names), presorted=true)
end

"""
`aggregate_vec(f::Vector{Function}, x::IndexedTable)`

Combine adjacent rows with equal indices using multiple functions from vector to scalar.
The result has multiple data columns, one for each function, named based on the functions.
"""
function aggregate_vec(fs::Vector, x::IndexedTable)
    _aggregate_vec(x, map(Symbol, fs), fs)
end

"""
`aggregate_vec(x::IndexedTable; funs...)`

Combine adjacent rows with equal indices using multiple functions from vector to scalar.
The result has multiple data columns, one for each function provided by `funs`.
"""
function aggregate_vec(x::IndexedTable; funs...)
    _aggregate_vec(x, [x[1] for x in funs], [x[2] for x in funs])
end


"""
`convertdim(x::IndexedTable, d::DimName, xlate; agg::Function, vecagg::Function, name)`

Apply function or dictionary `xlate` to each index in the specified dimension.
If the mapping is many-to-one, `agg` or `vecagg` is used to aggregate the results.
If `agg` is passed, it is used as a 2-argument reduction function over the data.
If `vecagg` is passed, it is used as a vector-to-scalar function to aggregate
the data.
`name` optionally specifies a new name for the translated dimension.
"""
function convertdim(x::IndexedTable, d::DimName, xlat; agg=nothing, vecagg=nothing, name=nothing)
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
        y = IndexedTable(cols[1:n-1]..., d2, cols[n+1:end]..., x.data, copy=false, names=names)
        idxs, data = aggregate_vec_to(vecagg, y.index, y.data)
        return IndexedTable(idxs, data, copy=false)
    end
    IndexedTable(cols[1:n-1]..., d2, cols[n+1:end]..., x.data, agg=agg, copy=true, names=names)
end

convertdim(x::IndexedTable, d::Int, xlat::Dict; agg=nothing, vecagg=nothing, name=nothing) = convertdim(x, d, i->xlat[i], agg=agg, vecagg=vecagg, name=name)

convertdim(x::IndexedTable, d::Int, xlat, agg) = convertdim(x, d, xlat, agg=agg)

sum(x::IndexedTable) = sum(x.data)

function reducedim(f, x::IndexedTable, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    select(x, keep..., agg=f)
end

reducedim(f, x::IndexedTable, dims::Symbol) = reducedim(f, x, [dims])

"""
`reducedim_vec(f::Function, arr::IndexedTable, dims)`

Like `reducedim`, except uses a function mapping a vector of values to a scalar instead
of a 2-argument scalar function.
"""
function reducedim_vec(f, x::IndexedTable, dims)
    keep = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    cols = Columns(x.index.columns[[keep...]])
    if issorted(cols)
        idxs, d = aggregate_vec_to(f, cols, x.data)
    else
        p = sortperm(cols)
        idxs = cols[p]
        xd = x.data[p]
        d = _aggregate_vec!(f, idxs, xd)
    end
    IndexedTable(idxs, d, presorted=true, copy=false)
end

reducedim_vec(f, x::IndexedTable, dims::Symbol) = reducedim_vec(f, x, [dims])

function dedup_names(ns)
    count = Dict{Symbol,Int}()
    for n in ns
        if haskey(count, n)
            count[n] += 1
        else
            count[n] = 1
        end
    end

    repeated = filter((k,v) -> v > 1, count)
    for k in keys(repeated)
        repeated[k] = 0
    end
    [haskey(repeated, n) ? Symbol(n, "_", repeated[n]+=1) : n for n in ns]
end

function mapslices(f, x::IndexedTable, dims; name = nothing)
    iterdims = setdiff([1:ndims(x);], map(d->fieldindex(x.index.columns,d), dims))
    idx = Any[Colon() for v in x.index.columns]

    iter = Columns(astuple(x.index.columns)[[iterdims...]])
    if !isempty(dims) || !issorted(iter)
        iter = sort(iter)
    end

    for j in 1:length(iterdims)
        d = iterdims[j]
        idx[d] = iter[1][j]
    end

    if isempty(dims)
        idx[end] = vcat(idx[end])
    end

    y = f(x[idx...]) # Apply on first slice

    if isa(y, IndexedTable)
        # this means we need to concatenate outputs into a big IndexedTable
        ns = vcat(dimlabels(x)[iterdims], dimlabels(y))
        if !all(x->isa(x, Symbol), ns)
            ns = nothing
        else
            ns = dedup_names(ns)
        end
        n = length(y)
        index_first = similar(iter, n)
        for j=1:n
            @inbounds index_first[j] = iter[1]
        end
        index = Columns(index_first.columns..., astuple(copy(y.index).columns)...; names=ns)
        data = copy(y.data)
        output = IndexedTable(index, data)
        if isempty(dims)
            _mapslices_itable_singleton!(f, output, x, 2)
        else
            _mapslices_itable!(f, output, x, iter, iterdims, 2)
        end
    else
        ns = dimlabels(x)[iterdims]
        if !all(x->isa(x, Symbol), ns)
            ns = nothing
        end
        index = Columns(iter[1:1].columns...; names=ns)
        if name === nothing
            output = IndexedTable(index, [y])
        else
            output = IndexedTable(index, Columns([y], names=[name]))
        end
        if isempty(dims)
            error("calling mapslices with no dimensions and scalar return value -- use map instead")
        else
            _mapslices_scalar!(f, output, x, iter, iterdims, 2, name!==nothing ? x->(x,) : identity)
        end
    end
end

function _mapslices_scalar!(f, output, x, iter, iterdims, start, coerce)
    idx = Any[Colon() for v in x.index.columns]

    for i = start:length(iter)
        if i != 1 && roweq(iter, i-1, i) # We've already visited this slice
            continue
        end
        for j in 1:length(iterdims)
            d = iterdims[j]
            idx[d] = iter[i][j]
        end
        if length(idx) == length(iterdims)
            idx[end] = vcat(idx[end])
        end
        y = f(x[idx...])
        push!(output.index, iter[i])
        push!(output.data, coerce(y))
    end
    output
end

function _mapslices_itable_singleton!(f, output, x, start)
    I = output.index
    D = output.data

    I1 = Columns(I.columns[1:ndims(x)])
    I2 = Columns(I.columns[ndims(x)+1:end])
    i = start
    for i in start:length(x)
        k = x.index[i]
        y = f(IndexedTable(x.index[i:i], x.data[i:i]))
        n = length(y)

        foreach((x,y)->append_n!(x,y,n), I1.columns, k)
        append!(I2, y.index)
        append!(D, y.data)
    end
    IndexedTable(I,D)
end

function _mapslices_itable!(f, output, x, iter, iterdims, start)
    idx = Any[Colon() for v in x.index.columns]
    I = output.index
    D = output.data
    initdims = length(iterdims)

    I1 = Columns(I.columns[1:initdims]) # filled from existing table
    I2 = Columns(I.columns[initdims+1:end]) # filled from output tables

    for i = start:length(iter)
        if i != 1 && roweq(iter, i-1, i) # We've already visited this slice
            continue
        end
        for j in 1:length(iterdims)
            d = iterdims[j]
            idx[d] = iter[i][j]
        end
        if length(idx) == length(iterdims)
            idx[end] = vcat(idx[end])
        end
        subtable = x[idx...]
        y = f(subtable)
        n = length(y)

        foreach((x,y)->append_n!(x,y,n), I1.columns, iter[i])
        append!(I2, y.index)
        append!(D, y.data)
    end
    IndexedTable(I,D)
end
