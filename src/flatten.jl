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

function mapslices(f, x::NDSparse, dims; name = nothing)
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

    if isa(y, NDSparse)
        # this means we need to concatenate outputs into a big NDSparse
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
        output = NDSparse(index, data)
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
            output = NDSparse(index, [y])
        else
            output = NDSparse(index, Columns([y], names=[name]))
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
        y = f(NDSparse(x.index[i:i], x.data[i:i]))
        n = length(y)

        foreach((x,y)->append_n!(x,y,n), I1.columns, k)
        append!(I2, y.index)
        append!(D, y.data)
    end
    NDSparse(I,D)
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
    NDSparse(I,D)
end
