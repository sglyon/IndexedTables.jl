# getindex

getindex(t::NDSparse, idxs...) = (flush!(t); _getindex(t, idxs))

_getindex{T,D<:Tuple}(t::NDSparse{T,D}, idxs::D) = _getindex_scalar(t, idxs)
_getindex(t::NDSparse, idxs::Tuple{Vararg{Real}}) = _getindex_scalar(t, idxs)

function _getindex_scalar(t, idxs)
    i = searchsorted(t.index, idxs)
    length(i) != 1 && throw(KeyError(idxs))
    t.data[first(i)]
end

_in(x, y) = in(x, y)
_in(x, ::Colon) = true
_in(x, v::AbstractVector) = (idx=searchsortedfirst(v, x); idx<=length(v) && v[idx]==x)
_in(x, v::AbstractString) = x == v

import Base: tail
# test whether row r is within product(idxs...)
#row_in(r::Tuple{}, idxs) = true
row_in(r, idxs) = row_in(r[1], idxs[1], tail(r), tail(idxs))
@inline row_in(r1, i1, rr, ri) = _in(r1,i1) & row_in(rr[1], ri[1], tail(rr), tail(ri))
@inline row_in(r1, i1, rr::Tuple{}, ri) = _in(r1,i1)

range_estimate(col, idx) = 1:length(col)
range_estimate(col, idx::AbstractArray) = searchsortedfirst(col,first(idx)):searchsortedlast(col,last(idx))

index_by_col!(idx, col, out) = filt_by_col!(x->_in(x, idx), col, out)

function _getindex(t::NDSparse, idxs)
    I = t.index
    if length(idxs) != length(I.columns)
        error("wrong number of indices")
    end
    for idx in idxs
        isa(idx, AbstractVector) && (issorted(idx) || error("indices must be sorted for ranged/vector indexing"))
    end
    out = convert(Vector{Int32}, range_estimate(I.columns[1], idxs[1]))
    filter!(i->row_in(I[i], idxs), out)
    # column-wise algorithm
    #for c in 2:ndims(t)
    #    index_by_col!(idxs[c], I.columns[c], out)
    #end
    NDSparse(Columns(map(x->x[out], I.columns)), t.data[out], presorted=true)
end

# iterators over indices - lazy getindex

function where(d::NDSparse, idxs...)
    I = d.index
    data = d.data
    rng = range_estimate(I.columns[1], idxs[1])
    (data[i] for i in Filter(r->row_in(I[r], idxs), rng))
end

pairs(d::NDSparse) = (d.index[i]=>d.data[i] for i in 1:length(d))

function pairs(d::NDSparse, idxs...)
    I = d.index
    data = d.data
    rng = range_estimate(I.columns[1], idxs[1])
    (I[i]=>data[i] for i in Filter(r->row_in(I[r], idxs), rng))
end

# setindex!

setindex!(t::NDSparse, rhs, idxs...) = _setindex!(t, rhs, fixi(t, idxs))
setindex!(t::NDSparse, rhs, idxs::Real...) = _setindex!(t, rhs, idxs)

_setindex!{T,D}(t::NDSparse{T,D}, rhs::AbstractArray, idxs::D) = _setindex_scalar!(t, rhs, idxs)
_setindex!(t::NDSparse, rhs::AbstractArray, idxs::Tuple{Vararg{Real}}) = _setindex_scalar!(t, rhs, idxs)
_setindex!{T,D}(t::NDSparse{T,D}, rhs, idxs::D) = _setindex_scalar!(t, rhs, idxs)
#_setindex!(t::NDSparse, rhs, idxs::Tuple{Vararg{Real}}) = _setindex_scalar!(t, rhs, idxs)

function _setindex_scalar!(t, rhs, idxs)
    push!(t.index_buffer, idxs)
    push!(t.data_buffer, rhs)
    t
end

_setindex!(t::NDSparse, rhs::NDSparse, idxs::Tuple{Vararg{Real}}) = _setindex!(t, rhs.data, idxs)
_setindex!(t::NDSparse, rhs::NDSparse, idxs) = _setindex!(t, rhs.data, idxs)

@inline function fixi(t::NDSparse, n::Int, i1::Colon, irest...)
    u = unique(t.index.columns[n])
    (n==1 ? u : sort(u), fixi(t, n+1, irest...)...)
end

fixi(t::NDSparse, n::Int, i1, irest...) = (i1, fixi(t, n+1, irest...)...)
fixi(t::NDSparse, n::Int) = ()
fixi(t::NDSparse, idxs) = fixi(t, 1, idxs...)

function _setindex!{T,D}(t::NDSparse{T,D}, rhs::AbstractArray, idxs)
    # TODO performance
    for (x,I) in zip(rhs,product(idxs...))
        _setindex!(t, x, convert(D, I))
    end
end

function _setindex!{T,D}(t::NDSparse{T,D}, rhs, idxs)
    # TODO performance
    for I in product(idxs...)
        _setindex!(t, rhs, convert(D, I))
    end
end

# sort and merge data from accumulation buffer
function flush!(t::NDSparse)
    if !isempty(t.data_buffer)
        # 1. sort the buffer
        p = sortperm(t.index_buffer)
        ibuf = t.index_buffer[p]
        dbuf = t.data_buffer[p]
        temp = NDSparse(ibuf, dbuf)
        aggregate!(right, temp)  # keep later values only

        # 2. merge to a new copy
        new = _merge(t, temp)

        # 3. resize and copy data into t
        for i = 1:length(t.index.columns)
            resize!(t.index.columns[i], length(new.index.columns[i]))
            copy!(t.index.columns[i], new.index.columns[i])
        end
        resize!(t.data, length(new.data))
        copy!(t.data, new.data)

        # 4. clear buffer
        for c in t.index_buffer.columns; empty!(c); end
        empty!(t.data_buffer)
    end
    nothing
end
