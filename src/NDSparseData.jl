module NDSparseData

using NamedTuples

import Base:
    show, eltype, length, getindex, setindex!, ndims, map, convert,
    ==, broadcast, broadcast!, empty!, copy, similar, sum, merge,
    permutedims, reducedim

export NDSparse, flush!, aggregate!, where, pairs, convertdim

const Tup = Union{Tuple,NamedTuple}
const DimName = Union{Int,Symbol}

include("utils.jl")
include("columns.jl")

immutable NDSparse{T, D<:Tuple, C<:Tup, V<:AbstractVector}
    index::Columns{D,C}
    data::V

    index_buffer::Columns{D,C}
    data_buffer::V
end

function NDSparse{T,D,C}(I::Columns{D,C}, d::AbstractVector{T}; agg=nothing, presorted=false)
    length(I) == length(d) || error("index and data must have the same number of elements")
    # ensure index is a `Columns` that generates tuples
    dt = D
    if eltype(I) <: NamedTuple
        dt = eltypes(typeof((I.columns...,)))
        I = Columns{dt,C}(I.columns)
    end
    if !presorted && !issorted(I)
        p = sortperm(I)
        I = I[p]
        d = d[p]
    end
    nd = NDSparse{T,dt,C,typeof(d)}(I, d, similar(I,0), similar(d,0))
    agg===nothing || aggregate!(agg, nd)
    return nd
end

function NDSparse(columns...; names=nothing, rest...)
    keys, data = columns[1:end-1], columns[end]
    NDSparse(Columns(keys..., names=names), data; rest...)
end

similar(t::NDSparse) = NDSparse(similar(t.index), empty!(similar(t.data)))

function copy(t::NDSparse)
    flush!(t)
    NDSparse(copy(t.index), copy(t.data), presorted=true)
end

function (==)(a::NDSparse, b::NDSparse)
    flush!(a); flush!(b)
    return a.index == b.index && a.data == b.data
end

function empty!(t::NDSparse)
    empty!(t.index)
    empty!(t.data)
    empty!(t.index_buffer)
    empty!(t.data_buffer)
    return t
end

if isless(Base.VERSION, v"0.5.0-")
writemime(io::IO, m::MIME"text/plain", t::NDSparse) = show(io, t)
end
showarray(io::IO, t::NDSparse) = show(io, t)

function show{T,D<:Tuple}(io::IO, t::NDSparse{T,D})
    flush!(t)
    print(io, "NDSparse $D => $T:")
    n = length(t)
    n == 0 && return
    rows = n > 20 ? [1:min(n,10); (n-9):n] : [1:min(n,10);]
    nc = length(t.index.columns)
    reprs  = [ sprint(io->showcompact(io,t.index.columns[j][i])) for i in rows, j in 1:nc ]
    if isa(t.data, Columns)
        dreprs = [ sprint(io->showcompact(io,t.data[i][j])) for i in rows, j in 1:nfields(eltype(t.data)) ]
    else
        dreprs = [ sprint(io->showcompact(io,t.data[i])) for i in rows ]
    end
    ndc = size(dreprs,2)
    inames = isa(t.index.columns, NamedTuple) ? map(string,keys(t.index.columns)) : fill("", nc)
    dnames = eltype(t.data) <: NamedTuple ? map(string,fieldnames(eltype(t.data))) : fill("", ndc)
    widths  = [ max(strwidth(inames[c]), maximum(map(strwidth, reprs[:,c]))) for c in 1:nc ]
    dwidths = [ max(strwidth(dnames[c]), maximum(map(strwidth, dreprs[:,c]))) for c in 1:ndc ]
    println(io)
    if isa(t.index.columns, NamedTuple) || isa(t.data, Columns)
        for c in 1:nc
            print(io, rpad(inames[c], widths[c]+(c==nc ? 1 : 2), " "))
        end
        print(io, "│ ")
        for c in 1:ndc
            print(io, rpad(dnames[c], dwidths[c]+(c==ndc ? 1 : 2), " "))
        end
        println(io)
        print(io, "─"^(sum(widths)+2*nc-1), "┼", "─"^(sum(dwidths)+2*ndc-1))
    else
        print(io, "─"^(sum(widths)+2*nc-1), "┬", "─"^(sum(dwidths)+2*ndc-1))
    end
    for r in 1:size(reprs,1)
        println(io)
        for c in 1:nc
            print(io, rpad(reprs[r,c], widths[c]+(c==nc ? 1 : 2), " "))
        end
        print(io, "│ ")
        for c in 1:ndc
            print(io, rpad(dreprs[r,c], dwidths[c]+(c==ndc ? 1 : 2), " "))
        end
        if n > 20 && r == 10
            println(io); print(io, " ⋮")
        end
    end
end

ndims(t::NDSparse) = length(t.index.columns)
length(t::NDSparse) = (flush!(t);length(t.index))
eltype{T,D,C,V}(::Type{NDSparse{T,D,C,V}}) = T

start(a::NDSparse) = start(a.data)
next(a::NDSparse, st) = next(a.data, st)
done(a::NDSparse, st) = done(a.data, st)

# ensure array is in correct storage order -- meant for internal use
function order!(t::NDSparse)
    p = sortperm(t.index)
    permute!(t.index, p)
    copy!(t.data, t.data[p])
    return t
end

function permutedims(t::NDSparse, p::AbstractVector)
    if !(length(p) == ndims(t) && isperm(p))
        throw(ArgumentError("argument to permutedims must be a valid permutation"))
    end
    flush!(t)
    NDSparse(Columns(t.index.columns[p]), t.data)
end

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
_in(x, v::Dimension) = x == v

import Base: tail
# test whether row r is within product(idxs...)
#row_in(r::Tuple{}, idxs) = true
row_in(r, idxs) = row_in(r[1], idxs[1], tail(r), tail(idxs))
@inline row_in(r1, i1, rr, ri) = _in(r1,i1) & row_in(rr[1], ri[1], tail(rr), tail(ri))
@inline row_in(r1, i1, rr::Tuple{}, ri) = _in(r1,i1)

range_estimate(col, idx) = 1:length(col)
range_estimate(col, idx::AbstractArray) = searchsortedfirst(col,first(idx)):searchsortedlast(col,last(idx))

# sizehint, making sure to return first argument
_sizehint!{T}(a::Array{T,1}, n::Integer) = (sizehint!(a, n); a)
_sizehint!(a::AbstractArray, sz::Integer) = a

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
        aggregate!((x,y)->y, temp)  # keep later values only

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

function count_overlap{D}(I::Columns{D}, J::Columns{D})
    lI, lJ = length(I), length(J)
    i = j = 1
    overlap = 0
    while i <= lI && j <= lJ
        c = rowcmp(I, i, J, j)
        if c == 0
            overlap += 1
            i += 1
            j += 1
        elseif c < 0
            i += 1
        else
            j += 1
        end
    end
    return overlap
end

# assign y into x out-of-place
merge{T,S,D}(x::NDSparse{T,D}, y::NDSparse{S,D}) = (flush!(x);flush!(y); _merge(x, y))
# merge without flush!
function _merge{T,S,D}(x::NDSparse{T,D}, y::NDSparse{S,D})
    I, J = x.index, y.index
    lI, lJ = length(I), length(J)
    n = lI + lJ - count_overlap(I, J)
    K = Columns(map(c->similar(c,n), I.columns))::typeof(I)
    data = similar(x.data, n)
    i = j = 1
    @inbounds for k = 1:n
        if i <= lI && j <= lJ
            c = rowcmp(I, i, J, j)
            if c >= 0
                K[k] = J[j]
                data[k] = y.data[j]
                if c==0; i += 1; end
                j += 1
            else
                K[k] = I[i]
                data[k] = x.data[i]
                i += 1
            end
        elseif i <= lI
            # TODO: copy remaining data columnwise
            K[k] = I[i]
            data[k] = x.data[i]
            i += 1
        elseif j <= lJ
            K[k] = J[j]
            data[k] = y.data[j]
            j += 1
        else
            break
        end
    end
    NDSparse(K, data, presorted=true)
end

map{T,S,D}(f, x::NDSparse{T,D}, y::NDSparse{S,D}) = naturaljoin(x, y, f)

map(f, x::NDSparse) = NDSparse(copy(x.index), map(f, x.data), presorted=true)

tslice(t::Tuple, I) = ntuple(i->t[I[i]], length(I))

function match_indices(A::NDSparse, B::NDSparse)
    Ap = typeof(A).parameters[2].parameters
    Bp = typeof(B).parameters[2].parameters
    matches = zeros(Int, length(Ap))
    J = IntSet(1:length(Bp))
    for i = 1:length(Ap)
        for j in J
            if Ap[i] == Bp[j]
                matches[i] = j
                delete!(J, j)
                break
            end
        end
    end
    isempty(J) || error("unmatched source indices: $(collect(J))")
    tuple(matches...)
end

function broadcast!(f::Function, A::NDSparse, B::NDSparse, C::NDSparse)
    flush!(A); flush!(B); flush!(C)
    B_inds = match_indices(A, B)
    C_inds = match_indices(A, C)
    all(i->B_inds[i] > 0 || C_inds[i] > 0, 1:ndims(A)) ||
        error("some destination indices are uncovered")
    common = filter(i->B_inds[i] > 0 && C_inds[i] > 0, 1:ndims(A))
    B_common = tslice(B_inds, common)
    C_common = tslice(C_inds, common)
    B_perm = sortperm(Columns(B.index.columns[[B_common...]]))
    C_perm = sortperm(Columns(C.index.columns[[C_common...]]))
    empty!(A)
    m, n = length(B_perm), length(C_perm)
    jlo = klo = 1
    while jlo <= m && klo <= n
        b_common = tslice(B.index[B_perm[jlo]], B_common)
        c_common = tslice(C.index[C_perm[klo]], C_common)
        x = cmp(b_common, c_common)
        x < 0 && (jlo += 1; continue)
        x > 0 && (klo += 1; continue)
        jhi, khi = jlo + 1, klo + 1
        while jhi <= m && tslice(B.index[B_perm[jhi]], B_common) == b_common
            jhi += 1
        end
        while khi <= n && tslice(C.index[C_perm[khi]], C_common) == c_common
            khi += 1
        end
        for ji = jlo:jhi-1
            j = B_perm[ji]
            b_row = B.index[j]
            for ki = klo:khi-1
                k = C_perm[ki]
                c_row = C.index[k]
                vals = ntuple(ndims(A)) do i
                    B_inds[i] > 0 ? b_row[B_inds[i]] : c_row[C_inds[i]]
                end
                push!(A.index, vals)
                push!(A.data, f(B.data[j], C.data[k]))
            end
        end
        jlo, klo = jhi, khi
    end
    order!(A)
end

# TODO: allow B to subsume columns of A as well?

broadcast(f::Function, A::NDSparse, B::NDSparse) = broadcast!(f, similar(A), A, B)

broadcast(f::Function, x::NDSparse, y) = NDSparse(x.index, broadcast(f, x.data, y), presorted=true)
broadcast(f::Function, y, x::NDSparse) = NDSparse(x.index, broadcast(f, y, x.data), presorted=true)

convert(::Type{NDSparse}, m::SparseMatrixCSC) = NDSparse(findnz(m)[[2,1,3]]..., presorted=true)

function convert{T}(::Type{NDSparse}, a::AbstractArray{T})
    n = length(a)
    data = Vector{T}(n)
    nd = ndims(a)
    idxs = [ Vector{Int}(n) for i = 1:nd ]
    i = 1
    for I in CartesianRange(size(a))
        val = a[I]
        for j = 1:nd
            idxs[j][i] = I[j]
        end
        data[i] = val
        i += 1
    end
    NDSparse(Columns(reverse(idxs)...), data, presorted=true)
end

# combine adjacent rows with equal index using the given function
function aggregate!(f, x::NDSparse)
    idxs, data = x.index, x.data
    n = length(idxs)
    newlen = 1
    current = newlen
    for i = 2:n
        if roweq(idxs, i, current)
            data[newlen] = f(data[newlen], data[i])
        else
            newlen += 1
            if newlen != i
                data[newlen] = data[i]
                copyrow!(idxs, newlen, i)
            end
            current = newlen
        end
    end
    resize!(data, newlen)
    for c in idxs.columns
        resize!(c, newlen)
    end
    x
end

# convert dimension `d` of `x` using the given translation function.
# if the relation is many-to-one, aggregate with function `agg`
function convertdim(x::NDSparse, d::DimName, xlat; agg=nothing)
    cols = x.index.columns
    d2 = map(xlat, cols[d])
    NDSparse(map(copy,cols[1:d-1])..., d2, map(copy,cols[d+1:end])..., copy(x.data), agg=agg)
end

convertdim(x::NDSparse, d::Int, xlat::Dict; agg=nothing) = convertdim(x, d, i->xlat[i], agg=agg)

convertdim(x::NDSparse, d::Int, xlat, agg) = convertdim(x, d, xlat, agg=agg)

sum(x::NDSparse) = sum(x.data)

function reducedim(f, x::NDSparse, dims)
    keep = setdiff([1:ndims(x);], dims)
    if isempty(keep)
        throw(ArgumentError("to remove all dimensions, use `reduce(f, A)`"))
    end
    select(x, keep..., agg=f)
end

include("query.jl")

end # module
