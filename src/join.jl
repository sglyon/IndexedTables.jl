export naturaljoin, innerjoin, leftjoin, asofjoin

## Joins

# Natural Join (Both NDSParse arrays must have the same number of columns, in the same order)

function naturaljoin(left::NDSparse, right::NDSparse, op)
    lD, rD = left.data, right.data
    _naturaljoin(left, right, op, similar(lD, typeof(op(lD[1],rD[1])), 0))
end

const innerjoin = naturaljoin

combine_op(a, b) = tuple
combine_op(a::Columns, b::Columns) = (l, r)->(l..., r...)
combine_op(a, b::Columns) = (l, r)->(l, r...)
combine_op(a::Columns, b) = (l, r)->(l..., r)
similarz(a) = similar(a,0)

function naturaljoin(left::NDSparse, right::NDSparse)
    lD, rD = left.data, right.data
    op = combine_op(lD, rD)
    cols(v) = (v,)
    cols(v::Columns) = v.columns
    _naturaljoin(left, right, op, Columns((map(similarz,cols(lD))...,map(similarz,cols(rD))...)))
end

function _naturaljoin(left::NDSparse, right::NDSparse, op, data)
    flush!(left); flush!(right)
    lI, rI = left.index, right.index
    lD, rD = left.data, right.data
    ll, rr = length(lI), length(rI)

    # Guess the length of the result
    guess = min(ll, rr)

    # Initialize output array components
    I = _sizehint!(similar(lI,0), guess)
    _sizehint!(data, guess)

    # Match and insert rows
    i = j = 1

    while i <= ll && j <= rr
        c = rowcmp(lI, i, rI, j)
        if c == 0
            push!(I, lI[i])
            push!(data, op(lD[i], rD[j]))
            i += 1
            j += 1
        elseif c < 0
            i += 1
        else
            j += 1
        end
    end

    # Generate final datastructure
    NDSparse(I, data, presorted=true)
end

map{T,S,D}(f, x::NDSparse{T,D}, y::NDSparse{S,D}) = naturaljoin(x, y, f)

# left join

function leftjoin(left::NDSparse, right::NDSparse, op = NDSparseData.right)
    flush!(left); flush!(right)
    lI, rI = left.index, right.index
    lD, rD = left.data, right.data
    ll, rr = length(lI), length(rI)

    data = similar(lD)

    i = j = 1

    while i <= ll && j <= rr
        c = rowcmp(lI, i, rI, j)
        if c < 0
            @inbounds data[i] = lD[i]
            i += 1
        elseif c == 0
            @inbounds data[i] = op(lD[i], rD[j])
            i += 1
            j += 1
        else
            j += 1
        end
    end
    data[i:ll] = lD[i:ll]

    NDSparse(copy(lI), data, presorted=true)
end

# asof join

function asofjoin(left::NDSparse, right::NDSparse)
    flush!(left); flush!(right)
    lI, rI = left.index, right.index
    lD, rD = left.data, right.data
    ll, rr = length(lI), length(rI)

    data = similar(lD)

    i = j = 1

    while i <= ll && j <= rr
        c = rowcmp(lI, i, rI, j)
        if c < 0
            @inbounds data[i] = lD[i]
            i += 1
        elseif row_asof(lI, i, rI, j)  # all equal except last col left>=right
            j += 1
            while j <= rr && row_asof(lI, i, rI, j)
                j += 1
            end
            j -= 1
            @inbounds data[i] = rD[j]
            i += 1
        else
            j += 1
        end
    end
    data[i:ll] = lD[i:ll]

    NDSparse(copy(lI), data, presorted=true)
end

# merge - union join

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
merge{T,S,D<:Tuple}(x::NDSparse{T,D}, y::NDSparse{S,D}) = (flush!(x);flush!(y); _merge(x, y))
# merge without flush!
function _merge{T,S,D}(x::NDSparse{T,D}, y::NDSparse{S,D})
    I, J = x.index, y.index
    lI, lJ = length(I), length(J)
    #if isless(I[end], J[1])
    #    return NDSparse(vcat(x.index, y.index), vcat(x.data, y.data), presorted=true)
    #elseif isless(J[end], I[1])
    #    return NDSparse(vcat(y.index, x.index), vcat(y.data, x.data), presorted=true)
    #end
    n = lI + lJ - count_overlap(I, J)
    K = similar(I, n)::typeof(I)
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

function merge(x::NDSparse, xs::NDSparse...)
    as = [x, xs...]
    for a in as; flush!(a); end
    sort!(as, by=y->first(y.index))
    if all(i->isless(as[i-1].index[end], as[i].index[1]), 2:length(as))
        # non-overlapping
        return NDSparse(vcat(map(a->a.index, as)...),
                        vcat(map(a->a.data,  as)...),
                        presorted=true)
    end
    error("this case of `merge` is not yet implemented")
end

# broadcast join - repeat data along a dimension missing from one array

tslice(t::Tuple, I) = ntuple(i->t[I[i]], length(I))

function find_corresponding(Ap, Bp)
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

function match_indices(A::NDSparse, B::NDSparse)
    if isa(A.index.columns, NamedTuple) && isa(B.index.columns, NamedTuple)
        Ap = fieldnames(A.index.columns)
        Bp = fieldnames(B.index.columns)
    else
        Ap = typeof(A).parameters[2].parameters
        Bp = typeof(B).parameters[2].parameters
    end
    find_corresponding(Ap, Bp)
end

# broadcast over trailing dimensions, i.e. C's dimensions are a prefix
# of B's. this is an easy case since it's just an inner join plus
# sometimes repeating values from the right argument.
function _broadcast_trailing!(f, A::NDSparse, B::NDSparse, C::NDSparse)
    I = A.index
    data = A.data
    lI, rI = B.index, C.index
    lD, rD = B.data, C.data
    ll, rr = length(lI), length(rI)

    i = j = 1

    while i <= ll && j <= rr
        c = rowcmp(lI, i, rI, j)
        if c == 0
            while true
                push!(I, lI[i])
                push!(data, f(lD[i], rD[j]))
                i += 1
                (i <= ll && rowcmp(lI, i, rI, j)==0) || break
            end
            j += 1
        elseif c < 0
            i += 1
        else
            j += 1
        end
    end

    return A
end

function broadcast!(f::Function, A::NDSparse, B::NDSparse, C::NDSparse)
    flush!(A); flush!(B); flush!(C)
    B_inds = match_indices(A, B)
    C_inds = match_indices(A, C)
    all(i->B_inds[i] > 0 || C_inds[i] > 0, 1:ndims(A)) ||
        error("some destination indices are uncovered")
    if B_inds == ntuple(identity, ndims(A)) && C_inds[1:ndims(C)] == ntuple(identity, ndims(C))
        return _broadcast_trailing!(f, A, B, C)
    end
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

function broadcast(f::Function, A::NDSparse, B::NDSparse)
    if ndims(B) > ndims(A)
        broadcast!((x,y)->f(y,x), similar(B), B, A)
    else
        broadcast!(f, similar(A), A, B)
    end
end

broadcast(f::Function, x::NDSparse, y) = NDSparse(x.index, broadcast(f, x.data, y), presorted=true)
broadcast(f::Function, y, x::NDSparse) = NDSparse(x.index, broadcast(f, y, x.data), presorted=true)
