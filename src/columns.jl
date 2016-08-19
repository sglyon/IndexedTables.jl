# a type that stores an array of tuples as a tuple of arrays

import Base:
    linearindexing, push!, size, sort, sort!, permute!, issorted, sortperm,
    summary, resize!

export Columns

immutable Columns{D<:Tup, C<:Tup} <: AbstractVector{D}
    columns::C

    function Columns(c)
        length(c) > 0 || error("must have at least one column")
        n = length(c[1])
        for i = 2:length(c)
            length(c[i]) == n || error("all columns must have same length")
        end
        new(c)
    end
end

function Columns(cols::AbstractVector...; names::Union{Vector{Symbol},Tuple{Vararg{Symbol}},Void}=nothing)
    if isa(names, Void)
        Columns{eltypes(typeof(cols)),typeof(cols)}(cols)
    else
        dt = eval(:(@NT($(names...)))){map(eltype, cols)...}
        ct = eval(:(@NT($(names...)))){map(typeof, cols)...}
        Columns{dt,ct}(ct(cols...))
    end
end

Columns(; pairs...) = Columns(map(x->x[2],pairs)..., names=Symbol[x[1] for x in pairs])

Columns(c::Tuple) = Columns{eltypes(typeof(c)),typeof(c)}(c)
Columns(c::NamedTuple) = Columns{eltypes(Tuple{typeof(c).types...}),typeof(c)}(c)

eltype{D,C}(::Type{Columns{D,C}}) = D
length(c::Columns) = length(c.columns[1])
ndims(c::Columns) = 1
size(c::Columns) = (length(c),)
linearindexing{T<:Columns}(::Type{T}) = Base.LinearFast()
summary{D<:Tuple}(c::Columns{D}) = "Columns{$D}"

empty!(c::Columns) = (map(empty!, c.columns); c)
similar{D,C}(c::Columns{D,C}) = empty!(Columns{D,C}(map(similar, c.columns)))
similar{D,C}(c::Columns{D,C}, n::Integer) = Columns{D,C}(map(a->similar(a,n), c.columns))
copy{D,C}(c::Columns{D,C}) = Columns{D,C}(map(copy, c.columns))

getindex{D<:Tuple}(c::Columns{D}, i::Integer) = ith_all(i, c.columns)
getindex{D<:NamedTuple}(c::Columns{D}, i::Integer) = D(ith_all(i, c.columns)...)

getindex{D,C}(c::Columns{D,C}, p::AbstractVector) = Columns{D,C}(map(c->c[p], c.columns))

setindex!(I::Columns, r::Tup, i::Integer) = (foreach((c,v)->(c[i]=v), I.columns, r); I)

push!(I::Columns, r::Tup) = (foreach(push!, I.columns, r); I)

resize!(I::Columns, n::Int) = (foreach(c->resize!(c,n), I.columns); I)

function ==(x::Columns, y::Columns)
    length(x.columns) == length(y.columns) || return false
    n = length(x)
    length(y) == n || return false
    for i in 1:n
        x[i] == y[i] || return false
    end
    return true
end

function sortperm(c::Columns)
    if length(c.columns) == 1
        return sortperm(c.columns[1], alg=MergeSort)
    end
    sort!([1:length(c);], lt=(x,y)->rowless(c, x, y), alg=MergeSort)
end
issorted(c::Columns) = issorted(1:length(c), lt=(x,y)->rowless(c, x, y))

function permute!(c::Columns, p::AbstractVector)
    for v in c.columns
        copy!(v, v[p])
    end
    return c
end
sort!(c::Columns) = permute!(c, sortperm(c))
sort(c::Columns) = c[sortperm(c)]

map(p::Proj, c::Columns) = p(c.columns)
(p::Proj)(c::Columns) = p(c.columns)

# fused indexing operations
# these can be implemented for custom vector types like PooledVector where
# you can get big speedups by doing indexing and an operation in one step.

@inline cmpelts(a, i, j) = (@inbounds x=cmp(a[i], a[j]); x)
@inline copyelt!(a, i, j) = (@inbounds a[i] = a[j])

# row operations

copyrow!(I::Columns, i, src) = foreach(c->copyelt!(c, i, src), I.columns)

@generated function rowless{D,C}(c::Columns{D,C}, i, j)
    N = length(C.parameters)
    ex = :(cmpelts(getfield(c.columns,$N), i, j) < 0)
    for n in N-1:-1:1
        ex = quote
            let d = cmpelts(getfield(c.columns,$n), i, j)
                (d == 0) ? ($ex) : (d < 0)
            end
        end
    end
    ex
end

@generated function roweq{D,C}(c::Columns{D,C}, i, j)
    N = length(C.parameters)
    ex = :(cmpelts(getfield(c.columns,1), i, j) == 0)
    for n in 2:N
        ex = :(($ex) && (cmpelts(getfield(c.columns,$n), i, j)==0))
    end
    ex
end

@generated function rowcmp{D}(c::Columns{D}, i, d::Columns{D}, j)
    N = length(D.parameters)
    ex = :(cmp(getfield(c.columns,$N)[i], getfield(d.columns,$N)[j]))
    for n in N-1:-1:1
        ex = quote
            let k = cmp(getfield(c.columns,$n)[i], getfield(d.columns,$n)[j])
                (k == 0) ? ($ex) : k
            end
        end
    end
    ex
end

# test that the row on the right is "as of" the row on the left, i.e.
# all columns are equal except left >= right in last column.
# Could be generalized to some number of trailing columns, but I don't
# know whether that has applications.
@generated function row_asof{D,C}(c::Columns{D,C}, i, d::Columns{D,C}, j)
    N = length(C.parameters)
    if N == 1
        ex = :(!isless(getfield(c.columns,1)[i], getfield(d.columns,1)[j]))
    else
        ex = :(isequal(getfield(c.columns,1)[i], getfield(d.columns,1)[j]))
    end
    for n in 2:N
        if N == n
            ex = :(($ex) && !isless(getfield(c.columns,$n)[i], getfield(d.columns,$n)[j]))
        else
            ex = :(($ex) && isequal(getfield(c.columns,$n)[i], getfield(d.columns,$n)[j]))
        end
    end
    ex
end
