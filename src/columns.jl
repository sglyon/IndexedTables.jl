# a type that stores an array of tuples as a tuple of arrays

import Base:
    linearindexing, push!, size, sort, sort!, permute!, issorted, sortperm,
    summary

export Columns

immutable Columns{D<:Union{Tuple,NamedTuple}, C<:Tuple} <: AbstractVector{D}
    columns::C
end

function Columns(cols::AbstractVector...; names::Union{Vector{Symbol},Tuple{Vararg{Symbol}},Void}=nothing)
    if isa(names, Void)
        Columns{eltypes(typeof(cols)),typeof(cols)}(cols)
    else
        et = eval(:(@NT($(names...)))){map(eltype, cols)...}
        Columns{et,typeof(cols)}(cols)
    end
end

Columns(; pairs...) = Columns(map(x->x[2],pairs)..., names=Symbol[x[1] for x in pairs])

(::Type{Columns{D}}){D}(columns::AbstractVector...) = Columns{D,typeof(columns)}(columns)

@generated function astuple(n::NamedTuple)
    Expr(:tuple, [ Expr(:., :n, Expr(:quote, fieldname(n,i))) for i = 1:nfields(n) ]...)
end

eltype{D,C}(::Type{Columns{D,C}}) = D
length(c::Columns) = length(c.columns[1])
ndims(c::Columns) = 1
size(c::Columns) = (length(c),)
linearindexing{T<:Columns}(::Type{T}) = Base.LinearFast()
summary{D<:Tuple}(c::Columns{D}) = "Columns{$D}"

empty!(c::Columns) = (map(empty!, c.columns); c)
similar{D}(c::Columns{D}) = empty!(Columns{D}(map(similar, c.columns)...))
similar{D}(c::Columns{D}, n::Integer) = Columns{D}(map(a->similar(a,n), c.columns)...)
copy{D}(c::Columns{D}) = Columns{D}(map(copy, c.columns)...)

@inline ith_all(i, ::Tuple{}) = ()
@inline ith_all(i, as) = (as[1][i], ith_all(i, tail(as))...)

getindex{D<:Tuple}(c::Columns{D}, i::Integer) = ith_all(i, c.columns)
getindex{D<:NamedTuple}(c::Columns{D}, i::Integer) = D(ith_all(i, c.columns)...)

getindex{D}(c::Columns{D}, p::AbstractVector) = Columns{D}(map(c->c[p], c.columns)...)

setindex!(I::Columns, r::Tuple, i) = (_setindex!(I.columns[1], r[1], i, tail(I.columns), tail(r)); I)
@inline _setindex!(c1, r1, i, cr, rr) = (c1[i]=r1; _setindex!(cr[1], rr[1], i, tail(cr), tail(rr)))
@inline _setindex!(c1, r1, i, cr::Tuple{}, rr) = (c1[i] = r1)

push!(I::Columns, r::Tuple) = _pushrow!(I.columns[1], r[1], tail(I.columns), tail(r))
@inline _pushrow!(c1, r1, cr, rr) = (push!(c1, r1); _pushrow!(cr[1], rr[1], tail(cr), tail(rr)))
@inline _pushrow!(c1, r1, cr::Tuple{}, rr) = push!(c1, r1)

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

# row operations

@inline cmpelts(a, i, j) = (@inbounds x=cmp(a[i], a[j]); x)

@generated function rowless{D,C}(c::Columns{D,C}, i, j)
    N = length(C.parameters)
    ex = :(cmpelts(c.columns[$N], i, j) < 0)
    for n in N-1:-1:1
        ex = quote
            let d = cmpelts(c.columns[$n], i, j)
                (d == 0) ? ($ex) : (d < 0)
            end
        end
    end
    ex
end

@generated function rowcmp{D}(c::Columns{D}, i, d::Columns{D}, j)
    N = length(D.parameters)
    ex = :(cmp(c.columns[$N][i], d.columns[$N][j]))
    for n in N-1:-1:1
        ex = quote
            let k = cmp(c.columns[$n][i], d.columns[$n][j])
                (k == 0) ? ($ex) : k
            end
        end
    end
    ex
end

@generated function roweq{D,C}(c::Columns{D,C}, i, j)
    N = length(C.parameters)
    ex = :(cmpelts(c.columns[1], i, j) == 0)
    for n in 2:N
        ex = quote
            ($ex) && (cmpelts(c.columns[$n], i, j)==0)
        end
    end
    ex
end

@inline copyelt!(a, i, j) = (@inbounds a[i] = a[j])

copyrow!(I::Columns, i, src) = _copyrow!(I.columns[1], tail(I.columns), i, src)
@inline _copyrow!(c1, cr, i, src) = (copyelt!(c1, i, src); _copyrow!(cr[1], tail(cr), i, src))
@inline _copyrow!(c1, cr::Tuple{}, i, src) = copyelt!(c1, i, src)
