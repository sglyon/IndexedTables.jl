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

# showing

if isless(Base.VERSION, v"0.5.0-")
writemime(io::IO, m::MIME"text/plain", t::NDSparse) = show(io, t)
end

function show{T,D<:Tuple}(io::IO, t::NDSparse{T,D})
    flush!(t)
    print(io, "NDSparse $D => $T:")
    n = length(t)
    n == 0 && return
    rows = n > 20 ? [1:min(n,10); (n-9):n] : [1:n;]
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
    if isa(t.index.columns, NamedTuple) || (isa(t.data, Columns) && isa(t.data.columns, NamedTuple))
        for c in 1:nc
            print(io, rpad(inames[c], widths[c]+(c==nc ? 1 : 2), " "))
        end
        print(io, "│ ")
        for c in 1:ndc
            print(io, c==ndc ? dnames[c] : rpad(dnames[c], dwidths[c]+2, " "))
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
            print(io, c==ndc ? dreprs[r,c] : rpad(dreprs[r,c], dwidths[c]+2, " "))
        end
        if n > 20 && r == 10
            println(io)
            print(io, " "^(sum(widths)+2*nc-1))
            print(io, "⋮")
        end
    end
end

# map and convert

map(f, x::NDSparse) = NDSparse(copy(x.index), map(f, x.data), presorted=true)

# NDSparse uses lex order, Base arrays use colex order, so we need to
# reorder the data. transpose and permutedims are used for this.
convert(::Type{NDSparse}, m::SparseMatrixCSC) = NDSparse(findnz(m.')[[2,1,3]]..., presorted=true)

function convert{T}(::Type{NDSparse}, a::AbstractArray{T})
    n = length(a)
    nd = ndims(a)
    a = permutedims(a, [nd:-1:1;])
    data = reshape(a, (n,))
    idxs = [ Vector{Int}(n) for i = 1:nd ]
    i = 1
    for I in CartesianRange(size(a))
        for j = 1:nd
            idxs[j][i] = I[j]
        end
        i += 1
    end
    NDSparse(Columns(reverse(idxs)...), data, presorted=true)
end

# getindex and setindex!
include("indexing.jl")

# joins
include("join.jl")

# query and aggregate
include("query.jl")

end # module
