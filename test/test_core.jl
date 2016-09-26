using Base.Test
using NDSparseData

let a = Columns([1,2,1],["foo","bar","baz"]),
    b = Columns([2,1,1],["bar","baz","foo"]),
    c = Columns([1,1,2],["foo","baz","bar"])
    @test a != b
    @test a != c
    @test b != c
    sort!(a)
    @test sort(b) == a
    sort!(b); sort!(c)
    @test a == b == c
    @test size(a) == size(b) == size(c) == (3,)
    @test eltype(a) == Tuple{Int,String}
end

let c = Columns([1,1,1,2,2], [1,2,4,3,5]),
    d = Columns([1,1,2,2,2], [1,3,1,4,5]),
    e = Columns([1,1,1], sort([rand(),0.5,rand()])),
    f = Columns([1,1,1], sort([rand(),0.5,rand()]))
    @test merge(NDSparse(c,ones(5)),NDSparse(d,ones(5))).index == Columns([1,1,1,1,2,2,2,2],[1,2,3,4,1,3,4,5])
    @test length(merge(NDSparse(e,ones(3)),NDSparse(f,ones(3)))) == 5
    @test summary(c) == "Columns{Tuple{Int64,Int64}}"
end

let c = Columns([1,1,1,2,2], [1,2,4,3,5]),
    d = Columns([1,1,2,2,2], [1,3,1,4,5]),
    e = Columns([1,1,1], sort([rand(),0.5,rand()])),
    f = Columns([1,1,1], sort([rand(),0.5,rand()]))
    @test map(+,NDSparse(c,ones(5)),NDSparse(d,ones(5))).index == Columns([1,2],[1,5])
    @test length(map(+,NDSparse(e,ones(3)),NDSparse(f,ones(3)))) == 1
    @test eltype(c) == Tuple{Int,Int}
end

srand(123)
A = NDSparse(rand(1:3,10), rand('A':'F',10), map(UInt8,rand(1:3,10)), collect(1:10), randn(10))
B = NDSparse(map(UInt8,rand(1:3,10)), rand('A':'F',10), rand(1:3,10), randn(10))
C = NDSparse(map(UInt8,rand(1:3,10)), rand(1:3,10), rand(1:3,10), randn(10))

let a = NDSparse([12,21,32], [52,41,34], [11,53,150]), b = NDSparse([12,23,32], [52,43,34], [56,13,10])
    @test eltype(a) == Int
    @test sum(a) == 214

    c = similar(a)
    @test typeof(c) == typeof(a)
    @test length(c.index) == 0

    c = copy(a)
    @test typeof(c) == typeof(a)
    @test length(c.index) == length(a.index)
    empty!(c)
    @test length(c.index) == 0

    c = convertdim(convertdim(a, 1, Dict(12=>10, 21=>20, 32=>20)), 2, Dict(52=>50, 34=>20, 41=>20), -)
    @test c[20,20] == 97

    c = map(+, a, b)
    @test length(c.index) == 2
    @test sum(map(-, c, c)) == 0

    @test map(iseven, a) == NDSparse([12,21,32], [52,41,34], [false,false,true])
end

let S = spdiagm(1:5)
    nd = convert(NDSparse, S)
    @test sum(S) == sum(nd) == sum(convert(NDSparse, full(S)))

    @test sum(broadcast(+, 10, nd)) == (sum(nd) + 10*nnz(S))
    @test sum(broadcast(+, nd, 10)) == (sum(nd) + 10*nnz(S))
    @test sum(broadcast(+, nd, nd)) == 2*(sum(nd))

    nd[1:5,1:5] = 2
    @test sum(nd[1:5, 1:5]) == 50
end

let S = sprand(10,10,.1), v = rand(10)
    nd = convert(NDSparse, S)
    ndv = convert(NDSparse,v)
    @test broadcast(*, nd, ndv) == convert(NDSparse, S .* v)
    # test matching dimensions by name
    ndt0 = convert(NDSparse, S .* (v'))
    ndt = NDSparse(Columns(a=ndt0.index.columns[1], b=ndt0.index.columns[2]), ndt0.data, presorted=true)
    @test broadcast(*,
                    NDSparse(Columns(a=nd.index.columns[1], b=nd.index.columns[2]), nd.data),
                    NDSparse(Columns(b=ndv.index.columns[1]), ndv.data)) == ndt
end

let a = rand(10), b = rand(10), c = rand(10)
    @test NDSparse(a, b, c) == NDSparse(a, b, c)
    c2 = copy(c)
    c2[1] += 1
    @test NDSparse(a, b, c) != NDSparse(a, b, c2)
    b2 = copy(b)
    b2[1] += 1
    @test NDSparse(a, b, c) != NDSparse(a, b2, c)
end

let a = rand(10), b = rand(10), c = rand(10), d = rand(10)
    local nd = NDSparse(a,b,c,d)
    @test permutedims(nd,[3,1,2]) == NDSparse(c,a,b,d)
    @test_throws ArgumentError permutedims(nd, [1,2])
    @test_throws ArgumentError permutedims(nd, [1,3])
    @test_throws ArgumentError permutedims(nd, [1,2,2])
end

let r=1:5, s=1:2:5
    A = NDSparse([r;], [r;], [r;])
    @test A[s, :] == NDSparse([s;], [s;], [s;])
    @test_throws ErrorException A[s, :, :]
end

let a = NDSparse([1,2,2,2], [1,2,3,4], [10,9,8,7])
    @test a[1,1] == 10
    @test a[2,3] == 8
    #@test_throws ErrorException a[2]
    @test a[2,:] == NDSparse([2,2,2], [2,3,4], [9,8,7])
    @test a[:,1] == NDSparse([1], [1], [10])
    @test collect(where(a, 2, :)) == [9,8,7]
    @test collect(pairs(a)) == [(1,1)=>10, (2,2)=>9, (2,3)=>8, (2,4)=>7]
    @test first(pairs(a, :, 3)) == ((2,3)=>8)

    update!(x->x+10, a, 2, :)
    @test a == NDSparse([1,2,2,2], [1,2,3,4], [10,19,18,17])
    update!(77, a, 2, 2:3)
    @test a == NDSparse([1,2,2,2], [1,2,3,4], [10,77,77,17])
end

let a = NDSparse([1,2,2,2], [1,2,3,4], zeros(4))
    a2 = copy(a); a3 = copy(a)
    #a[2,:] = 1
    #@test a == NDSparse([1,2,2,2], [1,2,3,4], Float64[0,1,1,1])
    a2[2,[2,3]] = 1
    @test a2 == NDSparse([1,2,2,2], [1,2,3,4], Float64[0,1,1,0])
    a3[2,[2,3]] = [8,9]
    @test a3 == NDSparse([1,2,2,2], [1,2,3,4], Float64[0,8,9,0])
end

# issue #15
let a = NDSparse([1,2,3,4], [1,2,3,4], [1,2,3,4])
    a[5,5] = 5
    a[5,5] = 6
    @test a[5,5] == 6
end

let a = NDSparse([1,2,2,3,4,5], [1,2,2,3,4,5], [1,2,20,3,4,5], agg=+)
    @test a == NDSparse([1,2,3,4,5], [1,2,3,4,5], [1,22,3,4,5])
end

let a = rand(5,5,5)
    for dims in ([2,3], [1], [2])
        r = squeeze(reducedim(+, a, dims), (dims...,))
        asnd = convert(NDSparse,a)
        b = reducedim(+, asnd, dims)
        bv = reducedim_vec(sum, asnd, dims)
        c = convert(NDSparse, r)
        @test b.index == c.index == bv.index
        @test_approx_eq b.data c.data
        @test_approx_eq bv.data c.data
    end
    @test_throws ArgumentError reducedim(+, convert(NDSparse,a), [1,2,3])
end

for a in (rand(2,2), rand(3,5))
    nd = convert(NDSparse, a)
    @test nd == convert(NDSparse, sparse(a))
    for (I,d) in zip(nd.index, nd.data)
        @test a[I...] == d
    end
end

_colnames(x::NDSparse) = keys(x.index.columns)

@test _colnames(NDSparse(ones(2),ones(2),ones(2),names=[:a,:b])) == [:a, :b]
@test _colnames(NDSparse(Columns(x=ones(2),y=ones(2)), ones(2))) == [:x, :y]

let x = NDSparse(Columns(x = [1,2,3], y = [4,5,6], z = [7,8,9]), [10,11,12])
    names = [:x, :y, :z]
    @test _colnames(x) == names
    @test _colnames(filter(a->a==11, x)) == names
    @test _colnames(select(x, :z, :x)) == [:z, :x]
    @test _colnames(select(x, :y)) == [:y]
    @test _colnames(select(x, :x=>a->a>1, :z=>a->a>7)) == names
    @test _colnames(x[1:2, 4:5, 8:9]) == names
    @test convertdim(x, :y, a->0) == NDSparse(Columns([1,2,3], [0,0,0], [7,8,9]), [10,11,12])
    @test convertdim(x, :y, a->0, name=:yy) == NDSparse(Columns(x=[1,2,3], yy=[0,0,0], z=[7,8,9]), [10,11,12])
end

# test showing
@test repr(NDSparse([1,2,3],[3,2,1],Float64[4,5,6])) == """
─────┬────
1  3 │ 4.0
2  2 │ 5.0
3  1 │ 6.0"""

@test repr(NDSparse(Columns(a=[1,2,3],test=[3,2,1]),Float64[4,5,6])) == """
a  test │ 
────────┼────
1  3    │ 4.0
2  2    │ 5.0
3  1    │ 6.0"""

@test repr(NDSparse(Columns(a=[1,2,3],test=[3,2,1]),Columns(x=Float64[4,5,6],y=[9,8,7]))) == """
a  test │ x    y
────────┼───────
1  3    │ 4.0  9
2  2    │ 5.0  8
3  1    │ 6.0  7"""

@test repr(NDSparse([1,2,3],[3,2,1],Columns(x=Float64[4,5,6],y=[9,8,7]))) == """
     │ x    y
─────┼───────
1  3 │ 4.0  9
2  2 │ 5.0  8
3  1 │ 6.0  7"""

@test repr(NDSparse([1:21;],ones(Int,21))) == """
───┬──
1  │ 1
2  │ 1
3  │ 1
4  │ 1
5  │ 1
6  │ 1
7  │ 1
8  │ 1
9  │ 1
10 │ 1
   ⋮
12 │ 1
13 │ 1
14 │ 1
15 │ 1
16 │ 1
17 │ 1
18 │ 1
19 │ 1
20 │ 1
21 │ 1"""
