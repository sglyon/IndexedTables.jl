using Base.Test
using NDSparseData

let a = Indexes([1,2,1],["foo","bar","baz"]),
    b = Indexes([2,1,1],["bar","baz","foo"]),
    c = Indexes([1,1,2],["foo","baz","bar"])
    @test a != b
    @test a != c
    @test b != c
    @test sort!(a) == sort!(b) == sort!(c)
    @test size(a) == size(b) == size(c) == (3,)
    @test eltype(a) == Tuple{Int,String}
end

let c = Indexes([1,1,1,2,2], [1,2,4,3,5]),
    d = Indexes([1,1,2,2,2], [1,3,1,4,5]),
    e = Indexes([1,1,1], sort([rand(),0.5,rand()])),
    f = Indexes([1,1,1], sort([rand(),0.5,rand()]))
    @test union(c,d) == Indexes([1,1,1,1,2,2,2,2],[1,2,3,4,1,3,4,5])
    @test length(union(e,f).columns[1]) == 5
    @test summary(c) == "Indexes{Tuple{Int64,Int64}}"
end

let c = Indexes([1,1,1,2,2], [1,2,4,3,5]),
    d = Indexes([1,1,2,2,2], [1,3,1,4,5]),
    e = Indexes([1,1,1], sort([rand(),0.5,rand()])),
    f = Indexes([1,1,1], sort([rand(),0.5,rand()]))
    @test intersect(c,d) == Indexes([1,2],[1,5])
    @test length(intersect(e,f).columns[1]) == 1
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
    @test length(c.indexes) == 0

    c = copy(a)
    @test typeof(c) == typeof(a)
    @test length(c.indexes) == length(a.indexes)
    empty!(c)
    @test length(c.indexes) == 0

    c = convertdim(convertdim(a, 1, Dict(12=>10, 21=>20, 32=>20)), 2, Dict(52=>50, 34=>20, 41=>20), -)
    @test c[20,20] == 97

    c = map(+, a, b)
    @test length(c.indexes) == 2
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
end

let a = NDSparse([1,2,2,2], [1,2,3,4], [10,9,8,7])
    @test a[2,:] == NDSparse([2,2,2], [2,3,4], [9,8,7])
    @test a[:,1] == NDSparse([1], [1], [10])
    @test collect(where(a, 2, :)) == [9,8,7]
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
        @test reducedim(+, convert(NDSparse,a), 4-dims) == convert(NDSparse,
                                                                   squeeze(reducedim(+, a, dims), (dims...,)))
    end
    @test_throws ArgumentError reducedim(+, convert(NDSparse,a), [1,2,3])
end
