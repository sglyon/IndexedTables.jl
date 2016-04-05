using Base.Test
using NDSparseData

let a = Indexes([1,2,1],["foo","bar","baz"]),
    b = Indexes([2,1,1],["bar","baz","foo"]),
    c = Indexes([1,1,2],["foo","baz","bar"])
    @test a != b
    @test a != c
    @test b != c
    @test sort!(a) == sort!(b) == sort!(c)
end

let c = Indexes([1,1,1,2,2], [1,2,4,3,5]),
    d = Indexes([1,1,2,2,2], [1,3,1,4,5]),
    e = Indexes([1,1,1], sort([rand(),0.5,rand()])),
    f = Indexes([1,1,1], sort([rand(),0.5,rand()]))
    @test union(c,d) == Indexes([1,1,1,1,2,2,2,2],[1,2,3,4,1,3,4,5])
    @test length(union(e,f).columns[1]) == 5
end

let c = Indexes([1,1,1,2,2], [1,2,4,3,5]),
    d = Indexes([1,1,2,2,2], [1,3,1,4,5]),
    e = Indexes([1,1,1], sort([rand(),0.5,rand()])),
    f = Indexes([1,1,1], sort([rand(),0.5,rand()]))
    @test intersect(c,d) == Indexes([1,2],[1,5])
    @test length(intersect(e,f).columns[1]) == 1
end

srand(123)
A = NDSparse(rand(1:3,10), rand('A':'F',10), map(UInt8,rand(1:3,10)), collect(1:10), randn(10))
B = NDSparse(map(UInt8,rand(1:3,10)), rand('A':'F',10), rand(1:3,10), randn(10))
C = NDSparse(map(UInt8,rand(1:3,10)), rand(1:3,10), rand(1:3,10), randn(10))

