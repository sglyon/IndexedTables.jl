using Base.Test
using IndexedTables
using PooledArrays
using DataValues

let a = NDSparse([12,21,32], [52,41,34], [11,53,150]), b = NDSparse([12,23,32], [52,43,34], [56,13,10])
    c = naturaljoin(a, b, +)
    @test c[12,52] == 67
    @test c[32,34] == 160
    @test length(c.index) == 2
    @test naturaljoin(a, b) == NDSparse([12,32], [52,34], Columns([11,150], [56,10]))

    c = NDSparse([12,32], [52,34], Columns([0,1], [2,3]))
    @test naturaljoin(a, c) == NDSparse([12,32], [52,34], Columns([11,150], [0,1], [2,3]))
    @test naturaljoin(c, a) == NDSparse([12,32], [52,34], Columns([0,1], [2,3], [11,150]))

    c = select(a, 1=>x->x<30, 2=>x->x>40)
    @test c[12,52] == 11
    @test c[21,41] == 53
    @test length(c.index) == 2

    c = filter(x->x>100, a)
    @test c[32,34] == 150
    @test length(c.index) == 1
end

let a = NDSparse([1,1,2,2], [1,2,1,2], [6,7,8,9])
    @test select(a, 1, agg=+) == NDSparse([1,2], [13,17])
    @test select(a, 2, agg=+) == NDSparse([1,2], [14,16])
end

@test leftjoin(NDSparse([1,1,1,2], [2,3,4,4], [5,6,7,8]),
               NDSparse([1,1,3],   [2,4,4],   [9,10,12])) ==
    NDSparse([1,1,1,2], [2,3,4,4], Columns([5, 6, 7, 8], [9, NA, 10, NA]))

@test leftjoin(NDSparse([1,1,1,2], [2,3,4,4], [5,6,7,8]),
               NDSparse([1,1,2],   [2,4,4],   [9,10,12])) ==
    NDSparse([1,1,1,2], [2,3,4,4], Columns([5, 6, 7, 8], [9, NA, 10, 12]))

@test asofjoin(NDSparse([:msft,:ibm,:ge], [1,3,4], [100,200,150]),
               NDSparse([:ibm,:msft,:msft,:ibm], [0,0,0,2], [100,99,101,98])) ==
                   NDSparse([:msft,:ibm,:ge], [1,3,4], [101, 98, 150])

@test asofjoin(NDSparse([:AAPL, :IBM, :MSFT], [45, 512, 454], [63, 93, 54]),
               NDSparse([:AAPL, :MSFT, :AAPL], [547,250,34], [88,77,30])) ==
                   NDSparse([:AAPL, :MSFT, :IBM], [45, 454, 512], [30, 77, 93])

@test asofjoin(NDSparse([:aapl,:ibm,:msft,:msft],[1,1,1,3],[4,5,6,7]),
               NDSparse([:aapl,:ibm,:msft],[0,0,0],[8,9,10])) ==
                   NDSparse([:aapl,:ibm,:msft,:msft],[1,1,1,3],[8,9,10,10])

t = NDSparse([1, 1, 1, 1, 2, 2],
                 [2, 2, 2, 3, 3, 3],
                 [1, 4, 3, 5, 2, 0], presorted=true)

@test aggregate(max, t) == NDSparse([1, 1, 2], [2, 3, 3], [4,5,2])
@test aggregate(max, t, with=2) == NDSparse([1, 1, 2], [2, 3, 3], [2,3,3])
@test aggregate(max, t, with=column(t, 2)) == NDSparse([1, 1, 2], [2, 3, 3], [2,3,3])
@test aggregate(max, t, by=(2, iseven.(column(t, 3)))) == NDSparse([2, 2, 3, 3], [false, true, false, true], [3, 4, 5, 2])

@test aggregate_vec(maximum,
                    NDSparse([1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 3, 3, 3],
                                 [1, 4, 3, 5, 2, 0], presorted=true)) ==
                    NDSparse([1, 1], [2, 3], [4, 5])

@test aggregate_vec(maximum,
                    NDSparse([1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 3, 3, 3],
                                 [1, 4, 3, 5, 2, 0], presorted=true), with=(2,3)) ==
                    NDSparse([1, 1], [2, 3], [(2,4), (3,5)])

@test aggregate_vec([maximum, minimum],
                    NDSparse([1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 3, 3, 3],
                                 [1, 4, 3, 5, 2, 0], presorted=true)) ==
                    NDSparse([1, 1], [2, 3], Columns(maximum=[4, 5], minimum=[1, 0]))

@test aggregate_vec(NDSparse([1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 3, 3, 3],
                                 [1, 4, 3, 5, 2, 0], presorted=true),
                    maxv = maximum, minv = minimum) ==
                    NDSparse([1, 1], [2, 3], Columns(maxv=[4, 5], minv=[1, 0]))

@test convertdim(NDSparse([1, 1, 1, 1, 1, 1],
                              [0, 1, 2, 3, 4, 5],
                              [1, 4, 3, 5, 2, 0], presorted=true), 2, x->div(x,3), vecagg=maximum) ==
                    NDSparse([1, 1], [0, 1], [4, 5])

let A = rand(3,3), B = rand(3,3), C = rand(3,3)
    nA = convert(NDSparse, A)
    nB = convert(NDSparse, B)
    nB.index.columns[1][:] += 3
    @test merge(nA,nB) == convert(NDSparse, vcat(A,B))
    nC = convert(NDSparse, C)
    nC.index.columns[1][:] += 6
    @test merge(nA,nB,nC) == merge(nA,nC,nB) == convert(NDSparse, vcat(A,B,C))
    merge!(nA,nB)
    @test nA == convert(NDSparse, vcat(A,B))

    t1 = NDSparse(Columns(a=[1,1,2,2], b=[1,2,1,2]), [1,2,3,4])
    t2 = NDSparse(Columns(a=[0,1,2,3], b=[1,2,1,2]), [1,2,3,4])
    @test merge(t1, t2, agg=+) == NDSparse(Columns(a=[0,1,1,2,2,3], b=[1,1,2,1,2,2]), [1,1,4,6,4,4])
    @test merge(t1, t2, agg=nothing) == NDSparse(Columns(a=[0,1,1,1,2,2,2,3], b=[1,1,2,2,1,1,2,2]), [1,1,2,2,3,3,4,4])
end

let
    # scalar
    x=NDSparse(Columns(a=[1,1,1,2,2],b=PooledArray(["a","b","c","a","b"])),[1,2,3,4,5])
    t = mapslices(y->sum(y), x, (1,))
    @test t == NDSparse(Columns(b=["a","b","c"]), [5,7,3])

    A = [1]
    # shouldn't mutate input
    mapslices(x, [:a]) do slice
        NDSparse(Columns(A), A)
    end
    @test A == [1]

    # scalar
    r = Ref(0)
    t = mapslices(x, [:a]) do slice
        r[] += 1
        n = length(slice)
        NDSparse(Columns(c=[1:n;]), [r[] for i=1:n])
    end
    @test t == NDSparse(Columns(b=["a","a","b","b","c"], c=[1,2,1,2,1]), [1,1,2,2,3])

    # dedup names
    x=NDSparse(Columns(a=[1],b=[1]),Columns(c=[1]))
    t = mapslices(x,[:b]) do slice
            NDSparse(Columns(a=[2], c=[2]),
                         Columns(d=[1]))
    end
    @test t==NDSparse(Columns(a_1=[1], a_2=[2], c=[2]), Columns(d=[1]))

    # signleton slices
    x=NDSparse(Columns([1,2]),Columns([1,2]))
    @test_throws ErrorException mapslices(x,()) do slice
        true
    end
    t = mapslices(x,()) do slice
        @test slice == NDSparse(Columns([1]), Columns([1])) || slice == NDSparse(Columns([2]), Columns([2]))
        NDSparse(Columns([1]), ([1]))
    end
    @test t == NDSparse(Columns([1,2], [1,1]), [1,1])
end
