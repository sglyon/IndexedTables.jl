using Base.Test
using IndexedTables

let a = IndexedTable([12,21,32], [52,41,34], [11,53,150]), b = IndexedTable([12,23,32], [52,43,34], [56,13,10])
    c = naturaljoin(a, b, +)
    @test c[12,52] == 67
    @test c[32,34] == 160
    @test length(c.index) == 2
    @test naturaljoin(a, b) == IndexedTable([12,32], [52,34], Columns([11,150], [56,10]))

    c = IndexedTable([12,32], [52,34], Columns([0,1], [2,3]))
    @test naturaljoin(a, c) == IndexedTable([12,32], [52,34], Columns([11,150], [0,1], [2,3]))
    @test naturaljoin(c, a) == IndexedTable([12,32], [52,34], Columns([0,1], [2,3], [11,150]))

    c = select(a, 1=>x->x<30, 2=>x->x>40)
    @test c[12,52] == 11
    @test c[21,41] == 53
    @test length(c.index) == 2

    c = filter(x->x>100, a)
    @test c[32,34] == 150
    @test length(c.index) == 1
end

let a = IndexedTable([1,1,2,2], [1,2,1,2], [6,7,8,9])
    @test select(a, 1, agg=+) == IndexedTable([1,2], [13,17])
    @test select(a, 2, agg=+) == IndexedTable([1,2], [14,16])
end

@test leftjoin(IndexedTable([1,1,1,2], [2,3,4,4], [5,6,7,8]),
               IndexedTable([1,1,3],   [2,4,4],   [9,10,12])) ==
                   IndexedTable([1,1,1,2], [2,3,4,4], [9, 6, 10, 8])

@test leftjoin(IndexedTable([1,1,1,2], [2,3,4,4], [5,6,7,8]),
               IndexedTable([1,1,2],   [2,4,4],   [9,10,12])) ==
                   IndexedTable([1,1,1,2], [2,3,4,4], [9, 6, 10, 12])

@test asofjoin(IndexedTable([:msft,:ibm,:ge], [1,3,4], [100,200,150]),
               IndexedTable([:ibm,:msft,:msft,:ibm], [0,0,0,2], [100,99,101,98])) ==
                   IndexedTable([:msft,:ibm,:ge], [1,3,4], [101, 98, 150])

@test asofjoin(IndexedTable([:AAPL, :IBM, :MSFT], [45, 512, 454], [63, 93, 54]),
               IndexedTable([:AAPL, :MSFT, :AAPL], [547,250,34], [88,77,30])) ==
                   IndexedTable([:AAPL, :MSFT, :IBM], [45, 454, 512], [30, 77, 93])

@test asofjoin(IndexedTable([:aapl,:ibm,:msft,:msft],[1,1,1,3],[4,5,6,7]),
               IndexedTable([:aapl,:ibm,:msft],[0,0,0],[8,9,10])) ==
                   IndexedTable([:aapl,:ibm,:msft,:msft],[1,1,1,3],[8,9,10,10])

@test aggregate_vec(maximum,
                    IndexedTable([1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 3, 3, 3],
                                 [1, 4, 3, 5, 2, 0], presorted=true)) ==
                    IndexedTable([1, 1], [2, 3], [4, 5])

@test aggregate_vec([maximum, minimum],
                    IndexedTable([1, 1, 1, 1, 1, 1],
                                 [2, 2, 2, 3, 3, 3],
                                 [1, 4, 3, 5, 2, 0], presorted=true)) ==
                    IndexedTable([1, 1], [2, 3], Columns(maximum=[4, 5], minimum=[1, 0]))

@test convertdim(IndexedTable([1, 1, 1, 1, 1, 1],
                              [0, 1, 2, 3, 4, 5],
                              [1, 4, 3, 5, 2, 0], presorted=true), 2, x->div(x,3), vecagg=maximum) ==
                    IndexedTable([1, 1], [0, 1], [4, 5])

let A = rand(3,3), B = rand(3,3), C = rand(3,3)
    nA = convert(IndexedTable, A)
    nB = convert(IndexedTable, B)
    nB.index.columns[1][:] += 3
    @test merge(nA,nB) == convert(IndexedTable, vcat(A,B))
    nC = convert(IndexedTable, C)
    nC.index.columns[1][:] += 6
    @test merge(nA,nB,nC) == merge(nA,nC,nB) == convert(IndexedTable, vcat(A,B,C))
    merge!(nA,nB)
    @test nA == convert(IndexedTable, vcat(A,B))
end
