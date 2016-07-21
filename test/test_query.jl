using Base.Test
using NDSparseData

let a = NDSparse([12,21,32], [52,41,34], [11,53,150]), b = NDSparse([12,23,32], [52,43,34], [56,13,10])
    c = naturaljoin(a, b, +)
    @test c[12,52] == 67
    @test c[32,34] == 160
    @test length(c.indexes) == 2

    c = select(a, 1=>x->x<30, 2=>x->x>40)
    @test c[12,52] == 11
    @test c[21,41] == 53
    @test length(c.indexes) == 2

    c = filter(x->x>100, a)
    @test c[32,34] == 150
    @test length(c.indexes) == 1
end

let a = NDSparse([1,1,2,2], [1,2,1,2], [6,7,8,9])
    @test select(a, 1, agg=+) == NDSparse([1,2], [13,17])
    @test select(a, 2, agg=+) == NDSparse([1,2], [14,16])
end
