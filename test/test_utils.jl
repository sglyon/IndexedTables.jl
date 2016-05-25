using Base.Test
using NDSparseData

let lo=1, hi=10
    for left in [true, false]
        for right in [true, false]
            I = NDSparseData.Interval{Int,left,right}(lo, hi)
            @test (lo in I) == left
            @test (hi in I) == right
            @test (hi+1 in I) == false
            @test (lo-1 in I) == false
            @test (lo+1 in I) == true
        end
    end
end

let a = NDSparse([12,21,32], [52,41,34], [11,53,150]), b = NDSparse([12,23,32], [52,43,34], [56,13,10])
    p = collect(NDSparseData.product(a, b))
    @test p == [(11,56), (11,13), (11,10), (53,56), (53,13), (53,10), (150,56), (150,13), (150,10)]

    p = collect(NDSparseData.product(a, b, a))
    @test p == [(11,56,11),(11,56,53),(11,56,150),(11,13,11),(11,13,53),(11,13,150),(11,10,11),(11,10,53),(11,10,150),
                (53,56,11),(53,56,53),(53,56,150),(53,13,11),(53,13,53),(53,13,150),(53,10,11),(53,10,53),(53,10,150),
                (150,56,11),(150,56,53),(150,56,150),(150,13,11),(150,13,53),(150,13,150),(150,10,11),(150,10,53),(150,10,150)]
end
