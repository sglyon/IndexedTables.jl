@testset "NextTable constructor" begin
    cs = Columns([1], [2])
    t = NextTable(cs)
    @test t.primarykey == Int[]
    @test t.columns == [(1,2)]
    @test column(t.columns,1) !== cs.columns[1]
    t = NextTable(cs, copy=false)
    @test column(t.columns,1) === cs.columns[1]
    t = NextTable(cs, copy=false, primarykey=[1])
    @test column(t.columns,1) === cs.columns[1]
    cs = Columns([2, 1], [3,4])
    t = NextTable(cs, copy=false, primarykey=[1])
    @test t.primarykey == Int[1]
    cs = Columns([2, 1], [3,4])
    t = NextTable(cs, copy=false, primarykey=[1])
    @test column(t.columns,1) === cs.columns[1]
    @test t.primarykey == Int[1]
    @test t.columns == [(1,4), (2,3)]

    cs = Columns(x=[2, 1], y=[3,4])
    t = NextTable(cs, copy=false, primarykey=:x)
    @test column(t.columns,1) === cs.columns.x
    @test t.primarykey == Int[1]
    @test t.columns == [@NT(x=1,y=4), @NT(x=2,y=3)]

    cs = Columns([2, 1], [3,4])
    t = NextTable(cs, presorted=true, primarykey=[1])
    @test t.primarykey == Int[1]
    @test t.columns == [(2,3), (1,4)]
end

@testset "Getindex" begin
    cs = Columns(x=[1.2, 3.4], y=[3,4])
    t = NextTable(cs, copy=false, primarykey=:x)

    @test t[1] == @NT(x=1.2, y=3)
    @test t[[1,2]].columns == t.columns
    @test_throws ArgumentError t[[2,1]]
end

import IndexedTables: primaryperm, sortpermby, best_perm_estimate

@testset "sortpermby" begin
    cs = Columns(x=[1,1,2,1,2], y=[1,2,2,1,2], z=[7,6,5,4,3])
    t = NextTable(cs, copy=false, primarykey=[:x, :y])
    # x=[1,1,1,2,2], y=[1,1,2,2,2], z=[7,4,6,5,3]
    @test column(t, :z) == [7,4,6,5,3]
    @test issorted(rows(t, (:x,:y)))
    @test sortpermby(t, (:y, :z)) == [2,1,5,4,3]
    @test t.perms[1].perm == [2,1,5,4,3]
    perms = [primaryperm(t), t.perms;]

    @test sortpermby(t, (:y, :x), cache=false) == [2,1,3,5,4]
    @test length(t.perms) == 1

    # fully known
    @test best_perm_estimate(perms, [1,2]) == (2, Base.OneTo(5))
    @test best_perm_estimate(perms, [2,3]) == (2, [2, 1, 5, 4, 3])

    # first column known
    @test best_perm_estimate(perms, [1,3]) == (1, Base.OneTo(5))
    @test best_perm_estimate(perms, [2,1]) == (1, [2, 1, 5, 4, 3])

    # nothing known
    @test best_perm_estimate(perms, [3,1]) == (0, nothing)
end

@testset "select" begin
    a = NextTable([12,21,32], [52,41,34], [11,53,150], primarykey=[1,2])
    b = NextTable([12,23,32], [52,43,34], [56,13,10], primarykey=[1,2])

    c = select(a, 1=>x->x<30, 2=>x->x>40)
    @test rows(c) == [(12,52,11), (21,41,53)]
    @test c.primarykey == [1,2]

    c = select(a, 1, 2)
    @test c == NextTable(column(a, 1), column(a, 2))
    @test c.primarykey == [1,2]
end

@testset "groupreduce" begin
    a = NextTable([1, 1, 2], [2, 3, 3], [4, 5, 2], primarykey=(1,2))
    b = NextTable(Columns(a=[1, 1, 2], b=[3, 2, 2], c=[4, 5, 2]), primarykey=(1,2))

    @test groupreduce(min, a, select=3) == a
    @test groupreduce(min, b, select=3) == b
    @test rows(groupreduce(min, b, 2, select=3)) == Columns(b=[2,3], c=[2,4])
end

@testset "groupby" begin
    x = Columns(a=[1, 1, 1, 1, 1, 1],
                b=[2, 2, 2, 3, 3, 3],
                c=[1, 4, 3, 5, 2, 0])

    a = NextTable(x, primarykey=[1,2], presorted=true)
    @test groupby(maximum, a, select=3) == NextTable(Columns(a=[1, 1], b=[2, 3], c=[4, 5]))

    @test groupby([maximum, minimum], a, select=3) ==
                NextTable(Columns(a=[1, 1], b=[2, 3],
                                  maximum=[4, 5], minimum=[1, 0]))

    @test groupby(length, a, select=3, name=:length) ==
                NextTable(Columns(a=[1, 1], b=[2, 3], length=[3, 3]))

    @test groupby([maximum, minimum], a, select=3, name=[:max, :min]) ==
                NextTable(Columns(a=[1, 1], b=[2, 3],
                                  max=[4, 5], min=[1, 0]))
end
