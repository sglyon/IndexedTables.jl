using IndexedTables
using TableTraits
using NamedTuples
using Base.Test

@testset "TableTraits" begin

source_it = NDSparse(Columns(a=[1,2,3]), Columns(b=[1.,2.,3.], c=["A","B","C"]))

@test isiterable(source_it) == true

target_array = collect(getiterator(source_it))

@test length(target_array) == 3
@test target_array[1] == @NT(a=1, b=1., c="A")
@test target_array[2] == @NT(a=2, b=2., c="B")
@test target_array[3] == @NT(a=3, b=3., c="C")

source_array = [@NT(a=1,b=1.,c="A"), @NT(a=2,b=2.,c="B"), @NT(a=3,b=3.,c="C")]

it1 = NDSparse(source_array)
@test length(it1) == 3
@test it1[1,1.].c == "A"
@test it1[2,2.].c == "B"
@test it1[3,3.].c == "C"

it2 = NDSparse(source_array, idxcols=[:a])
@test length(it2) == 3
@test it2[1] == @NT(b=1., c="A")
@test it2[2] == @NT(b=2., c="B")
@test it2[3] == @NT(b=3., c="C")

it3 = NDSparse(source_array, datacols=[:b, :c])
@test length(it3) == 3
@test it3[1] == @NT(b=1., c="A")
@test it3[2] == @NT(b=2., c="B")
@test it3[3] == @NT(b=3., c="C")

end
