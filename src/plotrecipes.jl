import RecipesBase

RecipesBase.@recipe function f(o::AbstractIndexedTable, x::Symbol, y::Symbol)
    xlab --> x 
    label --> y
    select(o, x), select(o, y)
end

RecipesBase.@recipe function f(o::AbstractIndexedTable, x::Symbol, y::AbstractVector{Symbol})
    for yi in y 
        RecipesBase.@series begin 
            o, x, yi 
        end
    end
end