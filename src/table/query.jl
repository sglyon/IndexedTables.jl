filt_by_col!(f, col, indxs) = filter!(i->f(col[i]), indxs)

"""
`select(arr::NextTable, conditions::Pair...)`

Filter based on index columns. Conditions are accepted as column-function pairs.

Example: `select(arr, 1 => x->x>10, 3 => x->x!=10 ...)`
"""
function Base.select(arr::NextTable, conditions::Pair...)
    indxs = [1:length(arr);]
    for (c,f) in conditions
        filt_by_col!(f, column(arr, c), indxs)
    end
    arr[indxs]
end

"""
`select(arr::NextTable, which::DimName...)`

Select a subset of columns.
"""
function Base.select(t::NextTable, which::DimName...)
    # TODO: Keep permutations which are subset of `which`
    NextTable(rows(t, which))
end

# Filter on data field
function Base.filter(fn::Function, t::NextTable)
    indxs = filter(i->fn(t[i]), eachindex(t))
    t[indxs]
end

