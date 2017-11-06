export groupreduce, groupby

"""
    filter(conditions, t)

Filter rows in `t` according to conditions
"""
function Base.filter(conditions, t::Dataset)
    indxs = [1:length(t);]
    for (c,f) in conditions
        filt_by_col!(f, rows(t, c), indxs)
    end
    subtable(t, indxs)
end

Base.filter(pred::NamedTuple, t::Dataset) = filter(zip(fieldnames(pred), astuple(pred)), t)

function Base.filter(fn::Function, t::Dataset)
    indxs = filter(i->fn(t[i]), eachindex(t))
    t[indxs]
end


"""
`select(t::NextTable, which::DimName...)`

Select a subset of columns.
"""
function Base.select(t::AbstractIndexedTable, which::DimName...)
    ColDict(t)[which]
end

# Filter on data field
function groupreduce_to!(f, key, data, dest_key, dest_data, perm)
    n = length(key)
    i1 = 1
    while i1 <= n
        val = data[perm[i1]]
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            val = f(val, data[perm[i]])
            i += 1
        end
        push!(dest_key, key[perm[i1]])
        push!(dest_data, val)
        i1 = i
    end
    dest_key, dest_data
end

function bestname(t, col, fallback)
    if isa(col, Pair{Symbol, <:Any})
        col[1]
    elseif isa(col, Union{Symbol, Integer}) && eltype(t) <: NamedTuple
        # keep the name of the original column
        name = fieldnames(eltype(t))[colindex(t, col)]
    else
        fallback
    end
end

function groupreduce(f, t::NextTable, by=pkeynames(t);
                     select=rows(t), name=nothing)

    key  = rows(t, by)
    data = rows(t, select)
    perm = sortpermby(t, by)

    dest_key = similar(key, 0)
    dest_data = similar(data, 0)

    groupreduce_to!(f, key, data, dest_key, dest_data, perm)

    if !(typeof(dest_data) <: Columns)
        if name == nothing
            name = bestname(t, select, Symbol(f))
        end
        dest_data = Columns(dest_data, names=[name])
    end

    if !(typeof(dest_key) <: Columns)
        name = bestname(t, by, :key)
        dest_key = Columns(dest_key, names=[name])
    end

    convert(NextTable, dest_key, dest_data)
end

function groupby(f, t::NextTable, by=pkeynames(t); select=rows(t), name=nothing)
    key  = rows(t, by)
    data = rows(t, select)

    perm = sortpermby(t, by)

    if isa(f, AbstractVector)
        T = isa(name, AbstractVector) ?
            namedtuple(name...) : namedtuple(map(Symbol, f)...)

        f = T(f...)
    end

    dest_key, dest_data = _groupby(f, key, data, perm)

    if !(typeof(dest_data) <: Columns)
        if name == nothing
            name = bestname(t, select, Symbol(f))
        end
        dest_data = Columns(dest_data, names=[name])
    end

    if !(typeof(dest_key) <: Columns)
        name = bestname(t, by, :key)
        dest_key = Columns(dest_key, names=[name])
    end

    convert(NextTable, dest_key, dest_data)
end

@inline _apply_many(f::Tup, xs) = map(g->g(xs), f)
@inline _apply_many(f, xs) = f(xs)

function _groupby(f, key, data, perm, dest_key=similar(key,0),
                  dest_data=nothing, i1=1)
    n = length(key)
    while i1 <= n
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            i += 1
        end
        val = _apply_many(f, data[perm[i1:(i-1)]])
        push!(dest_key, key[perm[i1]])
        if dest_data === nothing
            newdata = [val]
            if isa(val, Tup)
                newdata = convert(Columns, newdata)
            end
            return _groupby(f, key, data, perm, dest_key, newdata, i)
        else
            push!(dest_data, val)
        end
        i1 = i
    end
    (dest_key, dest_data===nothing ? Union{}[] : dest_data)
end
