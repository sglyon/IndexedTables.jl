using OnlineStatsBase
export groupreduce, groupby

@inline _apply(f::Series, g, x) = fit!(g, x)
@inline _apply(f::Tup, y::Tup, x::Tup) = map(_apply, f, y, x)
@inline _apply(f, y, x) = f(y, x)

@inline init_first(f, x) = x
@inline init_first(f::Series, x) = copy(f)
@inline init_first(f::Tup, x::Tup) = map(init_first, f, x)

# Initialize type of output, functions to apply, input and output vectors

function reduced_type(f, x)
    _promote_op((a,b)->_apply(f, init_first(f, a), b),
                eltype(x), eltype(x))
end

function init_groupreduce(f, x, noutput=0) # normal functions
    T = reduced_type(f, x)
    f, x, similar(arrayof(T), noutput)
end

function init_groupreduce(f::Tuple, input, noutput=0)
    reducers = map(f) do g
        if isa(g, Pair)
            name = g[1]
            if isa(g[2], Pair)
                selector, fn = g[2]
                vec = rows(input, selector)
            else
                vec = input
                fn = g[2]
            end
            (name, vec, fn)
        else
            (Symbol(g), input, g)
        end
    end
    ns = map(x->x[1], reducers)
    xs = map(x->x[2], reducers)
    fs = map(x->x[3], reducers)

    output_eltypes = map(reduced_type, fs, xs)

    NT = namedtuple(ns...)

    NT(fs...),
        rows(NT(xs...)),
        similar(arrayof(NT{output_eltypes...}), noutput) # output
end

function groupreduce_to!(f, key, data, dest_key, dest_data, perm)
    n = length(key)
    i1 = 1
    while i1 <= n
        val = init_first(f, data[perm[i1]])
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            _apply(f, val, data[perm[i]])
            val = _apply(f, val, data[perm[i]])
            i += 1
        end
        push!(dest_key, key[perm[i1]])
        push!(dest_data, val)
        i1 = i
    end
    dest_key, dest_data
end

"""
`groupreduce(f, t[, by::Selection]; select::Selection, name)`

Group rows by a given key (a [Selection](@ref)).
Apply a function `f` on the rows pair-wise to reduce each group to a single value.
"""
function groupreduce(f, t::NextTable, by=pkeynames(t);
                     select=excludecols(t, by))

    if isa(f, Pair)
        return groupreduce((f,), t, by, select=select)
    end
    if !isa(by, Tuple)
        by=(by,)
    end
    key  = rows(t, by)
    data = rows(t, select)
    perm = sortpermby(t, by)

    dest_key = similar(key, 0)
    dest_data = similar(data, 0)

    fs, input, dest_data = init_groupreduce(f, data)
    groupreduce_to!(fs, key, input, dest_key, dest_data, perm)

    convert(NextTable, dest_key, dest_data)
end

function groupby(f, t::NextTable, by=pkeynames(t); select=rows(t))
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

function _groupby(f, key, data, perm, dest_key=similar(key,0),
                  dest_data=nothing, i1=1)
    n = length(key)
    while i1 <= n
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            i += 1
        end
        val = _apply(f, data[perm[i1:(i-1)]])
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
