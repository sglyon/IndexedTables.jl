using OnlineStatsBase
export groupreduce, groupby

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

    fs, input, T = init_arrays(f, data, reduced_type, false)
    dest_data = similar(arrayof(T), 0)

    groupreduce_to!(fs, key, input, dest_key, dest_data, perm)

    convert(NextTable, dest_key, dest_data)
end

function groupby(f, t::NextTable, by=pkeynames(t); select=rows(t))

    if !isa(by, Tuple)
        by=(by,) # this will help keep the column name
    end
    if !isa(f, Tup)
        f=(f,)
    end

    key  = rows(t, by)
    data = rows(t, select)

    perm = sortpermby(t, by)
    fs, input, T = init_arrays(f, data, reduced_type, true)
    # Note: we're not using T here, we'll let _groupby figure it out
    dest_key, dest_data = _groupby(fs, key, input, perm)

    convert(NextTable, dest_key, dest_data)
end

struct SubArrClosure{R}
    r::R
end

(f::SubArrClosure)(x) = SubArray(x, f.r)

function _groupby(f, key, data, perm, dest_key=similar(key,0),
                  dest_data=nothing, i1=1)
    n = length(key)
    cs = columns(data)
    while i1 <= n
        i = i1+1
        while i <= n && roweq(key, perm[i], perm[i1])
            i += 1
        end
        # needed this hack to avoid allocations. i loses type info
        #val = _apply(f, map(x->SubArray(x, (perm[i1:(i-1)],)), cs))
        val = _apply(f, map(SubArrClosure((perm[i1:(i-1)],)), cs))
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
