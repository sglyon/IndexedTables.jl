using DataValues

# product-join on equal lkey and rkey starting at i, j
function joinequalblock(typ, grp, f, I, data, lout, rout, lkey, rkey, ldata, rdata, lperm, rperm, i,j)
    ll = length(lkey)
    rr = length(rkey)

    i1 = i
    j1 = j
    while i1 < ll && rowcmp(lkey, lperm[i1], lkey, lperm[i1+1]) == 0
        i1 += 1
    end
    while j1 < rr && rowcmp(rkey, rperm[j1], rkey, rperm[j1+1]) == 0
        j1 += 1
    end
    if !isa(typ, Val{:anti})
        if isa(grp, Val{false})
            for x=i:i1
                for y=j:j1
                    push!(I, lkey[lperm[x]])
                    # optimized push! method for when
                    # concatenating
                    _push!(Val{:both}(), f, data,
                           lout, rout, ldata, rdata,
                           lperm[x], rperm[y], NA, NA)
                end
            end
        else
            push!(I, lkey[lperm[i]])
            group = similar(eltype(data), 0)
            for x=i:i1
                for y=j:j1
                    push!(group, f(ldata[lperm[x]], rdata[rperm[y]]))
                end
            end
            push!(data, group)
        end
    end
    return i1,j1
end

# copy without allocating struct
@inline function _push!{part}(::Val{part}, f::typeof(concat_tup), data,
                              lout, rout, ldata, rdata,
                              lidx, ridx, lnull, rnull)
    if part === :left
        pushrow!(lout, ldata, lidx)
        push!(rout, rnull)
    elseif part === :right
        pushrow!(rout, rdata, ridx)
        push!(lout, lnull)
    elseif part === :both
        pushrow!(lout, ldata, lidx)
        pushrow!(rout, rdata, ridx)
    end
end

@inline function _push!{part}(::Val{part}, f, data,
                              lout, rout, ldata, rdata,
                              lidx, ridx, lnull, rnull)
    if part === :left
        push!(data, f(ldata[lidx], rnull))
    elseif part === :right
        push!(data, f(lnull, rdata[ridx]))
    elseif part === :both
        push!(data, f(ldata[lidx], rdata[ridx]))
    end
end

@inline function _append!{part}(p::Val{part}, f, data,
                              lout, rout, ldata, rdata,
                              lidx, ridx, lnull, rnull)
    if part === :left
        for i in lidx
            _push!(p, f, data, lout, rout, ldata, rdata,
                   i, ridx, lnull, rnull)
        end
    elseif part === :right
        for i in ridx
            _push!(p, f, data, lout, rout, ldata, rdata,
                   lidx, i, lnull, rnull)
        end
    end
end

function _join(typ, grp, f, I, data, lout, rout, lnull, rnull,
               lkey, rkey, ldata, rdata, lperm, rperm)

    ll, rr = length(lkey), length(rkey)

    i = j = prevj = 1

    while i <= ll && j <= rr
        c = rowcmp(lkey, lperm[i], rkey, rperm[j])
        if c < 0
            if isa(typ, Union{Val{:outer}, Val{:left}, Val{:anti}})
                push!(I, lkey[lperm[i]])
                if isa(grp, Val{true})
                    # empty group
                    push!(data, similar(eltype(data), 0))
                else
                    _push!(Val{:left}(), f, data, lout, rout,
                           ldata, rdata, lperm[i], 0, lnull, rnull)
                end
            end
            i += 1
        elseif c==0
            i, j = joinequalblock(typ, grp, f, I, data, lout, rout,
                                  lkey, rkey, ldata, rdata, lperm, rperm,
                                  i, j)
            i += 1
            j += 1
        else
            if isa(typ, Val{:outer})
                push!(I, rkey[rperm[j]])
                if isa(grp, Val{true})
                    # empty group
                    push!(data, similar(eltype(data), 0))
                else
                    _push!(Val{:right}(), f, data, lout, rout,
                           ldata, rdata, 0, rperm[j], lnull, rnull)
                end
            end
            j += 1
        end
    end

    # finish up
    if !isa(typ, Val{:inner})
        if isa(typ, Union{Val{:left}, Val{:anti}, Val{:outer}}) && i <= ll
            append!(I, lkey[i:ll])
            if isa(grp, Val{true})
                # empty group
                append!(data, fill(similar(eltype(data), 0), length(i:ll)))
            else
                _append!(Val{:left}(), f, data, lout, rout,
                       ldata, rdata, lperm[i:ll], 0, lnull, rnull)
            end
        elseif isa(typ, Val{:outer}) && j <= rr
            append!(I, rkey[j:rr])
            if isa(grp, Val{true})
                # empty group
                append!(data, fill(similar(eltype(data), 0), length(j:rr)))
            else
                _append!(Val{:right}(), f, data, lout, rout,
                       ldata, rdata, 0, rperm[j:rr], lnull, rnull)
            end
        end
    end

    convert(NextTable, I, data, presorted=true)
end

nullrow(t::Type{<:Tuple}) = tuple(map(x->x(), [t.parameters...])...)
nullrow(t::Type{<:NamedTuple}) = t(map(x->x(), [t.parameters...])...)
nullrow(t::Type) = t()

function _init_output(typ, grp, f, ldata, rdata, lkey, rkey)
    lnull = nothing
    rnull = nothing
    loutput = nothing
    routput = nothing

    if isa(grp, Val{false})
        if isa(typ, Union{Val{:left}, Val{:inner}, Val{:anti}})
            # left cannot be null in these joins
            left_type = eltype(ldata)
        else
            left_type = map_params(x->DataValue{x}, eltype(ldata))
            lnull = nullrow(left_type)
        end
        if isa(typ, Val{:inner})
            # right cannot be null in innnerjoin
            right_type = eltype(rdata)
        else
            right_type = map_params(x->DataValue{x}, eltype(rdata))
            rnull = nullrow(right_type)
        end

        if f === concat_tup
            out_type = concat_tup_type(left_type, right_type)
            # separate left and right parts of the output
            # this is for optimizations in _push!
            loutput = similar(arrayof(left_type), 0)
            routput = similar(arrayof(right_type), 0)
            data = rows(concat_tup(columns(loutput), columns(routput)))
        else
            out_type = _promote_op(f, left_type, right_type)
            data = similar(arrayof(out_type), 0)
        end
    else
        left_type = eltype(ldata)
        right_type = eltype(rdata)
        out_type = _promote_op(f, left_type, right_type)
        data = similar(arrayof(arrayof(out_type)), 0)
    end
    
    if isa(typ, Val{:inner})
        guess = min(length(lkey), length(rkey))
    else
        guess = length(lkey)
    end

    _sizehint!(similar(lkey,0), guess), _sizehint!(data, guess), loutput, routput, lnull, rnull
end

function excludecols(t::NextTable, cols)
    ns = colnames(t)
    cols = Iterators.filter(c->c in ns || isa(c, Integer),
                  map(x->isa(x, As) ? x.src : x, cols))
    (setdiff(ns, cols)...)
end

function Base.join(f, left::NextTable, right::NextTable;
               how=:inner, group=false,
               lkey=pkeynames(left), rkey=pkeynames(right),
               lselect=excludecols(left, lkey),
               rselect=excludecols(right, rkey))

    lperm = sortpermby(left, lkey)
    rperm = sortpermby(right, rkey)

    lkey = rows(left, lkey)
    rkey = rows(right, rkey)

    ldata = rows(left, lselect)
    rdata = rows(right, rselect)

    typ, grp = Val{how}(), Val{group}()
    I, data, lout, rout, lnull, rnull = _init_output(typ, grp, f, ldata, rdata, lkey, rkey)

    _join(typ, grp, f, I, data, lout, rout, lnull, rnull, lkey, rkey,
          ldata, rdata, lperm, rperm)
end

function Base.join(left::NextTable, right::NextTable; kwargs...)
    join(concat_tup, left, right; kwargs...)
end

for (fn, how) in [:naturaljoin =>     (:inner, false, concat_tup),
                  :leftjoin =>        (:left,  false, concat_tup),
                  :outerjoin =>       (:outer, false, concat_tup),
                  :antijoin =>        (:anti,  false, (x, y) -> x),
                  :naturalgroupjoin =>(:inner, true, concat_tup),
                  :leftgroupjoin =>   (:left,  true, concat_tup),
                  :outergroupjoin =>  (:outer, true, concat_tup)]

    how, group, f = how

    @eval export $fn

    @eval function $fn(f, left::NextTable, right::NextTable; kwargs...)
        join(f, left, right; group=$group, how=$(Expr(:quote, how)), kwargs...)
    end

    @eval function $fn(left::NextTable, right::NextTable; kwargs...)
        $fn($f, left, right; kwargs...)
    end
end
