using DataValues

function naturaljoin(f, left::NextTable, right::NextTable;
                   lkey=pkeynames(left), rkey=pkeynames(right),
                   lselect=rows(left), rselect=rows(right))

    lperm = sortpermby(left, lkey)
    rperm = sortpermby(right, rkey)

    lkey = rows(left, lkey)
    rkey = rows(right, rkey)

    ldata = rows(left, lselect)
    rdata = rows(right, rselect)

    out_type = _promote_op(f, eltype(ldata), eltype(rdata))
    data = similar(arrayof(out_type), 0)

    _naturaljoin(f, data, lkey, rkey, ldata, rdata, lperm, rperm)
end

function naturaljoin(left::NextTable, right::NextTable; kwargs...)
    naturaljoin(concat_tup, left, right; kwargs...)
end

# product-join on equal lkey and rkey starting at i, j
function joinequalblock(f, I, data, lkey, rkey, ldata, rdata, lperm, rperm, i,j)
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
    for x=i:i1
        for y=j:j1
            push!(I, lkey[lperm[x]])
            push!(data, f(ldata[lperm[x]], rdata[rperm[y]]))
        end
    end
    return i1,j1
end


function _naturaljoin(f, data, lkey, rkey, ldata, rdata, lperm, rperm)
    ll, rr = length(lkey), length(rkey)

    # Guess the length of the result
    guess = min(ll, rr)

    # Initialize output array components
    I = _sizehint!(similar(lkey,0), guess)
    _sizehint!(data, guess)

    # Match and insert rows
    i = j = 1

    while i <= ll && j <= rr
        c = rowcmp(lkey, lperm[i], rkey, lperm[j])
        if c == 0
            i, j = joinequalblock(f, I, data, lkey, rkey,
                                  ldata, rdata, lperm, rperm, i, j)
            i += 1
            j += 1
        elseif c < 0
            i += 1
        else
            j += 1
        end
    end

    # Generate final datastructure
    convert(NextTable, I, data, presorted=true)
end

@testset "naturaljoin" begin
    t1 = NextTable([1,2,3,4], [5,6,7,8], primarykey=[1])
    t2 = NextTable([0,3,4,5], [5,6,7,8], primarykey=[1])
    t3 = NextTable([0,3,4,4], [5,6,7,8], primarykey=[1])
    t4 = NextTable([1,3,4,4], [5,6,7,8], primarykey=[1])
    @test naturaljoin(+, t1, t2, lselect=2, rselect=2) == NextTable([3,4], [13, 15])
    @test naturaljoin(t1, t2, lselect=2, rselect=2) == NextTable([3,4],[7,8],[6,7])
    @test naturaljoin(t1, t2) == NextTable([3,4],[3,4],[7,8],[3,4],[6,7])
    @test naturaljoin(+, t1, t3, lselect=2, rselect=2) == NextTable([3,4,4], [13, 15, 16])
    @test naturaljoin(+, t3, t4, lselect=2, rselect=2) == NextTable([3,4,4,4,4], [12, 14,15,15,16])
end


function leftjoin(f, left::NextTable, right::NextTable;
                  lkey=pkeynames(left), rkey=pkeynames(right),
                  lselect=rows(left), rselect=rows(right))
    
    lperm = sortpermby(left, lkey)
    rperm = sortpermby(right, rkey)

    lkey = rows(left, lkey)
    rkey = rows(right, rkey)

    ldata = rows(left, lselect)
    rdata = rows(right, rselect)

    rT = eltype(rdata)
    out_type = _promote_op(f, eltype(ldata), Union{DataValue{rT}, rT})
    data = similar(arrayof(out_type), 0)

    _leftjoin(f, data, lkey, rkey, ldata, rdata, lperm, rperm)
end

@inline _right(x, y) = isa(y, typeof(NA)) ? x : y

function leftjoin(left::NextTable, right::NextTable; kwargs...)
    leftjoin(_right, left, right; kwargs...)
end

function _leftjoin(f, data, lkey, rkey, ldata, rdata, lperm, rperm)
    ll, rr = length(lkey), length(rkey)

    I = _sizehint!(similar(lkey,0), length(lkey))
    _sizehint!(data, length(lkey))

    i = j = prevj = 1

    while i <= ll && j <= rr
        c = rowcmp(lkey, lperm[i], rkey, rperm[j])
        if c < 0
            push!(I, lkey[lperm[i]])
            push!(data, f(ldata[lperm[i]], NA))
            i += 1
        elseif c==0
            i, j = joinequalblock(f, I, data, lkey, rkey,
                                  ldata, rdata, lperm, rperm, i, j)
            i += 1
            j += 1
        else
            j += 1
        end
    end
    append!(I, lkey[i:ll])
    append!(data, f.(ldata[i:ll], (NA,)))

    convert(NextTable, I, data, presorted=true)
end

@testset "leftjoin" begin
    t1 = NextTable([1,2,3,4], [5,6,7,8], primarykey=[1])
    t2 = NextTable([0,3,4,5], [5,6,7,8], primarykey=[1])
    t3 = NextTable([0,3,4,4], [5,6,7,8], primarykey=[1])
    t4 = NextTable([1,3,4,4], [5,6,7,8], primarykey=[1])

    # default: take values from left
    @test leftjoin(t1, t2, lselect=2, rselect=2) == NextTable([1,2,3,4], [5,6,6,7])

    # null instead of missing row
    @test leftjoin(+, t1, t2, lselect=2, rselect=2) == NextTable([1,2,3,4], [NA, NA, 13, 15])

    @test leftjoin(t1, t2) == NextTable([1,2,3,4], [(1,5), (2,6), (3,6), (4,7)])
    @test leftjoin(+, t1, t3, lselect=2, rselect=2)  == NextTable([1,2,3,4,4],[NA,NA,13,15,16])
    @test leftjoin(+, t3, t4, lselect=2, rselect=2) == NextTable([0,3,4,4,4,4], [NA, 12, 14,15,15,16])
end
