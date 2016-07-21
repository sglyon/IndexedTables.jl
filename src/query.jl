export naturaljoin

## Joins

# Natural Join (Both NDSParse arrays must have the same number of columns, in the same order)

function naturaljoin(left::NDSparse, right::NDSparse, op::Function)
   flush!(left); flush!(right)
   lI = left.indexes
   rI = right.indexes
   lD = left.data
   rD = right.data

   ll, rr = length(lI), length(rI)

   # Guess the length of the result
   guess = min(ll, rr)

   # Initialize output array components
   I = Indexes(map(c->_sizehint!(similar(c,0), guess), lI.columns)...)
   data = _sizehint!(similar(lD, typeof(op(lD[1],rD[1])), 0), guess)

   # Match and insert rows
   i = j = 1

   while i <= ll && j <= rr
      lt, rt = lI[i], rI[j]
      c = cmp(lt, rt)
      if c == 0
         pushrow!(I, lt)
         push!(data, op(lD[i], rD[j]))
         i += 1
         j += 1
      elseif c < 0
         i += 1
      else
         j += 1
      end
   end

   # Generate final datastructure
   NDSparse(I, data)
end


# Column-wise filtering (Accepts conditions as column-function pairs)
# Example: select(arr, 1 => x->x>10, 3 => x->x!=10 ...)

filt_by_col!(f, col, indxs) = filter!(i->f(col[i]), indxs)

function Base.select(arr::NDSparse, conditions::Pair...)
    flush!(arr)
    indxs = [1:length(arr);]
    cols = arr.indexes.columns
    for (c,f) in conditions
        filt_by_col!(f, cols[c], indxs)
    end
    NDSparse(Indexes(map(x->x[indxs], cols)...), arr.data[indxs])
end

# select a subset of columns
function Base.select(arr::NDSparse, cols::Int...; agg=nothing)
    flush!(arr)
    return NDSparse([copy(arr.indexes.columns[c]) for c in cols]..., copy(arr.data), agg=agg)
end

# Filter on data field
function Base.filter(fn::Function, arr::NDSparse)
   flush!(arr)
   cols = arr.indexes.columns
   data = arr.data
   indxs = filter(i->fn(data[i]), eachindex(data))
   NDSparse(Indexes(map(x->x[indxs], cols)...), data[indxs])
end
