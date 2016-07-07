export naturaljoin

## Joins

# Natural Join (Both NDSParse arrays must have the same number of columns, in the same order)

function naturaljoin(left::NDSparse, right::NDSparse, op::Function)
   lI = left.indexes
   rI = right.indexes
   lD = left.data
   rD = right.data

   ll, rr = length(lI), length(rI)

   # Guess the length of the result
   guess = min(ll, rr)

   # Initialize output array components
   I = Indexes(map(c->_sizehint!(similar(c,0), guess), lI.columns)...)
   data = _sizehint!(similar(lD, 0), guess)
   default = left.default

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
   NDSparse(I, data, default)
end


# Column-wise filtering (Accepts conditions as column-function pairs)
# Example: select(arr, 1 => x->x>10, 3 => x->x!=10 ...)

function Base.select(arr::NDSparse, conditions::Pair...)
   indxs = 1:length(arr)
   cols = arr.indexes.columns
   for (c,f) in conditions
      indxs = intersect(indxs, filter(i->f(cols[c][i]), indxs))
   end
   NDSparse(Indexes(map(x->x[indxs], cols)...), arr.data[indxs], arr.default)
end

# Filter on data field
function Base.filter(fn::Function, arr::NDSparse)
   cols = arr.indexes.columns
   data = arr.data
   indxs = filter(i->fn(data[i]), eachindex(data))
   NDSparse(Indexes(map(x->x[indxs], cols)...), data[indxs], arr.default)
end
