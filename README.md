# NDSparseData.jl
N-dimensional sparse array structure in julia

## Introduction

This package provides a data structure called `NDSparse`.
This structure maps tuples of indices to data values.
Hence, it is similar to a hash table mapping tuples to values, but with a few key
differences.
First, the index tuples are stored columnwise, with one vector per index position:
there is a vector of first indices, a vector of second indices, and so on.
The index vectors are expected to be homogeneous to allow more efficient storage.
Second, the indices must have a total order, and are stored lexicographically sorted
(first by the first index, then by the second index, and so on, left-to-right).
While the indices must have totally-ordered types, the data values can be anything.
Finally, for purposes of many operations an `NDSparse` acts like an N-dimensional
array of its data values, where the number of dimensions is the number of index
columns.

## Construction

The `NDSparse` constructor accepts a series of vectors.
The last vector contains the data values, and the first N vectors contain the
indices for each of the N dimensions.
As an example, let's construct an array of the high temperatures for three days
in two cities:

    julia> using Dates
    julia> hitemps = NDSparse([fill("New York",3); fill("Boston",3)],
                              repmat(Date(2016,7,6):Date(2016,7,8), 2),
                              [91,89,91,95,83,76])
    NDSparse{Int64,Tuple{String,Date}}:
     ("Boston",2016-07-06) => 95
     ("Boston",2016-07-07) => 83
     ("Boston",2016-07-08) => 76
     ("New York",2016-07-06) => 91
     ("New York",2016-07-07) => 89
     ("New York",2016-07-08) => 91

Notice that the data was sorted first by city name, then date, giving a different
order than we initially provided.
On construction, `NDSparse` takes ownership of the columns and sorts them in place
(the original vectors are modified).

## Importing data

Importing data from column-based sources is straightforward.
For example, csv files can be imported using CSV.jl with the following snippet:

    NDSparse(CSV.read(filename).columns...)

Of course, this assumes the file already has the "data column" in the rightmost
position.
If not, the columns can be reordered first.

## Indexing

Most lookup and filtering operations on `NDSparse` are done via indexing.
Our `hitemps` array behaves like a 2-d array of integers, accepting two
indices:

    julia> hitemps["Boston", Date(2016,7,8)]
    76

If the given indices exactly match the element types of the index columns,
then the result is a scalar.
In other cases, a new `NDSparse` is returned, giving data for all matching
locations:

    julia> hitemps["Boston", :]
    NDSparse{Int64,Tuple{String,Date}}:
     ("Boston",2016-07-06) => 95
     ("Boston",2016-07-07) => 83
     ("Boston",2016-07-08) => 76

Like other arrays, `NDSparse` generates its data values when iterated.
This allows the usual reduction functions (among others) in Base to work:

    julia> maximum(hitemps["Boston", :])
    95

## Permuting dimensions

As with other multi-dimensional arrays, dimensions can be permuted to change
the sort order.
With `NDSparse` the interpretation of this operation is especially natural:
simply imagine passing the index columns to the constructor in a different order,
and repeating the sorting process:

    julia> permutedims(hitemps, [2, 1])
    NDSparse{Int64,Tuple{Date,String}}:
     (2016-07-06,"Boston") => 95
     (2016-07-06,"New York") => 91
     (2016-07-07,"Boston") => 83
     (2016-07-07,"New York") => 89
     (2016-07-08,"Boston") => 76
     (2016-07-08,"New York") => 91

Now the data is sorted first by date.
In some cases such dimension permutations are needed for performance.
The leftmost column is esssentially the primary key --- indexing is fastest
in this dimension.

## Select and aggregate

In some cases one wants to consider a subset of dimensions, for example
when producing a simplified summary of data.
This can be done by passing dimension (column) numbers to `select`:

    julia> select(hitemps, 2)
    NDSparse{Int64,Tuple{Date}}:
     (2016-07-06,) => 95
     (2016-07-06,) => 91
     (2016-07-07,) => 83
     (2016-07-07,) => 89
     (2016-07-08,) => 76
     (2016-07-08,) => 91

In this case, the result has multiple values for some indices, and so
does not fully behave like a normal array anymore.
However it is now suitable for aggregation: combining all values associated
with the same indices.
This can be done using `aggregate!`, which operates in place:

    julia> aggregate!(max, select(hitemps, 2))
    NDSparse{Int64,Tuple{Date}}:
     (2016-07-06,) => 95
     (2016-07-07,) => 89
     (2016-07-08,) => 91

The first argument to `aggregate!` specifies a function to use to combine
values.

`select` also supports filtering columns with arbitrary predicates, by
passing `column=>predicate` pairs:

    julia> select(hitemps, 2=>isfriday)
    NDSparse{Int64,Tuple{String,Date}}:
     ("Boston",2016-07-08) => 76
     ("New York",2016-07-08) => 91

## Iterators

Indexing makes a copy of the selected data, and therefore can be expensive.
As an alternative, it is possible to construct an iterator over a subset of
an `NDSparse`.
The `where` function accepts the same arguments as indexing, but instead
returns an iterator that generates the data values at the selected
locations:

    julia> bos = where(hitemps, "Boston", :);
    julia> first(bos)
    95

The `pairs` function is similar, except yields `index=>value` pairs (where
`index` is a tuple).

## Broadcasting

`broadcast` is used to combine data with slightly different dimensions.
For example, say we have an array of low temperatures for Boston broken
down by zip code:

    julia> lotemps = NDSparse(fill("Boston",6),
                              repeat(Date(2016,7,6):Date(2016,7,8), inner=2),
                              repmat([02108,02134], 3),
                              [71,70,67,66,65,66])
    NDSparse{Int64,Tuple{String,Date,Int64}}:
     ("Boston",2016-07-06,2108) => 71
     ("Boston",2016-07-06,2134) => 70
     ("Boston",2016-07-07,2108) => 67
     ("Boston",2016-07-07,2134) => 66
     ("Boston",2016-07-08,2108) => 65
     ("Boston",2016-07-08,2134) => 66

We want to compute the daily temperature range (high minus low).
Since we don't have high temperatures available per zip code, we will assume
the high temperatures are city-wide averages applicable to every zip code.
The `broadcast` function implements this interpretation of the data,
automatically repeating data along missing dimensions:

    julia> broadcast((x,y)->y-x, lotemps, hitemps)
    NDSparse{Int64,Tuple{String,Date,Int64}}:
     ("Boston",2016-07-06,2108) => 24
     ("Boston",2016-07-06,2134) => 25
     ("Boston",2016-07-07,2108) => 16
     ("Boston",2016-07-07,2134) => 17
     ("Boston",2016-07-08,2108) => 11
     ("Boston",2016-07-08,2134) => 10

`broadcast` currently only allows the first argument to have more dimensions,
so we had to pass a function that subtracts its first argument from its second
instead of just `-`.
Notice that `broadcast` also automatically performs an inner join, selecting
only rows that match.

`broadcast` currently matches dimensions based on element type.
Specifying dimensions to match manually, or based on column names, is planned
future functionality.

## Assignment

`NDSparse` supports indexed assignment just like other arrays, but there are
caveats.
Since data is stored in a compact, sorted representation, inserting a single
element is potentially very inefficient (`O(n)`, since it requires moving up to half
of the existing elements).
Therefore single-element insertions are accumulated into a temporary buffer to
amortize cost.

When the next whole-array operation (e.g. indexing or broadcast) is performed,
the temporary buffer is merged into the main storage.
This operation is called `flush!`, and can also be invoked explicitly.
The cost of this operation is `O(n*log(n)) + O(m)`, where `n` is the number
of inserted items and `m` is the number of existing items.
This means that the worst case occurs when alternating between inserting a
small number of items, and performing whole-array operations.
To the extent possible, insertions should be batched, and in general done
rarely.
