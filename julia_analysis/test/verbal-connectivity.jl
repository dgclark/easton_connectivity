using Base.Test


src = begin
  curr_folder = pwd()
  proj_folder = dirname(curr_folder)
  joinpath(proj_folder, "src", "verbal-connectivity.jl")
end

include(src)

corrs = [1.0 0.7;
         0.7 1.0]

dist, prox = corrs_to_mats(corrs, ["a", "b"])
expected_prox = [0.0 0.7;
                 0.7 0.0]
expected_dist = [0.0 0.3;
                 0.3 0.0]

function within_arr(a::AbstractArray{Float64}, b::AbstractArray{Float64}, diff::Float64=10e-6)
  all(abs(a - b) .< diff)
end

@test within_arr(prox, expected_prox)
@test within_arr(dist, expected_dist)


function within(a::Float64, b::Float64, diff::Float64=10e-6)
  abs(a - b) < diff
end

dist, prox = corr_to_dist_prox(.3)
@test within(dist, .7)
@test within(prox, .3)

dist, prox = corr_to_dist_prox(.7)
@test within(dist, .3)
@test within(prox, .7)

dist, prox = corr_to_dist_prox(-.3)
@test within(dist, 1.0)
@test within(prox, 0.0)


###local metrics
distance = [0.0 0.2 0.3;
            0.2 0.0 0.5;
            0.3 0.5 0.0]
proximity = [0.0 0.8 0.7;
             0.8 0.0 0.5;
             0.7 0.5 0.0]
lm = calc_local_graph_metrics(distance, proximity, ["a", "b", "c"])

function test_node{T}(node::ASCIIString, field::Symbol, expected_val::T)
  @test within(lm[lm[:node].==node, field][1], expected_val)
end

test_node("a", :degree, 1.5)
test_node("b", :degree, 1.3)
test_node("c", :degree, 1.2)

cub_prod = (a, b, c) -> (a * b * c)^(1/3)
n_deg = r -> sum(proximity[r, :])/maximum(proximity)
test_node("a", :cluster, cub_prod(0.8, 0.5, 0.7)/n_deg(1))
test_node("b", :cluster, cub_prod(0.8, 0.7, 0.5)/n_deg(2))
test_node("c", :cluster, cub_prod(0.7, 0.8, 0.5)/n_deg(3))


hops = [1 3 2;
        3 2 3;
        1 1 3]

cnts = pass_counts!(hops)
@test cnts == [0,1,1]

degrees = [0.2, 0.1, 0.3]
weights = [0.0 0.1 0.2;
           0.1 0.0 0.3;
           0.2 0.3 0.0]
lmat = laplace_matrix(degrees, weights)

expected = [0.2 -0.1 -0.2;
            -0.1 0.1 -0.3;
            -0.2 -0.3 0.3]

@test expected == lmat


###Compare Vals
d1 = Dict{Symbol, Float64}[Dict(:a => 3, :b=>6), Dict(:a => 5, :b=>8)]
d2 = Dict{Symbol, Float64}[Dict(:a => 1, :b=>3), Dict(:a => 3, :b=>5)]
cmp = (x::Vector{Float64}, y::Vector{Float64}) -> mean(x) - mean(y)

cmpks = Symbol[:a]
@test compare_global_vals(d1, d2, cmpks, cmp) == Dict(:a => 2.0)

cmpks = Symbol[:a, :b]
@test compare_global_vals(d1, d2, cmpks, cmp) == Dict(:a => 2.0, :b => 3.0)

nodes = ["1", "2"]
mk_df = (a, b) -> DataFrame(nodes=copy(nodes), a=a, b=b)
df1s = DataFrame[mk_df([1.0, 2.0], [4.0, 5.0]), mk_df([3.0, 4.0], [6.0, 7.0])]
df2s = DataFrame[mk_df([3.0, 5.0], [3.0, 3.0]), mk_df([5.0, 7.0], [5.0, 5.0])]

expected = DataFrame(nodes=nodes, a=[-2.0, -3.0], b=[1.0, 2.0])

cmpks=Symbol[:a, :b]
@test compare_local_vals(df1s, df2s, nodes, cmpks, cmp) == expected


df1 = DataFrame(node=["a", "b"], degree=[.1, .2], cluster=[.3, .4])
df2 = DataFrame(node=["a", "b"], degree=[.2, .3], cluster=[.1, .2])
expected = DataFrame(node=["a", "b"], degree_mean=[.15, .25],
                     cluster_mean=[.2, .3])
actual = calc_local_stats([df1, df2], [df2])
@test  within_arr(expected[:degree_mean], actual[:degree_mean])
@test  within_arr(expected[:cluster_mean], actual[:cluster_mean])
