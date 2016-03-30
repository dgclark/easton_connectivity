using Base.Test
using HypothesisTests

include("test-helpers.jl")
@include_src "verbal-connectivity.jl"


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
d_low = DataFrame(a=[3, 5.])
d_high = DataFrame(a=[1., 3])
get_quants(k::Symbol) = quantile(d_high[k] - d_low[k], [.025, .975])
a_2dot5, a_97dot5 = get_quants(:a)

expected = DataFrame(
  a_mean_low = mean(d_low[:a]),
  a_mean_high = mean(d_high[:a]),
  a_std_low = std(d_low[:a]),
  a_std_high = std(d_high[:a]),
  a_2dot5 = get_quants(:a)[1],
  a_97dot5 = get_quants(:a)[2]
  )

@test compare_global_vals(d_low, d_high) == expected


nodes = ["L1", "L1", "L1", "R1", "R1", "R1"]

info = Dict()

@enum LowHigh low high
@enum Measure degree cluster

info[low] = Dict()
info[high] = Dict()

update_info(lh::LowHigh, m::Measure,
            l1_arr::AbstractVector{Float64},
            r1_arr::AbstractVector{Float64}) = info[lh][m] = Dict(
  "L1"=>collect(l1_arr), "R1"=>collect(r1_arr))

get_all(lh::LowHigh, m::Measure) = [info[lh][m]["L1"];  info[lh][m]["R1"]]

update_info(low, degree, .1:.1:.3, .2:.1:.4)
update_info(high, degree, .7:.1:.9, .1:.1:.3)

update_info(low, cluster, .7:.1:.9, .1:.1:.3)
update_info(high, cluster, .1:.1:.3, .2:.1:.4)

df_low = DataFrame(node=nodes, degree=get_all(low, degree), cluster=get_all(low, cluster))
df_high = DataFrame(node=nodes, degree=get_all(high, degree), cluster=get_all(high, cluster))

function node_info(n::AbstractString, m::Measure)
  low_arr, high_arr = info[low][m][n], info[high][m][n]
  low_mn, high_mn = mean(low_arr), mean(high_arr)
  t::OneSampleTTest = OneSampleTTest(high_arr, low_arr)
  low_mn, high_mn, pvalue(t), t.t
end

L1_degree_info = node_info("L1", degree)
R1_degree_info = node_info("R1", degree)
L1_cluster_info = node_info("L1", cluster)
R1_cluster_info = node_info("R1", cluster)

expected = DataFrame(node=["L1", "R1"],
                     degree_low_mean=[L1_degree_info[1], R1_degree_info[1]],
                     cluster_low_mean=[L1_cluster_info[1], R1_cluster_info[1]],
                     degree_high_mean=[L1_degree_info[2], R1_degree_info[2]],
                     cluster_high_mean=[L1_cluster_info[2], R1_cluster_info[2]],
                     cluster_p = [L1_cluster_info[3], R1_cluster_info[3]],
                     cluster_t = [L1_cluster_info[4], R1_cluster_info[4]],
                     degree_p = [L1_degree_info[3], R1_degree_info[3]],
                     degree_t = [L1_degree_info[4], R1_degree_info[4]]
                     )

@test compare_local_vals(df_low, df_high) == expected
