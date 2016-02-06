using DataFrames
using Graphs
using HypothesisTests
using PValueAdjust


function pass_counts!(hops::Matrix{Int64},
                     cnts::Array{Int64, 1}=zeros(Int64,size(hops)[1]))
  ixs::Array{Int64, 1} = range(1, size(hops)[1])
  for i in ixs
    for j in ixs[i+1:end]
      hop::Int64 = hops[i, j]
      if hop != j
        cnts[hop] = cnts[hop] + 1
      end
    end
  end
  return cnts
end


function laplace_matrix(degrees::AbstractArray{Float64, 1}, weights::AbstractMatrix{Float64})
  mat::Matrix{Float64} = eye(length(degrees))
  mat[mat .== 1.0] = degrees
  mat = mat - weights
  mat
end


function calc_local_graph_metrics(distance::Matrix{Float64}, proximity::Matrix{Float64}, nodes::Array{ASCIIString, 1})
  degrees::Array{Float64, 1} = sum(proximity, 1)[:]

  n_nodes::Int64 = length(nodes)
  ixs_nodes::Array{Tuple{Int64, ASCIIString}} = map(ie -> (ie[1], ie[2]), enumerate(nodes))
  clusters = begin
    degrees_n::Array{Float64, 1} = sum(proximity./maximum(proximity), 1)[:]
    ret = zeros(Float64, n_nodes)
    for (n1i, n1) in ixs_nodes
      for (n2i, n2) in ixs_nodes[n1i + 1: end]
        for (n3i, n3) in ixs_nodes[n2i + 1: end]
          p::Float64 = (proximity[n1i, n2i] * proximity[n2i, n3i] * proximity[n3i, n1i])^(1/3)
          ret[n1i] += p
          ret[n2i] += p
          ret[n3i] += p
        end
      end
    end
    ret./degrees_n
  end

  nxt = zeros(Int64, n_nodes, n_nodes)
  path_ds = floyd_warshall!(copy(distance), nxt)
  bw_centralities = pass_counts!(nxt)

  return DataFrame(node=nodes, degree = degrees, cluster=clusters, bw_centrality=bw_centralities)
end


function calc_global_graph_metrics(distance:: Matrix{Float64},
                                   proximity::Matrix{Float64},
                                   clusters::AbstractArray{Float64, 1},
                                   degrees::AbstractArray{Float64, 1})
  ret = Dict{Symbol, Vector{Float64}}()
  ret[:avg_path_length] = [mean(floyd_warshall(distance))]
  ret[:avg_cluster] = [mean(clusters)]
  ret[:small_worldness] = [ret[:avg_cluster][1]/ret[:avg_path_length][1]]

  ret[:algebraic_connectivity] = begin
    l_mat = laplace_matrix(degrees, proximity)
    lam, _ = eig(l_mat)
    [lam[2]]
  end

  return ret
end


function calc_graph_metrics(distance::Matrix{Float64},
                             proximity::Matrix{Float64},
                             nodes::Array{ASCIIString, 1})
  lm::DataFrame = calc_local_graph_metrics(distance, proximity, nodes)
  gm::DataFrame = calc_global_graph_metrics(distance, proximity, lm[:cluster], lm[:degree])
  return (lm, gm)
end

data_dir = begin
  root = dirname(pwd())
  joinpath(root, "data")
end


function median_split(samples::DataFrame)
  raw::Array{Float64, 1} = samples[:, :raw]
  median_score::Float64 = median(raw)
  eq_med::BitArray{1} = raw .== median_score
  gt_med::BitArray{1} = raw .> median_score
  lt_med::BitArray{1} = raw .< median_score
  lows::BitArray{1}, highs::BitArray{1} =
    sum(gt_med) > sum(lt_med) ? (lt_med | eq_med, gt_med) : (lt_med, gt_med | eq_med)
  (samples[lows, :id], samples[highs, :id])
end

data_f = f -> joinpath(data_dir, f)

function calc_corrs(ids::DataArray{UTF8String, 1}, rois::DataFrame)
  rois::Matrix{Float64} = Matrix(join(rois, DataFrame(id=ids), on=:id)[:, 2:end])
  cor(rois, rois)
end


function corr_to_dist_prox(cr::Float64)
  prox::Float64 = cr > 0.0 ? cr : 0.0
  dist::Float64 = 1.0 - prox
  dist, prox
end


function corrs_to_mats(corrs::Array{Float64, 2}, nodes::Array{ASCIIString, 1})
  const n_nodes::Int64 = length(nodes)
  edge_distances = zeros(Float64, n_nodes, n_nodes)
  edge_proximities = zeros(Float64, n_nodes, n_nodes)

  function set_vals!(arr::Array{Float64, 2}, si::Int64,  ti::Int64, v::Float64)
    arr[si, ti] = arr[ti, si] = v
  end

  edge_ix = 0
  for (si::Int64, src::ASCIIString) in enumerate(nodes)
    for (ti::Int64, target::ASCIIString) in enumerate(nodes[si+1:end])
      ti = ti + si
      edge_ix = edge_ix + 1

      local roi_cor = corrs[si, ti]::Float64
      dist::Float64, prox::Float64 = corr_to_dist_prox(roi_cor)
      set_vals!(edge_distances, si, ti, dist)
      set_vals!(edge_proximities, si, ti, prox)
    end
  end

  edge_distances, edge_proximities
end


function perm_metrics(num_perms=10, verbose=false)

  scores::DataFrame = readtable(data_f("animal_scores.csv"))[:, [:id, :raw]]
  const num_scores = size(scores)[1]::Int64

  function apply_both(fn::Function, d1, d2)
    fn(d1), fn(d2)
  end

  rois = readtable(data_f("rois_normalized.csv"))
  roi_names = ASCIIString["$n" for n in names(rois)[2:end]]
  num_rois = length(roi_names)

  len_rpt = num_rois * num_perms
  mk_zero_local = () -> zeros(Float64, len_rpt)
  local_df = () -> DataFrame(node=Array(ASCIIString, len_rpt),
                             degree=mk_zero_local(),
                             cluster=mk_zero_local(),
                             bw_centrality=mk_zero_local())

  low_lms::DataFrame = local_df()
  high_lms::DataFrame = local_df()

  mk_zero_global = () -> zeros(Float64, num_perms)
  global_df = () -> DataFrame(avg_path_length = mk_zero_global(),
                              avg_cluster = mk_zero_global(),
                              small_worldness = mk_zero_global(),
                              algebraic_connectivity = mk_zero_global())
  low_gms::DataFrame = global_df()
  high_gms::DataFrame = global_df()

  for p in range(1, num_perms)
    if verbose
        println("perm num: $p")
    end

    sample_ids::DataFrame = DataFrame(id=rand(scores[:, :id], num_scores))
    samples::DataFrame = join(sample_ids, scores, on=:id)

    low_ids::DataArray{UTF8String, 1}, high_ids::DataArray{UTF8String, 1} = median_split(samples)

    ((low_lm, low_gm), (high_lm, high_gm)) = apply_both(low_ids, high_ids) do ids
      corrs = calc_corrs(ids, rois)
      distance, proximity = corrs_to_mats(corrs, roi_names)
      lm, gm = calc_graph_metrics(distance, proximity, roi_names)
    end

    low_gms[p, :] = DataFrame(low_gm)
    high_gms[p, :] = DataFrame(high_gm)

    top::Int64 = (p - 1)*num_rois + 1
    bottom::Int64 = p*num_rois
    low_lms[top:bottom, :] = low_lm
    high_lms[top:bottom, :] = high_lm

  end

  ret = Dict{Symbol, Dict{Symbol, DataFrame}}()

  update_ret = (lms::DataFrame,
                gms::DataFrame,
                t::Symbol) -> ret[t] = Dict(:local => lms, :global => gms)

  update_ret(low_lms, low_gms, :low)
  update_ret(high_lms, high_gms, :high)

  ret

end


function rename_suffix(df::DataFrame,
                       orig_suf::ASCIIString,
                       new_suf::ASCIIString,
                       cols::Set{Symbol})
  rename!(df, [symbol(c, orig_suf)=>symbol(c, new_suf) for c in cols])
end


function compare_global_vals(low::DataFrame,
                             high::DataFrame)

  low_mean::DataFrame = aggregate(low, mean)
  high_mean::DataFrame = aggregate(high, mean)
  low_std::DataFrame = aggregate(low, std)
  high_std::DataFrame = aggregate(high, std)

  data_cols::Set{Symbol} = Set(names(high))
  map([(low_mean, "_mean", "_mean_low"), (high_mean, "_mean", "_mean_high"),
       (low_std, "_std", "_std_low"), (high_std, "_std", "_std_high")
       ]) do i
    rename_suffix(i[1]::DataFrame,
                  i[2]::ASCIIString,
                  i[3]::ASCIIString,
                  data_cols)
  end

  quants::DataFrame = begin
    d = Dict{Symbol, Vector{Float64}}()
    for k in data_cols
      lq, hq = quantile(high[k] - low[k], [.025, .975])
      d[symbol(k, "_2dot5")] = [lq]
      d[symbol(k, "_97dot5")] = [hq]
    end
    DataFrame(d)
  end

  hcat(low_mean, high_mean, low_std, high_std, quants)

end


function get_stats(low_df::DataFrame,
                   high_df::DataFrame,
                   node::ASCIIString,
                   col::Symbol)
  low::Array{Float64} = low_df[ low_df[:node] .== node, col]
  high::Array{Float64} = high_df[ high_df[:node] .== node, col]
  t::OneSampleTTest = OneSampleTTest(high, low)
  (pvalue(t), t.t)
end


function compare_local_vals(low_df::DataFrame,
                            high_df::DataFrame)

  data_cols = delete!(Set(names(high_df)), :node)

  low_mean::DataFrame = aggregate(low_df, :node, mean)
  high_mean::DataFrame = aggregate(high_df, :node, mean)

  rename_suffix(low_mean, "_mean", "_low_mean", data_cols)
  rename_suffix(high_mean, "_mean", "_high_mean", data_cols)

  stats::DataFrame = begin
    nodes::Array{ASCIIString} = low_mean[:node]
    num_nodes::Int64 = length(nodes)
    mk_zeros = () -> zeros(Float64, num_nodes)

    ret = DataFrame(node=nodes)
    function stat_cols_prep!(c::Symbol)
      c_p = symbol(c, "_p")
      c_t = symbol(c, "_t")
      ret[c_p] = mk_zeros()
      ret[c_t] = mk_zeros()
      c => (c_p, c_t)
    end

    stat_cols_lookup::Dict{Symbol, Tuple{Symbol, Symbol}} = Dict(
      map(stat_cols_prep!, data_cols)
      )

    for n::ASCIIString in nodes
      for c::Symbol in data_cols
        c_p, c_t = stat_cols_lookup[c]
        p, t = get_stats(low_df, high_df, n, c)
        curr_node::BitArray{1} = ret[:node] .== n
        ret[curr_node, c_p] = p
        ret[curr_node, c_t] = t
      end
    end
    ret
  end

  on_node = (df1::DataFrame, df2::DataFrame) -> join(df1, df2, on=:node)
  on_node(on_node(low_mean, high_mean), stats)

end


function compare_metrics(low::Dict{Symbol, DataFrame},
                         high::Dict{Symbol, DataFrame})

  gc::DataFrame = compare_global_vals(low[:global], high[:global])

  lc::DataFrame = begin
    local_p_keys = Symbol[:degree_p, :cluster_p, :bw_centrality_p]
    nodes = unique(high[:local][:node])
    df::DataFrame = compare_local_vals(low[:local], high[:local])
    for k in local_p_keys
      df[:, k] = padjust(df[:, k], BenjaminiYekutieli)
    end
    df
  end

  Dict(:global => gc, :local=>lc)
end


export roi_df_to_graph,
  main_part
