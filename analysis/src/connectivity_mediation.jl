include("utils.jl")
using DataFrames
using GLM
using PValueAdjust


# 1: covariates: age, sex, education, and total gray matter
# 2: run it again adding MCI as a covariate,
# 3: adding MCI and conversion status as covariates.

function add_to_default(add_covars::Vector{Symbol})
  covars = copy(default_covars)
  append!(covars, add_covars)
end

function mk_pre_process_fn(add_fns::Vector{Function}=Function[],
                           covars::Vector{Symbol}=default_covars)

  function pre_proc(df::DataFrame)
    calc_total_gray!(df)
    df[:flu] = Vector{Float64}(df[:raw])
    for fn = add_fns
      fn(df)
    end
    return covars
  end

  return pre_proc
end

function calc_is_mci!(df::DataFrame)
  df[:is_mci] = Float64[r[:dx] == "mci" for r in eachrow(df)]
end

pre_process_mci_conv = mk_pre_process_fn([calc_is_mci!], add_to_default([:is_mci, :conv]))
pre_process_no_cov = mk_pre_process_fn(Function[], Symbol[])


function flu_formula(var, covars=Symbol[])
  fm_str = length(covars) > 0 ? string("flu ~ $var + ", join(covars, " + ")) : "flu ~ $var"
  eval(parse(fm_str))
end


function view_covar_corrs()
  covars = pre_process_mci_conv(all_data)

  for c in covars
    fm = flu_formula(c)
    flu_r = lm(fm, all_data)
    ct = coeftable(flu_r)
    betacoef, tscore, pvalue = ct.mat[2, [1, 3, 4]]

    println("$c pvalue: $pvalue")
  end

end


function get_roi_flu_coefs(pre_process = mk_pre_process_fn())


  # This approach would give us three different graphs and we might be able
  # to say which edges are more likely to be disease specific.

  # btw, 288 * 287 = 82,656

    # Iterate through ROIs, checking to see if each ROI is associated with fluency
    # for gm_a in rois
    #    flu ~ beta1 * gm_a + covariates
    # end
    covars::Vector{Symbol} = pre_process(all_data)

    roi_cols = get_roi_cols(all_data)
    num_rois = length(roi_cols)

    flu_corrs = begin
      zeros_rois = () ->  zeros(Float64, num_rois)
      pvalues = zeros_rois()
      tscores = zeros_rois()
      betacoefs = zeros_rois()

      for (ix, roi) = enumerate(roi_cols)
        fm = flu_formula(roi, covars)
        flu_r = lm(fm, all_data)
        ct = coeftable(flu_r)
        betacoefs[ix], tscores[ix], pvalues[ix] = ct.mat[2, [1, 3, 4]]
      end

      ret = DataFrame(roi=roi_cols, pvalue=pvalues, tscore=tscores, betacoef=betacoefs)
      ret[:pvalue_adj] = padjust(ret[:pvalue], BenjaminiHochberg)

      println(covars)
      println("minimum pvalue: $(minimum(ret[:pvalue]))")
      # Use FDR to decide which ROIs to keep, put in list sig_rois
      # Also have to keep beta1 for each ROI.
      println("minimum p-adj: $(minimum(ret[:pvalue_adj]))")
      alpha=.05
      println("p count less than $alpha: $(sum(ret[:pvalue] .< alpha))")
      println("p_adj count less than $alpha: $(sum(ret[:pvalue_adj] .< alpha))")

      ret
    end

  return flu_corrs

end

function get_mediated_rois()
  betas::DataFrame = get_roi_flu_coefs()[[:betacoef, :roi]]
  rois::DataFrame = all_data[betas[:roi]]
end


function get_roi_corrs()

  mediated_rois::DataFrame = get_mediated_rois()

end

# Iterate through every pair of distinct items in sig_rois
# Create a matrix for p-values of dimension n_roi X n_roi
# for (gm_a,gm_b) in sig_rois X sig_rois if gm_a < gm_b -- i.e., go ahead and use ordered pairs
#    gm_a ~ beta2 * gm_b + covariates
#    gm_b ~ beta3 * gm_a + covariates
#    calculate p-values for beta2 and beta3 and place in corresponding locations in matrix
# Use FDR on p-values in matrix and threshold at p < 0.05
# This yields the possible connections among the ROIs.
# Create a list of (gm_a,gm_b) pairs s.t. FDR-corrected p < 0.05 for BOTH corresponding entries in the matrix,

# Make a function that does this:
# Takes as arguments fluency scores, list of ROI pairs, matrix with gray matter values for ROIs
# Returns a list of mediation effects (one for each ROI pair)
# Iterate through list of (gm_a,gm_b) pairs
# for (gm_a,gm_b) in roi_pairs
#     flu ~ beta4 * gm_a + beta5 * gm_b + covariates
#     Each pair gets 2 mediation effects. One is (beta1_gm_a - beta4) and the other is (beta1_gm_b - beta5)
#     Put these mediation effects in a list. In this case, you would include both (gm_a,gm_b) and (gm_b,gm_a)

# Perform permutation analysis... probably need 9,999 iterations.
# Repeat function above 9,999 times, each time using permuted fluency values, keeping the maximum mediation effect on each iteration
# and storing in a list of maximum values.

# Add the max mediation value from the unpermuted data to the list of maximum values to get 10,000
# Sort and select the 9,501st value (= threshold)
# Mediation effects from the original, unpermuted list that are >= threshold are considered significant
# Keep these edges and construct graph -- for a given pair of ROIs with both mediation effects meeting the threshold, you could
# actually average the mediation effects and create weighted graphs.
# Then we can create figures showing degree of ROIs, maybe betweenness centrality.
