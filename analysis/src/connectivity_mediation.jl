include("utils.jl")



# 1: covariates: age, sex, education, and total gray matter
# 2: run it again adding MCI as a covariate,
# 3: adding MCI and conversion status as covariates.

correct = correct_rois_for_covars

function calc_is_mci!(df::DataFrame)
  df[:is_mci] = Float64[r[:dx] == "mci" for r in eachrow(df)]
end

function covar_preprocess(df::DataFrame)
  calc_total_gray!(df)
  calc_is_mci!(df)
end

function correct_mci()
  correct_rois_for_covars(covars=[:age, :sex, :edu, :total_gray, :is_mci],
                          covar_preprocess=covar_preprocess)
end

function correct_mci_conv()
  correct_rois_for_covars(covars=[:age, :sex, :edu, :total_gray, :is_mci, :conv],
                          covar_preprocess=covar_preprocess)
end

corrections = Function[correct, correct_mci, correct_mci_conv]

# This approach would give us three different graphs and we might be able
# to say which edges are more likely to be disease specific.

# btw, 288 * 287 = 82,656

score_data = readtable("../data/animal_scores.csv")

for correction = corrections

  all_data = join(correction(), score_data, on=:id, kind=:inner)
  # Iterate through ROIs, checking to see if each ROI is associated with fluency
  # for gm_a in rois
  #    flu ~ beta1 * gm_a + covariates
  # end

  roi_cols::Vector{Symbol} = get_roi_cols(all_data, normalized=true)
  num_rois = length(roi_cols)

  roi_data::Matrix{Float64} = Matrix(all_data[roi_cols])

  flu::Vector{Float64} = Vector{Float64}(all_data[:raw])
  beta_flu::DataFrame = begin
    #flu = roi_data * beta_flus
    ret::Vector{Float64} = \(roi_data .- mean(roi_data, 1), flu - mean(flu))
    @assert length(ret) == num_rois
    ret_df = DataFrame(betacoef= ret)#, rois=roi_cols)
    #ret_df[sig_rois(ret_df), :]
  end

  # Use FDR to decide which ROIs to keep, put in list sig_rois
  # Also have to keep beta1 for each ROI.



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
